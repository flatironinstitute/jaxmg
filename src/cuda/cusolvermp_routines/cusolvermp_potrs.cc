// Copyright 2026 JAXMg contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Production cuSOLVERMp Cholesky solve FFI handler.
//
// This file is intentionally small: it owns only the user-facing `potrs`
// native entry point. Shared cuSOLVERMp loading and handle/grid/descriptor
// lifecycle live in cusolvermp.cc, and the 2D redistribution scheduler lives in
// memory_redist/block_cyclic_2d.cc. Keeping the orchestration here makes the
// production path easy to audit:
//
//   1. Validate padded local JAX buffers and output aliases.
//   2. Ask the native redistribution planner for the required scratch size.
//   3. Allocate scratch from XLA's FFI scratch allocator.
//   4. Redistribute A and B from block-sharded JAX layout to cuSOLVERMp's local
//      2D block-cyclic layout.
//   5. Call the shared cuSOLVERMp `potrf`/`potrs` helper using the borrowed
//      XLA-owned NCCL communicator.
//   6. Reverse-redistribute the solved B buffer back to the JAX-facing layout.
//   7. Restore solved B from column-major local storage back to JAX's ordinary
//      row-major local storage before returning through the FFI alias contract.

#include <array>

#include "../include/xla_comm_backend.h"

namespace xla::gpu {

// Prepare is intentionally limited to communicator setup.  XLA constructs the
// all-assigned P2P clique during compilation; dispatch later retrieves the
// communicator and uses it for both redistribution and cuSOLVERMp.
absl::Status XlaCusolverMpPotrsPrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests) {
  return RequestAllAssignedP2PCommunicator(
      collective_params, clique_requests, "cusolvermp_potrs");
}

absl::Status XlaCusolverMpPotrsDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    se::OwningScratchAllocator<> scratch, int64_t process_rows,
    int64_t process_cols, int64_t n, int64_t nrhs,
    int64_t b_distribution_cols, int64_t tile_size, int64_t grid_mapping,
    absl::Span<const int64_t> rank_map, ffi::AnyBuffer a, ffi::AnyBuffer b,
    ffi::Result<ffi::AnyBuffer> a_work, ffi::Result<ffi::AnyBuffer> b_out,
    ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  // Stage 1: validate the local FFI buffers.  Python has already padded A/B to
  // the user-visible storage capacity needed by the requested process grid and
  // tile size; native code only checks the local contracts it will dereference.
  if (a.dimensions().size() != 2 || b.dimensions().size() != 2 ||
      a_work->dimensions().size() != 2 || b_out->dimensions().size() != 2) {
    return absl::InvalidArgumentError(
        "cusolvermp_potrs expects rank-2 A/B buffers");
  }
  if (a.element_type() != b.element_type() ||
      a.element_type() != a_work->element_type() ||
      b.element_type() != b_out->element_type()) {
    return absl::InvalidArgumentError(
        "cusolvermp_potrs requires matching A/B dtypes");
  }
  if (a.dimensions()[0] != a_work->dimensions()[0] ||
      a.dimensions()[1] != a_work->dimensions()[1] ||
      b.dimensions()[0] != b_out->dimensions()[0] ||
      b.dimensions()[1] != b_out->dimensions()[1]) {
    return absl::InvalidArgumentError(
        "cusolvermp_potrs input/output shapes must match");
  }
  if (a.dimensions()[0] != b.dimensions()[0]) {
    return absl::InvalidArgumentError(
        "cusolvermp_potrs expects A and B to have matching local row "
        "capacity after padding");
  }
  if (b_distribution_cols < nrhs) {
    return absl::InvalidArgumentError(
        "cusolvermp_potrs requires b_distribution_cols >= nrhs");
  }

  // Stage 2: size one reusable scratch allocation for the whole fused call.
  // The memory_redist layer owns the formula so the solver wrapper does not
  // duplicate layout-conversion, edge-padding, or 2D cyclic scratch rules.
  // A and B can have different logical column counts. For B there are two
  // counts: `nrhs` is the real RHS width passed to cuSOLVERMp, while
  // `b_distribution_cols` is the JAX-visible routing width used to make skinny
  // RHS matrices shardable over process-grid columns. Describe both native
  // redistribution requests and let memory_redist allocate one reusable scratch
  // buffer large enough for every layout-conversion, padding, and block-cyclic
  // movement in this fused FFI call.
  const size_t element_bytes =
      a.size_bytes() / static_cast<size_t>(a.element_count());
  const std::array<Padded2DRedistScratchRequest, 2> scratch_requests = {{
      Padded2DRedistScratchRequest{
          /*process_rows=*/process_rows,
          /*process_cols=*/process_cols,
          /*tile_rows=*/tile_size,
          /*tile_cols=*/tile_size,
          /*logical_rows=*/n,
          /*logical_cols=*/n,
          /*local_rows=*/a.dimensions()[0],
          /*local_cols=*/a.dimensions()[1],
          /*rank_map=*/rank_map,
      },
      Padded2DRedistScratchRequest{
          /*process_rows=*/process_rows,
          /*process_cols=*/process_cols,
          /*tile_rows=*/tile_size,
          /*tile_cols=*/tile_size,
          /*logical_rows=*/n,
          /*logical_cols=*/b_distribution_cols,
          /*local_rows=*/b.dimensions()[0],
          /*local_cols=*/b.dimensions()[1],
          /*rank_map=*/rank_map,
      },
  }};
  absl::StatusOr<Padded2DRedistScratch> redistribution_scratch =
      AllocatePadded2DRedistScratch(
          scratch, element_bytes, absl::MakeConstSpan(scratch_requests),
          "cusolvermp_potrs_redistribution");
  if (!redistribution_scratch.ok()) {
    return redistribution_scratch.status();
  }
  se::DeviceAddressBase scratch_base = redistribution_scratch->base;
  const int64_t scratch_elements = redistribution_scratch->elements;

  ffi::AnyBuffer a_forward_input = a;
  ffi::AnyBuffer b_forward_input = b;
  // Stage 3: make the local buffer storage convention match cuSOLVERMp.  JAX
  // enters with row-major local shards; cuSOLVERMp descriptors interpret local
  // memory as column-major.  This in-place conversion avoids XLA's full-size
  // layout-copy allocation. When XLA aliases input/output buffers the copy is a
  // no-op; otherwise the explicit copy preserves correctness without a second
  // native matrix allocation.
  JAXMG_RETURN_IF_ERROR(CopyMatrixIfNeeded(cuda_stream, a, a_work));
  JAXMG_RETURN_IF_ERROR(CopyMatrixIfNeeded(cuda_stream, b, b_out));
  a_forward_input = *a_work;
  b_forward_input = *b_out;
  JAXMG_RETURN_IF_ERROR(ConvertRowMajorToColumnMajorInPlace(
      cuda_stream, "cusolvermp_potrs/a_layout_convert", a_forward_input,
      scratch_base, scratch_elements));
  JAXMG_RETURN_IF_ERROR(ConvertRowMajorToColumnMajorInPlace(
      cuda_stream, "cusolvermp_potrs/b_layout_convert", b_forward_input,
      scratch_base, scratch_elements));

  // Stage 4: move local shards into cuSOLVERMp's 2D block-cyclic distribution.
  // A is routed over the full n x n logical matrix; B is routed over the padded
  // distribution width but only the first nrhs columns are solved.
  JAXMG_RETURN_IF_ERROR(ExecutePadded2DNativePlanRaw(
      "cusolvermp_potrs/a_forward", stream, comm_stream, cuda_stream,
      process_rows, process_cols, tile_size, tile_size, n, n, /*reverse=*/0,
      rank_map, a_forward_input, a_work->device_memory(), scratch_base,
      scratch_elements, collective_params, collective_cliques));
  JAXMG_RETURN_IF_ERROR(ExecutePadded2DNativePlanRaw(
      "cusolvermp_potrs/b_forward", stream, comm_stream, cuda_stream,
      process_rows, process_cols, tile_size, tile_size, n, b_distribution_cols,
      /*reverse=*/0, rank_map, b_forward_input, b_out->device_memory(),
      scratch_base, scratch_elements, collective_params, collective_cliques));

  // The previous production prototype ran redistribution and cuSOLVERMp as
  // separate FFI calls, which gave XLA a hard sequencing boundary.  The fused
  // handler must provide the same safety explicitly before the host-side
  // cuSOLVERMp call reads the redistributed buffers.  This can be relaxed to an
  // event dependency after the correctness matrix is stable.
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));

  ffi::AnyBuffer a_cyclic = *a_work;
  ffi::AnyBuffer b_cyclic = *b_out;
  // Stage 5: run potrf/potrs on the redistributed buffers.  The shared helper
  // handles the cuSOLVERMp ABI boundary and borrowed NCCL communicator.
  JAXMG_RETURN_IF_ERROR(CusolverMpDistributedPotrsDispatchImpl(
      stream, comm_stream, cuda_stream, process_rows, process_cols, n, nrhs,
      tile_size, grid_mapping, rank_map, a_cyclic, b_cyclic, a_work, b_out,
      status, collective_params, collective_cliques,
      /*validate_solution=*/false));

  // cuSOLVERMp is issued on the same stream, but keep the fused reverse
  // redistribution boundary explicit for the same reason as the forward solve
  // boundary above.
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));

  ffi::AnyBuffer b_solved_cyclic = *b_out;
  // Stage 6: return only B to the JAX-facing distribution.  A is a factorized
  // work buffer after potrf and is not part of the public result.
  JAXMG_RETURN_IF_ERROR(ExecutePadded2DNativePlanRaw(
      "cusolvermp_potrs/b_reverse", stream, comm_stream, cuda_stream,
      process_rows, process_cols, tile_size, tile_size, n, b_distribution_cols,
      /*reverse=*/1, rank_map, b_solved_cyclic, b_out->device_memory(),
      scratch_base, scratch_elements, collective_params, collective_cliques));

  // Stage 7: restore the local storage convention for the user-visible output.
  JAXMG_RETURN_IF_ERROR(ConvertColumnMajorToRowMajorInPlace(
      cuda_stream, "cusolvermp_potrs/b_layout_restore", *b_out, scratch_base,
      scratch_elements));
  return absl::OkStatus();
}

}  // namespace xla::gpu
