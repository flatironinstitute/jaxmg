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

#include <algorithm>

#include "../include/xla_comm_backend.h"

namespace xla::gpu {

absl::Status XlaCusolverMpPotrsPrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests) {
  return XlaCusolverMpDistributedPotrsProbePrepare(collective_params,
                                                  clique_requests);
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

  // A and B can have different logical column counts.  For B there are two
  // counts: `nrhs` is the real RHS width passed to cuSOLVERMp, while
  // `b_distribution_cols` is the JAX-visible routing width used to make skinny
  // RHS matrices shardable over process-grid columns.  Compute both scratch
  // requirements and allocate one reusable buffer large enough for every
  // redistribution stage in this FFI call.
  absl::StatusOr<int64_t> a_scratch_elements =
      RequiredPadded2DNativePlanScratchElements(
          process_rows, process_cols, tile_size, tile_size, n, n,
          a.dimensions()[0], a.dimensions()[1], rank_map);
  if (!a_scratch_elements.ok()) {
    return a_scratch_elements.status();
  }
  absl::StatusOr<int64_t> b_scratch_elements =
      RequiredPadded2DNativePlanScratchElements(
          process_rows, process_cols, tile_size, tile_size, n,
          b_distribution_cols, b.dimensions()[0], b.dimensions()[1], rank_map);
  if (!b_scratch_elements.ok()) {
    return b_scratch_elements.status();
  }
  const int64_t scratch_elements =
      std::max(*a_scratch_elements, *b_scratch_elements);
  const size_t element_bytes =
      a.size_bytes() / static_cast<size_t>(a.element_count());
  const size_t scratch_bytes =
      std::max<size_t>(1, static_cast<size_t>(scratch_elements)) *
      element_bytes;
  absl::StatusOr<void*> redistribution_scratch =
      AllocateFfiScratch(scratch, scratch_bytes,
                         "cusolvermp_potrs_redistribution");
  if (!redistribution_scratch.ok()) {
    return redistribution_scratch.status();
  }
  se::DeviceAddressBase scratch_base(*redistribution_scratch, scratch_bytes);

  // Move the padded JAX buffers into the local 2D block-cyclic layout expected
  // by cuSOLVERMp. The input/output aliases mean a_work and b_out are the
  // donated storage slots that survive across the solve call.
  JAXMG_RETURN_IF_ERROR(ExecutePadded2DNativePlanRaw(
      "cusolvermp_potrs/a_forward", stream, comm_stream, cuda_stream,
      process_rows, process_cols, tile_size, tile_size, n, n, /*reverse=*/0,
      rank_map, a, a_work->device_memory(), scratch_base, scratch_elements,
      collective_params, collective_cliques));
  JAXMG_RETURN_IF_ERROR(ExecutePadded2DNativePlanRaw(
      "cusolvermp_potrs/b_forward", stream, comm_stream, cuda_stream,
      process_rows, process_cols, tile_size, tile_size, n, b_distribution_cols,
      /*reverse=*/0, rank_map, b, b_out->device_memory(), scratch_base,
      scratch_elements, collective_params, collective_cliques));

  // The previous production prototype ran redistribution and cuSOLVERMp as
  // separate FFI calls, which gave XLA a hard sequencing boundary.  The fused
  // handler must provide the same safety explicitly before the host-side
  // cuSOLVERMp call reads the redistributed buffers.  This can be relaxed to an
  // event dependency after the correctness matrix is stable.
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));

  ffi::AnyBuffer a_cyclic = *a_work;
  ffi::AnyBuffer b_cyclic = *b_out;
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
  return ExecutePadded2DNativePlanRaw(
      "cusolvermp_potrs/b_reverse", stream, comm_stream, cuda_stream,
      process_rows, process_cols, tile_size, tile_size, n, b_distribution_cols,
      /*reverse=*/1, rank_map, b_solved_cyclic, b_out->device_memory(),
      scratch_base, scratch_elements, collective_params, collective_cliques);
}

}  // namespace xla::gpu
