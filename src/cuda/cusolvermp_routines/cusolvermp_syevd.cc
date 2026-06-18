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
// Production cuSOLVERMp symmetric/Hermitian eigensolver FFI handler.
//
// This file owns only the vector-producing `syevd` orchestration. JAXMg does
// not expose a no-eigenvector mode: cuSOLVERMp 0.7.x/0.8.x validation has been
// focused on the `compz='Z'` path used by NVIDIA's distributed sample, and this
// package needs eigenvectors for its intended workflows.
//
// File workflow:
//   1. Validate the padded local JAX matrix and output aliases.
//   2. Allocate one native scratch buffer for redistribution.
//   3. Redistribute A from block-sharded JAX layout to local 2D block-cyclic
//      cuSOLVERMp layout.
//   4. Call the shared cuSOLVERMp SYEVD helper using the borrowed XLA-owned
//      NCCL communicator.
//   5. Reverse-redistribute eigenvectors back to the original JAX-facing
//      block-sharded layout. Eigenvalues are replicated by cuSOLVERMp.
//   6. Restore eigenvectors from column-major local storage back to JAX's
//      ordinary row-major local storage before returning through the FFI
//      contract.

#include <array>

#include "../include/xla_comm_backend.h"

namespace xla::gpu {

absl::Status XlaCusolverMpSyevdPrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests) {
  return RequestAllAssignedP2PCommunicator(
      collective_params, clique_requests, "cusolvermp_syevd");
}

absl::Status XlaCusolverMpSyevdDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    se::OwningScratchAllocator<> scratch, int64_t process_rows,
    int64_t process_cols, int64_t n, int64_t tile_size,
    int64_t grid_mapping, absl::Span<const int64_t> rank_map, ffi::AnyBuffer a,
    ffi::Result<ffi::AnyBuffer> eigenvalues,
    ffi::Result<ffi::AnyBuffer> work, ffi::Result<ffi::AnyBuffer> vectors,
    ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  if (a.dimensions().size() != 2 || work->dimensions().size() != 2 ||
      vectors->dimensions().size() != 2) {
    return absl::InvalidArgumentError(
        "cusolvermp_syevd expects rank-2 matrix buffers");
  }
  if (a.element_type() != work->element_type() ||
      a.element_type() != vectors->element_type()) {
    return absl::InvalidArgumentError(
        "cusolvermp_syevd requires matching matrix dtypes");
  }
  if (a.dimensions()[0] != work->dimensions()[0] ||
      a.dimensions()[1] != work->dimensions()[1] ||
      a.dimensions()[0] != vectors->dimensions()[0] ||
      a.dimensions()[1] != vectors->dimensions()[1]) {
    return absl::InvalidArgumentError(
        "cusolvermp_syevd input/output matrix shapes must match");
  }

  const size_t element_bytes =
      a.size_bytes() / static_cast<size_t>(a.element_count());
  const std::array<Padded2DRedistScratchRequest, 1> scratch_requests = {{
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
  }};
  absl::StatusOr<Padded2DRedistScratch> redistribution_scratch =
      AllocatePadded2DRedistScratch(
          scratch, element_bytes, absl::MakeConstSpan(scratch_requests),
          "cusolvermp_syevd_redistribution");
  if (!redistribution_scratch.ok()) {
    return redistribution_scratch.status();
  }
  se::DeviceAddressBase scratch_base = redistribution_scratch->base;
  const int64_t scratch_elements = redistribution_scratch->elements;

  ffi::AnyBuffer a_forward_input = a;
  // Python supplies row-major JAX local shards. Convert the donated work alias
  // in-place to cuSOLVERMp's column-major local storage before the tile
  // redistribution. When XLA aliases input/output buffers this copy is a
  // no-op; otherwise the work output remains the single full-size storage slot
  // used by the fused call.
  JAXMG_RETURN_IF_ERROR(CopyMatrixIfNeeded(cuda_stream, a, work));
  a_forward_input = *work;
  JAXMG_RETURN_IF_ERROR(ConvertRowMajorToColumnMajorInPlace(
      cuda_stream, "cusolvermp_syevd/a_layout_convert", a_forward_input,
      scratch_base, scratch_elements));

  JAXMG_RETURN_IF_ERROR(ExecutePadded2DNativePlanRaw(
      "cusolvermp_syevd/a_forward", stream, comm_stream, cuda_stream,
      process_rows, process_cols, tile_size, tile_size, n, n, /*reverse=*/0,
      rank_map, a_forward_input, work->device_memory(), scratch_base,
      scratch_elements, collective_params, collective_cliques));

  // Preserve the old split-call sequencing semantics inside the fused handler:
  // the host-side cuSOLVERMp call must not observe the work buffer until all
  // pack/NCCL/unpack redistribution work is complete.
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));

  ffi::AnyBuffer a_cyclic = *work;
  JAXMG_RETURN_IF_ERROR(CusolverMpSyevdDispatchImpl(
      stream, comm_stream, cuda_stream, process_rows, process_cols, n,
      tile_size, grid_mapping, rank_map, a_cyclic, eigenvalues, work, vectors,
      status, collective_params, collective_cliques));

  // SYEVD writes the eigenvectors in cuSOLVERMp layout.  Keep the reverse
  // redistribution after a completed solver call, mirroring the old separate
  // FFI-call pipeline.
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));

  ffi::AnyBuffer vectors_cyclic = *vectors;
  JAXMG_RETURN_IF_ERROR(ExecutePadded2DNativePlanRaw(
      "cusolvermp_syevd/vectors_reverse", stream, comm_stream,
      cuda_stream, process_rows, process_cols, tile_size, tile_size, n, n,
      /*reverse=*/1, rank_map, vectors_cyclic, vectors->device_memory(),
      scratch_base, scratch_elements, collective_params, collective_cliques));

  JAXMG_RETURN_IF_ERROR(ConvertColumnMajorToRowMajorInPlace(
      cuda_stream, "cusolvermp_syevd/vectors_layout_restore", *vectors,
      scratch_base, scratch_elements));
  return absl::OkStatus();
}

}  // namespace xla::gpu
