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

#include <algorithm>

#include "../include/xla_comm_backend.h"

namespace xla::gpu {

absl::Status XlaCusolverMpSyevdPrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests) {
  return XlaCusolverMpDistributedPotrsProbePrepare(collective_params,
                                                  clique_requests);
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

  absl::StatusOr<int64_t> scratch_elements_or =
      RequiredPadded2DNativePlanScratchElements(
          process_rows, process_cols, tile_size, tile_size, n, n,
          a.dimensions()[0], a.dimensions()[1], rank_map);
  if (!scratch_elements_or.ok()) {
    return scratch_elements_or.status();
  }
  const int64_t scratch_elements = *scratch_elements_or;
  const size_t element_bytes =
      a.size_bytes() / static_cast<size_t>(a.element_count());
  const size_t scratch_bytes =
      std::max<size_t>(1, static_cast<size_t>(scratch_elements)) *
      element_bytes;
  absl::StatusOr<void*> redistribution_scratch =
      AllocateFfiScratch(scratch, scratch_bytes,
                         "cusolvermp_syevd_redistribution");
  if (!redistribution_scratch.ok()) {
    return redistribution_scratch.status();
  }
  se::DeviceAddressBase scratch_base(*redistribution_scratch, scratch_bytes);

  JAXMG_RETURN_IF_ERROR(ExecutePadded2DNativePlanRaw(
      "cusolvermp_syevd/a_forward", stream, comm_stream, cuda_stream,
      process_rows, process_cols, tile_size, tile_size, n, n, /*reverse=*/0,
      rank_map, a, work->device_memory(), scratch_base, scratch_elements,
      collective_params, collective_cliques));

  ffi::AnyBuffer a_cyclic = *work;
  JAXMG_RETURN_IF_ERROR(CusolverMpSyevdDispatchImpl(
      stream, comm_stream, cuda_stream, process_rows, process_cols, n,
      tile_size, grid_mapping, rank_map, a_cyclic, eigenvalues, work, vectors,
      status, collective_params, collective_cliques));

  ffi::AnyBuffer vectors_cyclic = *vectors;
  return ExecutePadded2DNativePlanRaw(
      "cusolvermp_syevd/vectors_reverse", stream, comm_stream,
      cuda_stream, process_rows, process_cols, tile_size, tile_size, n, n,
      /*reverse=*/1, rank_map, vectors_cyclic, vectors->device_memory(),
      scratch_base, scratch_elements, collective_params, collective_cliques);
}

}  // namespace xla::gpu
