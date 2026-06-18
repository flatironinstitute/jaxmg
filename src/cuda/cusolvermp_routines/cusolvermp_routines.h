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
// Production cuSOLVERMp FFI declarations.
//
// This header is the exported native ABI used by xla_ffi/handlers.cc.  It
// deliberately exposes only the user-facing fused solver entry points; shared
// cuSOLVERMp implementation details live in cusolvermp_common.h, and the full
// solver workflows live in cusolvermp_potrs.cc and cusolvermp_syevd.cc.

#ifndef JAXMG_CUSOLVERMP_ROUTINES_H_
#define JAXMG_CUSOLVERMP_ROUTINES_H_

#include "../memory_redist/memory_redist.h"

namespace xla::gpu {

absl::Status XlaCusolverMpPotrsPrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests);
absl::Status XlaCusolverMpPotrsDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    se::OwningScratchAllocator<> scratch, int64_t process_rows,
    int64_t process_cols, int64_t n, int64_t nrhs,
    int64_t b_distribution_cols, int64_t tile_size, int64_t grid_mapping,
    absl::Span<const int64_t> rank_map, ffi::AnyBuffer a, ffi::AnyBuffer b,
    ffi::Result<ffi::AnyBuffer> a_work, ffi::Result<ffi::AnyBuffer> b_out,
    ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques);

absl::Status XlaCusolverMpSyevdPrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests);
absl::Status XlaCusolverMpSyevdDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    se::OwningScratchAllocator<> scratch, int64_t process_rows,
    int64_t process_cols, int64_t n, int64_t tile_size,
    int64_t grid_mapping, absl::Span<const int64_t> rank_map, ffi::AnyBuffer a,
    ffi::Result<ffi::AnyBuffer> eigenvalues,
    ffi::Result<ffi::AnyBuffer> work, ffi::Result<ffi::AnyBuffer> vectors,
    ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques);

}  // namespace xla::gpu

#endif  // JAXMG_CUSOLVERMP_ROUTINES_H_
