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
// cuSOLVERMp production and diagnostic FFI declarations.
//
// The compact production entry points live in cusolvermp_potrs.cc and
// cusolvermp_syevd.cc. The shared cuSOLVERMp loading, descriptor lifecycle, and
// diagnostic probes live in cusolvermp.cc until that large integration file is
// split further.

#ifndef JAXMG_CUSOLVERMP_ROUTINES_H_
#define JAXMG_CUSOLVERMP_ROUTINES_H_

#include "../memory_redist/memory_redist.h"

namespace xla::gpu {

// Basic cuSOLVERMp load/grid probe.  This verifies that the runtime can load
// libcusolverMp, borrow the XLA communicator, create a handle, and create a
// process grid without running a solver.
absl::Status XlaCusolverMpInitProbePrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests);
absl::Status XlaCusolverMpInitProbeDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    int64_t process_rows, int64_t process_cols, int64_t matrix_rows,
    int64_t matrix_cols, int64_t tile_rows, int64_t tile_cols,
    ffi::AnyBuffer token, ffi::Result<ffi::BufferR1<S32>> out,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques);

// cuSOLVERMp scatter-layout probe.  This uses NVIDIA's host-to-device scatter
// path to confirm JAXMg's rank mapping and local leading dimensions agree with
// cuSOLVERMp before testing the custom redistribution path.
absl::Status XlaCusolverMpScatterLayoutProbePrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests);
absl::Status XlaCusolverMpScatterLayoutProbeDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    int64_t process_rows, int64_t process_cols, int64_t logical_rows,
    int64_t logical_cols, int64_t tile_rows, int64_t tile_cols,
    ffi::AnyBuffer matrix, ffi::Result<ffi::AnyBuffer> matrix_out,
    ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques);

// Small self-contained solver probes.  These generate or scatter inputs inside
// native code and are useful for separating cuSOLVERMp/runtime issues from
// JAXMg redistribution issues.
absl::Status XlaCusolverMpPotrsProbePrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests);
absl::Status XlaCusolverMpPotrsProbeDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    int64_t process_rows, int64_t process_cols, int64_t n, int64_t tile_size,
    ffi::AnyBuffer a, ffi::AnyBuffer b, ffi::Result<ffi::AnyBuffer> a_out,
    ffi::Result<ffi::AnyBuffer> b_out, ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques);
absl::Status XlaCusolverMpDistributedPotrsProbePrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests);
absl::Status XlaCusolverMpDistributedPotrsProbeDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    int64_t process_rows, int64_t process_cols, int64_t n, int64_t nrhs,
    int64_t tile_size, ffi::AnyBuffer a, ffi::AnyBuffer b,
    ffi::Result<ffi::AnyBuffer> a_out, ffi::Result<ffi::AnyBuffer> b_out,
    ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques);
absl::Status XlaCusolverMpSyevdProbePrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests);
absl::Status XlaCusolverMpSyevdProbeDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    int64_t process_rows, int64_t process_cols, int64_t n, int64_t tile_size,
    int64_t grid_mapping, int64_t use_private_stream,
    ffi::AnyBuffer token, ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques);

// Shared cuSOLVERMp solve helpers.
//
// cusolvermp.cc owns the dynamic dlopen/dlsym boundary and the low-level
// cuSOLVERMp handle/grid/descriptor lifecycle. Production wrappers call these
// helpers after memory_redist has redistributed JAX buffers into local 2D
// block-cyclic form.
absl::Status CusolverMpDistributedPotrsDispatchImpl(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    int64_t process_rows, int64_t process_cols, int64_t n, int64_t nrhs,
    int64_t tile_size, int64_t grid_mapping,
    absl::Span<const int64_t> rank_map, ffi::AnyBuffer a, ffi::AnyBuffer b,
    ffi::Result<ffi::AnyBuffer> a_out, ffi::Result<ffi::AnyBuffer> b_out,
    ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques, bool validate_solution);
absl::Status CusolverMpSyevdDispatchImpl(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    int64_t process_rows, int64_t process_cols, int64_t n,
    int64_t tile_size, int64_t grid_mapping,
    absl::Span<const int64_t> rank_map, ffi::AnyBuffer a,
    ffi::Result<ffi::AnyBuffer> eigenvalues,
    ffi::Result<ffi::AnyBuffer> work, ffi::Result<ffi::AnyBuffer> vectors,
    ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques);

// Production fused solver entry points.  The wrappers own local layout
// conversion and redistribution orchestration; the shared helpers above own the
// cuSOLVERMp call once buffers are in 2D block-cyclic local storage.
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
