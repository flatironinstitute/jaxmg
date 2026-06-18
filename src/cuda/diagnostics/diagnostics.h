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
// XLA communicator diagnostic FFI declarations.
//
// These probes are intentionally separate from the production solver API. They
// isolate clique construction, communicator lookup, all-reduce,
// CollectivePermute, and local rectangle addressing when the XLA integration
// needs debugging.

#ifndef JAXMG_DIAGNOSTICS_H_
#define JAXMG_DIAGNOSTICS_H_

#include "../include/xla_comm_common.h"

namespace xla::gpu {

// Metadata probe: returns local/global device ids and whether the resolved
// XLA communicator exposes a platform handle.
absl::Status XlaCommCollectiveProbePrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests);
absl::Status XlaCommCollectiveProbeDispatch(
    se::Stream* stream, ffi::AnyBuffer token,
    ffi::Result<ffi::BufferR1<S32>> out,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques);

// All-reduce probes.  The node-scoped version verifies local clique setup; the
// global version verifies the all-assigned communicator needed by production.
absl::Status XlaCommAllReduceProbePrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests);
absl::Status XlaCommAllReduceProbeDispatch(
    se::Stream* stream, se::Stream* comm_stream, ffi::BufferR1<U32> src,
    ffi::Result<ffi::BufferR1<U32>> dst,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques);
absl::Status XlaCommGlobalAllReduceProbePrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests);
absl::Status XlaCommGlobalAllReduceProbeDispatch(
    se::Stream* stream, se::Stream* comm_stream, ffi::BufferR1<U32> src,
    ffi::Result<ffi::BufferR1<U32>> dst,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques);

// Ring/shift/permutation probes for point-to-point collective behavior.
absl::Status XlaCommRingPermuteProbePrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests);
absl::Status XlaCommRingPermuteProbeDispatch(
    se::Stream* stream, se::Stream* comm_stream, ffi::BufferR1<U32> src,
    ffi::Result<ffi::BufferR1<U32>> dst,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques);
absl::Status XlaCommGlobalRingPermuteProbePrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests);
absl::Status XlaCommGlobalRingPermuteProbeDispatch(
    se::Stream* stream, se::Stream* comm_stream, ffi::BufferR1<U32> src,
    ffi::Result<ffi::BufferR1<U32>> dst,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques);
absl::Status XlaCommShiftPermuteProbePrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests);
absl::Status XlaCommShiftPermuteProbeDispatch(
    se::Stream* stream, se::Stream* comm_stream, int64_t shift,
    ffi::BufferR1<U32> src, ffi::Result<ffi::BufferR1<U32>> dst,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques);
absl::Status XlaCommPermuteProbePrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests);
absl::Status XlaCommPermuteProbeDispatch(
    se::Stream* stream, se::Stream* comm_stream,
    absl::Span<const int64_t> targets, ffi::BufferR1<U32> src,
    ffi::Result<ffi::BufferR1<U32>> dst,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques);

// Chunked permutation probe.  This validates the offset/count addressing used
// by lower-level movement tests without invoking the full matrix scheduler.
absl::Status XlaCommChunkPermuteProbePrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests);
absl::Status XlaCommChunkPermuteProbeDispatch(
    se::Stream* stream, se::Stream* comm_stream,
    absl::Span<const int64_t> targets,
    absl::Span<const int64_t> src_offsets,
    absl::Span<const int64_t> dst_offsets, int64_t count,
    ffi::BufferR1<U32> src, ffi::Result<ffi::BufferR1<U32>> dst,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques);

}  // namespace xla::gpu

#endif  // JAXMG_DIAGNOSTICS_H_
