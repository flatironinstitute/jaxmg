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
// Exported XLA FFI symbols for the XLA communicator cuSOLVERMp backend.
//
// Keeping the bindings in one translation unit makes Python registration names
// easy to audit and keeps solver implementation files focused on execution
// logic.
//
// File workflow:
//   1. Python loads libjaxmg_xla_comm_backend.so with ctypes.
//   2. Python registers each FFI target by name and points JAX at the prepare
//      and execute symbols defined here.
//   3. XLA calls the prepare symbol during lowering so the backend can request
//      communicator cliques.
//   4. XLA calls the execute symbol at runtime with streams, buffers,
//      attributes, scratch allocators, and collective contexts.
//
// The names in this file are therefore the ABI between Python registration and
// the native implementation. Renaming one requires the matching _setup.py entry
// to change as well.

#include "include/xla_comm_backend.h"

namespace xla::gpu {

// ---------------------------------------------------------------------------
// XLA communicator diagnostics.
// ---------------------------------------------------------------------------
//
// These handlers validate the pieces below the redistribution layer: collective
// prepare/dispatch plumbing, communicator lookup, all-reduce, and simple
// point-to-point CollectivePermute behavior.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCommCollectiveProbePrepareFFI, XlaCommCollectiveProbePrepare,
    ffi::Ffi::BindPrepare()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliqueRequests>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCommCollectiveProbeFFI, XlaCommCollectiveProbeDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::BufferR1<S32>>()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliques>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCommAllReduceProbePrepareFFI, XlaCommAllReduceProbePrepare,
    ffi::Ffi::BindPrepare()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliqueRequests>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCommAllReduceProbeFFI, XlaCommAllReduceProbeDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::CommunicationStream<0>>()
        .Arg<ffi::BufferR1<U32>>()
        .Ret<ffi::BufferR1<U32>>()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliques>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCommGlobalAllReduceProbePrepareFFI,
    XlaCommGlobalAllReduceProbePrepare,
    ffi::Ffi::BindPrepare()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliqueRequests>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCommGlobalAllReduceProbeFFI, XlaCommGlobalAllReduceProbeDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::CommunicationStream<0>>()
        .Arg<ffi::BufferR1<U32>>()
        .Ret<ffi::BufferR1<U32>>()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliques>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCommRingPermuteProbePrepareFFI, XlaCommRingPermuteProbePrepare,
    ffi::Ffi::BindPrepare()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliqueRequests>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCommRingPermuteProbeFFI, XlaCommRingPermuteProbeDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::CommunicationStream<1>>()
        .Arg<ffi::BufferR1<U32>>()
        .Ret<ffi::BufferR1<U32>>()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliques>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCommGlobalRingPermuteProbePrepareFFI,
    XlaCommGlobalRingPermuteProbePrepare,
    ffi::Ffi::BindPrepare()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliqueRequests>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCommGlobalRingPermuteProbeFFI, XlaCommGlobalRingPermuteProbeDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::CommunicationStream<1>>()
        .Arg<ffi::BufferR1<U32>>()
        .Ret<ffi::BufferR1<U32>>()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliques>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCommShiftPermuteProbePrepareFFI, XlaCommShiftPermuteProbePrepare,
    ffi::Ffi::BindPrepare()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliqueRequests>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCommShiftPermuteProbeFFI, XlaCommShiftPermuteProbeDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::CommunicationStream<1>>()
        .Attr<int64_t>("shift")
        .Arg<ffi::BufferR1<U32>>()
        .Ret<ffi::BufferR1<U32>>()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliques>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCommPermuteProbePrepareFFI, XlaCommPermuteProbePrepare,
    ffi::Ffi::BindPrepare()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliqueRequests>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCommPermuteProbeFFI, XlaCommPermuteProbeDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::CommunicationStream<1>>()
        .Attr<absl::Span<const int64_t>>("targets")
        .Arg<ffi::BufferR1<U32>>()
        .Ret<ffi::BufferR1<U32>>()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliques>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCommChunkPermuteProbePrepareFFI, XlaCommChunkPermuteProbePrepare,
    ffi::Ffi::BindPrepare()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliqueRequests>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCommChunkPermuteProbeFFI, XlaCommChunkPermuteProbeDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::CommunicationStream<1>>()
        .Attr<absl::Span<const int64_t>>("targets")
        .Attr<absl::Span<const int64_t>>("src_offsets")
        .Attr<absl::Span<const int64_t>>("dst_offsets")
        .Attr<int64_t>("count")
        .Arg<ffi::BufferR1<U32>>()
        .Ret<ffi::BufferR1<U32>>()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliques>());

// ---------------------------------------------------------------------------
// Rectangle and redistribution diagnostics.
// ---------------------------------------------------------------------------
//
// These handlers exercise local rectangle pack/unpack, one-hop raw NCCL
// rectangle movement, and the native 2D redistribution planners without calling
// cuSOLVERMp.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaRectPackUnpackProbePrepareFFI, XlaRectPackUnpackProbePrepare,
    ffi::Ffi::BindPrepare());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaRectPackUnpackProbeFFI, XlaRectPackUnpackProbeDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("row_start")
        .Attr<int64_t>("col_start")
        .Attr<int64_t>("row_count")
        .Attr<int64_t>("col_count")
        .Attr<int64_t>("target_row")
        .Attr<int64_t>("target_col")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaRectTransferProbePrepareFFI, XlaRectTransferProbePrepare,
    ffi::Ffi::BindPrepare()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliqueRequests>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaRectTransferProbeFFI, XlaRectTransferProbeDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::CommunicationStream<1>>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<absl::Span<const int64_t>>("targets")
        .Attr<absl::Span<const int64_t>>("src_row_starts")
        .Attr<absl::Span<const int64_t>>("src_col_starts")
        .Attr<absl::Span<const int64_t>>("dst_row_starts")
        .Attr<absl::Span<const int64_t>>("dst_col_starts")
        .Attr<int64_t>("row_count")
        .Attr<int64_t>("col_count")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliques>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaRect2DNativePlanPrepareFFI, XlaRect2DNativePlanPrepare,
    ffi::Ffi::BindPrepare()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliqueRequests>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaRect2DNativePlanFFI, XlaRect2DNativePlanDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::CommunicationStream<1>>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("process_rows")
        .Attr<int64_t>("process_cols")
        .Attr<int64_t>("tile_rows")
        .Attr<int64_t>("tile_cols")
        .Attr<absl::Span<const int64_t>>("rank_map")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliques>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaRectPadded2DNativePlanPrepareFFI, XlaRectPadded2DNativePlanPrepare,
    ffi::Ffi::BindPrepare()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliqueRequests>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaRectPadded2DNativePlanFFI, XlaRectPadded2DNativePlanDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::CommunicationStream<1>>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("process_rows")
        .Attr<int64_t>("process_cols")
        .Attr<int64_t>("tile_rows")
        .Attr<int64_t>("tile_cols")
        .Attr<int64_t>("logical_rows")
        .Attr<int64_t>("logical_cols")
        .Attr<int64_t>("reverse")
        .Attr<absl::Span<const int64_t>>("rank_map")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliques>());

// ---------------------------------------------------------------------------
// cuSOLVERMp diagnostics.
// ---------------------------------------------------------------------------
//
// These probes verify dynamic cuSOLVERMp loading, descriptor construction,
// scatter-layout agreement, and small direct solver calls. They remain separate
// from production wrappers so failures can be localized.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpInitProbePrepareFFI, XlaCusolverMpInitProbePrepare,
    ffi::Ffi::BindPrepare()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliqueRequests>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpInitProbeFFI, XlaCusolverMpInitProbeDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::CommunicationStream<1>>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("process_rows")
        .Attr<int64_t>("process_cols")
        .Attr<int64_t>("matrix_rows")
        .Attr<int64_t>("matrix_cols")
        .Attr<int64_t>("tile_rows")
        .Attr<int64_t>("tile_cols")
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::BufferR1<S32>>()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliques>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpScatterLayoutProbePrepareFFI,
    XlaCusolverMpScatterLayoutProbePrepare,
    ffi::Ffi::BindPrepare()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliqueRequests>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpScatterLayoutProbeFFI,
    XlaCusolverMpScatterLayoutProbeDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::CommunicationStream<1>>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("process_rows")
        .Attr<int64_t>("process_cols")
        .Attr<int64_t>("logical_rows")
        .Attr<int64_t>("logical_cols")
        .Attr<int64_t>("tile_rows")
        .Attr<int64_t>("tile_cols")
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::BufferR1<S32>>()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliques>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpPotrsProbePrepareFFI, XlaCusolverMpPotrsProbePrepare,
    ffi::Ffi::BindPrepare()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliqueRequests>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpPotrsProbeFFI,
    XlaCusolverMpPotrsProbeDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::CommunicationStream<1>>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("process_rows")
        .Attr<int64_t>("process_cols")
        .Attr<int64_t>("n")
        .Attr<int64_t>("tile_size")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::BufferR1<S32>>()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliques>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpDistributedPotrsProbePrepareFFI,
    XlaCusolverMpDistributedPotrsProbePrepare,
    ffi::Ffi::BindPrepare()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliqueRequests>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpDistributedPotrsProbeFFI,
    XlaCusolverMpDistributedPotrsProbeDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::CommunicationStream<1>>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("process_rows")
        .Attr<int64_t>("process_cols")
        .Attr<int64_t>("n")
        .Attr<int64_t>("nrhs")
        .Attr<int64_t>("tile_size")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::BufferR1<S32>>()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliques>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpSyevdProbePrepareFFI,
    XlaCusolverMpSyevdProbePrepare,
    ffi::Ffi::BindPrepare()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliqueRequests>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpSyevdProbeFFI,
    XlaCusolverMpSyevdProbeDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::CommunicationStream<1>>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("process_rows")
        .Attr<int64_t>("process_cols")
        .Attr<int64_t>("n")
        .Attr<int64_t>("tile_size")
        .Attr<int64_t>("grid_mapping")
        .Attr<int64_t>("use_private_stream")
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::BufferR1<S32>>()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliques>());

// ---------------------------------------------------------------------------
// Production fused solver handlers.
// ---------------------------------------------------------------------------
//
// Python-facing `potrs` and `syevd` register here. Each call performs local
// layout conversion, edge-padding compaction, 2D block-cyclic redistribution,
// cuSOLVERMp execution, reverse redistribution, and local layout restore inside
// one FFI dispatch.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpPotrsPrepareFFI, XlaCusolverMpPotrsPrepare,
    ffi::Ffi::BindPrepare()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliqueRequests>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpPotrsFFI, XlaCusolverMpPotrsDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::CommunicationStream<1>>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Attr<int64_t>("process_rows")
        .Attr<int64_t>("process_cols")
        .Attr<int64_t>("n")
        .Attr<int64_t>("nrhs")
        .Attr<int64_t>("b_distribution_cols")
        .Attr<int64_t>("tile_size")
        .Attr<int64_t>("grid_mapping")
        .Attr<absl::Span<const int64_t>>("rank_map")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::BufferR1<S32>>()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliques>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpSyevdPrepareFFI, XlaCusolverMpSyevdPrepare,
    ffi::Ffi::BindPrepare()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliqueRequests>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpSyevdFFI, XlaCusolverMpSyevdDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::CommunicationStream<1>>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Attr<int64_t>("process_rows")
        .Attr<int64_t>("process_cols")
        .Attr<int64_t>("n")
        .Attr<int64_t>("tile_size")
        .Attr<int64_t>("grid_mapping")
        .Attr<absl::Span<const int64_t>>("rank_map")
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::BufferR1<S32>>()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliques>());

}  // namespace xla::gpu
