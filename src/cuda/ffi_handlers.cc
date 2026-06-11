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
// Exported XLA FFI symbols for the XLA communicator cuSolverMg backend.
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

// Diagnostic communicator handlers. These are optional Python-facing probes and
// do not call cuSolverMg.
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
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliques>());

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

// Redistribution handlers. The step and batch versions are diagnostics; the
// native-plan handler is also called by Python's cyclic_1d helper.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCommMatrixColumnStepPrepareFFI, XlaCommMatrixColumnStepPrepare,
    ffi::Ffi::BindPrepare()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliqueRequests>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCommMatrixColumnStepFFI, XlaCommMatrixColumnStepDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::CommunicationStream<1>>()
        .Attr<int64_t>("kind")
        .Attr<int64_t>("source_rank")
        .Attr<int64_t>("target_rank")
        .Attr<int64_t>("source_col")
        .Attr<int64_t>("target_col")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliques>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCommMatrixColumnBatchPrepareFFI, XlaCommMatrixColumnBatchPrepare,
    ffi::Ffi::BindPrepare()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliqueRequests>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCommMatrixColumnBatchFFI, XlaCommMatrixColumnBatchDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::CommunicationStream<1>>()
        .Attr<absl::Span<const int64_t>>("kinds")
        .Attr<absl::Span<const int64_t>>("source_ranks")
        .Attr<absl::Span<const int64_t>>("target_ranks")
        .Attr<absl::Span<const int64_t>>("source_cols")
        .Attr<absl::Span<const int64_t>>("target_cols")
        .Attr<absl::Span<const int64_t>>("scratch_slots")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliques>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCommMatrixColumnNativePlanPrepareFFI, XlaCommMatrixColumnBatchPrepare,
    ffi::Ffi::BindPrepare()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliqueRequests>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCommMatrixColumnNativePlanFFI, XlaCommMatrixColumnNativePlanDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::CommunicationStream<1>>()
        .Attr<int64_t>("tile_size")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliques>());

// Production cuSolverMg handlers registered under the historical Python names
// potrs_mg, potri_mg, syevd_mg, and syevd_no_V_mg. The Python API can keep the
// old names while the implementation switches from the legacy CUDA peer
// shuffler to the XLA communicator backend.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCommPotrsMgNativePlanPrepareFFI, XlaCommMatrixColumnBatchPrepare,
    ffi::Ffi::BindPrepare()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliqueRequests>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCommPotrsMgNativePlanFFI, XlaCommPotrsMgNativePlanDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::CommunicationStream<1>>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Attr<int64_t>("T_A")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::BufferR1<S32>>()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliques>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCommPotriMgNativePlanPrepareFFI, XlaCommMatrixColumnBatchPrepare,
    ffi::Ffi::BindPrepare()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliqueRequests>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCommPotriMgNativePlanFFI, XlaCommPotriMgNativePlanDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::CommunicationStream<1>>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Attr<int64_t>("T_A")
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::BufferR1<S32>>()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliques>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCommSyevdMgNativePlanPrepareFFI, XlaCommMatrixColumnBatchPrepare,
    ffi::Ffi::BindPrepare()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliqueRequests>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCommSyevdMgNativePlanFFI, XlaCommSyevdMgNativePlanDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::CommunicationStream<1>>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Attr<int64_t>("T_A")
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::BufferR1<S32>>()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliques>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCommSyevdNoVMgNativePlanPrepareFFI, XlaCommMatrixColumnBatchPrepare,
    ffi::Ffi::BindPrepare()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliqueRequests>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCommSyevdNoVMgNativePlanFFI, XlaCommSyevdNoVMgNativePlanDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::CommunicationStream<1>>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Attr<int64_t>("T_A")
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::BufferR1<S32>>()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliques>());

}  // namespace xla::gpu
