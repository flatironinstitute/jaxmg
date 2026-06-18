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
