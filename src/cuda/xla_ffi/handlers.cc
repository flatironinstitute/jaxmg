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
// Exported XLA FFI symbols for the cuSOLVERMp backend.
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

#include "runtime.h"
#include "../cusolvermp_routines/cusolvermp_routines.h"

namespace xla::gpu {

// ---------------------------------------------------------------------------
// Production fused solver handlers.
// ---------------------------------------------------------------------------
//
// Python-facing `potrs`, `lu_solve`, and `syevd` register here. Each call
// performs local layout conversion, edge-padding compaction, 2D block-cyclic
// redistribution, cuSOLVERMp execution, reverse redistribution, and local
// layout restore inside one FFI dispatch.
// Registers the POTRS prepare target that asks XLA to construct the P2P
// communicator clique before runtime.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpPotrsPrepareFFI, XlaCusolverMpPotrsPrepare,
    ffi::Ffi::BindPrepare()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliqueRequests>());

// Registers the runtime POTRS target that receives padded JAX shards and
// returns the solved right-hand side plus a status vector.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpPotrsFFI, XlaCusolverMpPotrsDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::CommunicationStream<1>>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
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

// Registers the logdet POTRS prepare target. It requests the same communicator
// as ordinary POTRS because both cuSOLVERMp and the final scalar all-reduce use
// the all-assigned XLA-owned NCCL clique.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpPotrsLogdetPrepareFFI, XlaCusolverMpPotrsPrepare,
    ffi::Ffi::BindPrepare()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliqueRequests>());

// Registers the optional logdet runtime target. The extra float64 result is
// replicated by a native NCCL all-reduce before the FFI call completes.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpPotrsLogdetFFI, XlaCusolverMpPotrsLogdetDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::CommunicationStream<1>>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
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
        .Ret<ffi::BufferR1<F64>>()
        .Ret<ffi::BufferR1<S32>>()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliques>());

// Registers the LU-solve prepare target that asks XLA to construct the P2P
// communicator clique before runtime.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpLuSolvePrepareFFI, XlaCusolverMpLuSolvePrepare,
    ffi::Ffi::BindPrepare()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliqueRequests>());

// Registers the runtime LU-solve target that receives a general square matrix,
// factorizes it with GETRF, and solves the right-hand side with GETRS.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpLuSolveFFI, XlaCusolverMpLuSolveDispatch,
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

// Registers the SYEVD prepare target that requests the same all-assigned P2P
// communicator clique used by native redistribution and cuSOLVERMp.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpSyevdPrepareFFI, XlaCusolverMpSyevdPrepare,
    ffi::Ffi::BindPrepare()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliqueRequests>());

// Registers the runtime SYEVD target that returns eigenvalues, eigenvectors,
// solver work storage, and a status vector.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpSyevdFFI, XlaCusolverMpSyevdDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::CommunicationStream<1>>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
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
