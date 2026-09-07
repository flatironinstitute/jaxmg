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
// This file defines the native ABI used by Python registration in _setup.py.
// The prepare and execute symbol names must remain synchronized with those
// registrations.
//
// Registration workflow:
//   1. Python loads libjaxmg_xla_comm_backend.so with ctypes.
//   2. Python registers each FFI target by name and points JAX at the prepare
//      and execute symbols defined here.
//   3. XLA calls the prepare symbol during lowering so the backend can request
//      communicator cliques.
//   4. XLA calls the execute symbol at runtime with streams, buffers,
//      attributes, scratch allocators, and collective contexts.
//
// The execute bindings deliberately do NOT request
// `ffi::CommunicationStream<N>`. That indexes `CollectiveParams::async_streams`,
// whose size XLA derives from the async-collective thunks present in the
// surrounding module. These handlers are synchronous and emit none, so the pool
// is empty and the binding fails to decode. Through jax 0.11.0 a compatibility
// floor in `GetNumAdditionalStreams()` allocated the legacy pool for every
// executable, which is the only reason index 1 used to resolve; jax 0.11.1
// removed it, and XLA offers no way for a custom call to request one. The raw
// NCCL work therefore runs on the main compute stream: `ChooseNcclStream` in
// runtime.cc returns `uses_comm_stream=false` for a null `comm_stream`, and the
// cross-stream `WaitFor` pairs it guards are unnecessary there because ordering
// on a single stream is implicit.
//
#include "runtime.h"
#include "../cusolvermp_routines/cusolvermp_routines.h"

namespace xla::gpu {

// ---------------------------------------------------------------------------
// Fused solver handlers.
// ---------------------------------------------------------------------------
//
// Python-facing `potrs`, `lu_solve`, `syevd`, and `gesvd` register here. Each call
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

// Registers the optional logdet runtime target. The extra real scalar follows
// the matrix component precision and is replicated by a native NCCL all-reduce
// before the FFI call completes.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpPotrsLogdetFFI, XlaCusolverMpPotrsLogdetDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
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
        .Ret<ffi::AnyBuffer>()
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

// Registers the runtime SYEVD target that returns eigenvalues, solver work
// storage, eigenvectors, and a status vector.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpSyevdFFI, XlaCusolverMpSyevdDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
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

// Registers the eigenvalues-only SYEVD target. Omitting the eigenvector result
// prevents XLA from allocating an unused matrix-sized output.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpSyevdValuesFFI, XlaCusolverMpSyevdValuesDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
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
        .Ret<ffi::BufferR1<S32>>()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliques>());

// Registers the GESVD prepare target shared by all four output modes.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpGesvdPrepareFFI, XlaCusolverMpGesvdPrepare,
    ffi::Ffi::BindPrepare()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliqueRequests>());

// Registers GESVD with both left and right singular-vector outputs.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpGesvdUvFFI, XlaCusolverMpGesvdUvDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("process_rows")
        .Attr<int64_t>("process_cols")
        .Attr<int64_t>("m")
        .Attr<int64_t>("n")
        .Attr<int64_t>("tile_size")
        .Attr<int64_t>("grid_mapping")
        .Attr<int64_t>("full_matrices")
        .Attr<absl::Span<const int64_t>>("rank_map")
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::BufferR1<S32>>()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliques>());

// Registers GESVD with left singular vectors only.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpGesvdUFFI, XlaCusolverMpGesvdUDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("process_rows")
        .Attr<int64_t>("process_cols")
        .Attr<int64_t>("m")
        .Attr<int64_t>("n")
        .Attr<int64_t>("tile_size")
        .Attr<int64_t>("grid_mapping")
        .Attr<int64_t>("full_matrices")
        .Attr<absl::Span<const int64_t>>("rank_map")
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::BufferR1<S32>>()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliques>());

// Registers GESVD with right singular vectors only.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpGesvdVhFFI, XlaCusolverMpGesvdVhDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("process_rows")
        .Attr<int64_t>("process_cols")
        .Attr<int64_t>("m")
        .Attr<int64_t>("n")
        .Attr<int64_t>("tile_size")
        .Attr<int64_t>("grid_mapping")
        .Attr<int64_t>("full_matrices")
        .Attr<absl::Span<const int64_t>>("rank_map")
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::BufferR1<S32>>()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliques>());

// Registers values-only GESVD. The shorter result list prevents allocation of
// either singular-vector matrix.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpGesvdValuesFFI, XlaCusolverMpGesvdValuesDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("process_rows")
        .Attr<int64_t>("process_cols")
        .Attr<int64_t>("m")
        .Attr<int64_t>("n")
        .Attr<int64_t>("tile_size")
        .Attr<int64_t>("grid_mapping")
        .Attr<int64_t>("full_matrices")
        .Attr<absl::Span<const int64_t>>("rank_map")
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::BufferR1<S32>>()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliques>());

}  // namespace xla::gpu
