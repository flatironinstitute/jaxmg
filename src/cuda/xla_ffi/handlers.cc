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
//      attributes, and the public Collectives extension.
//
// All CUDA kernels, NCCL transfers and cuSOLVERMp calls share the platform
// stream supplied by XLA. No StreamExecutor or XLA C++ context is required.
//
#include "runtime.h"
#include "../cusolvermp_routines/cusolvermp_routines.h"

namespace jaxmg {

// ---------------------------------------------------------------------------
// Fused solver handlers.
// ---------------------------------------------------------------------------
//
// Python-facing solver and decomposition routines register here. Each call
// performs local layout conversion, edge-padding compaction, 2D block-cyclic
// redistribution, cuSOLVERMp execution, reverse redistribution, and local
// layout restore inside one FFI dispatch.

// Every target registers the same prepare symbol: all of them borrow the one
// all-assigned P2P clique, which XLA must construct before runtime dispatch.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpPrepareFFI, XlaCusolverMpPrepare,
    ffi::Ffi::BindPrepare()
        .Attrs()
        .Ctx<ffi::Extension<Collectives>>());

// Registers the runtime POTRS target that receives padded JAX shards and
// returns the solved right-hand side plus a status vector.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpPotrsFFI, XlaCusolverMpPotrsDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("process_rows")
        .Attr<int64_t>("process_cols")
        .Attr<int64_t>("n")
        .Attr<int64_t>("nrhs")
        .Attr<int64_t>("b_distribution_cols")
        .Attr<int64_t>("tile_size")
        .Attr<absl::Span<const int64_t>>("partition_slots")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ctx<ffi::Extension<Collectives>>());

// Registers the optional logdet runtime target. The extra real scalar follows
// the matrix component precision and is replicated by a native NCCL all-reduce
// before the FFI call completes.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpPotrsLogdetFFI, XlaCusolverMpPotrsLogdetDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("process_rows")
        .Attr<int64_t>("process_cols")
        .Attr<int64_t>("n")
        .Attr<int64_t>("nrhs")
        .Attr<int64_t>("b_distribution_cols")
        .Attr<int64_t>("tile_size")
        .Attr<absl::Span<const int64_t>>("partition_slots")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ctx<ffi::Extension<Collectives>>());

// Registers the factor-only POTRF target that returns the lower Cholesky
// factor, aliased to A, plus a status vector.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpPotrfFFI, XlaCusolverMpPotrfDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("process_rows")
        .Attr<int64_t>("process_cols")
        .Attr<int64_t>("n")
        .Attr<int64_t>("tile_size")
        .Attr<absl::Span<const int64_t>>("partition_slots")
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ctx<ffi::Extension<Collectives>>());

// Inversion shares one target; the static mode selects Cholesky or pivoted LU.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpInverseFFI, XlaCusolverMpInverseDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("process_rows")
        .Attr<int64_t>("process_cols")
        .Attr<int64_t>("n")
        .Attr<int64_t>("tile_size")
        .Attr<bool>("cholesky")
        .Attr<absl::Span<const int64_t>>("partition_slots")
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ctx<ffi::Extension<Collectives>>());

// Registers the runtime LU-solve target that receives a general square matrix,
// factorizes it with GETRF, and solves the right-hand side with GETRS.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpLuSolveFFI, XlaCusolverMpLuSolveDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("process_rows")
        .Attr<int64_t>("process_cols")
        .Attr<int64_t>("n")
        .Attr<int64_t>("nrhs")
        .Attr<int64_t>("b_distribution_cols")
        .Attr<int64_t>("tile_size")
        .Attr<absl::Span<const int64_t>>("partition_slots")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ctx<ffi::Extension<Collectives>>());

// Registers the factor-only GETRF target that returns the packed LU factor,
// aliased to A, the replicated 0-based pivots, and a status vector.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpGetrfFFI, XlaCusolverMpGetrfDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("process_rows")
        .Attr<int64_t>("process_cols")
        .Attr<int64_t>("n")
        .Attr<int64_t>("tile_size")
        .Attr<absl::Span<const int64_t>>("partition_slots")
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ctx<ffi::Extension<Collectives>>());

// Registers the rectangular least-squares target. A is returned as opaque
// work storage; B retains the solution in its first N rows, with a separate
// N-row output restored directly to the solution's distribution.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpGelsFFI, XlaCusolverMpGelsDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("process_rows")
        .Attr<int64_t>("process_cols")
        .Attr<int64_t>("m")
        .Attr<int64_t>("n")
        .Attr<int64_t>("nrhs")
        .Attr<int64_t>("b_distribution_cols")
        .Attr<int64_t>("x_distribution_rows")
        .Attr<int64_t>("tile_size")
        .Attr<absl::Span<const int64_t>>("partition_slots")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ctx<ffi::Extension<Collectives>>());

// Registers reduced QR with explicit Q and R outputs.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpQrFFI, XlaCusolverMpQrDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("process_rows")
        .Attr<int64_t>("process_cols")
        .Attr<int64_t>("m")
        .Attr<int64_t>("n")
        .Attr<int64_t>("tile_size")
        .Attr<absl::Span<const int64_t>>("partition_slots")
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ctx<ffi::Extension<Collectives>>());

// Registers the runtime SYEVD target that returns eigenvalues, solver work
// storage, eigenvectors, and a status vector.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpSyevdFFI, XlaCusolverMpSyevdDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("process_rows")
        .Attr<int64_t>("process_cols")
        .Attr<int64_t>("n")
        .Attr<int64_t>("tile_size")
        .Attr<absl::Span<const int64_t>>("partition_slots")
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ctx<ffi::Extension<Collectives>>());

// Registers the eigenvalues-only SYEVD target. Omitting the eigenvector result
// prevents XLA from allocating an unused matrix-sized output.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpSyevdValuesFFI, XlaCusolverMpSyevdValuesDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("process_rows")
        .Attr<int64_t>("process_cols")
        .Attr<int64_t>("n")
        .Attr<int64_t>("tile_size")
        .Attr<absl::Span<const int64_t>>("partition_slots")
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ctx<ffi::Extension<Collectives>>());

// Registers GESVD with both left and right singular-vector outputs.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpGesvdUvFFI, XlaCusolverMpGesvdUvDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("process_rows")
        .Attr<int64_t>("process_cols")
        .Attr<int64_t>("m")
        .Attr<int64_t>("n")
        .Attr<int64_t>("tile_size")
        .Attr<int64_t>("full_matrices")
        .Attr<absl::Span<const int64_t>>("partition_slots")
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ctx<ffi::Extension<Collectives>>());

// Registers GESVD with left singular vectors only.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpGesvdUFFI, XlaCusolverMpGesvdUDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("process_rows")
        .Attr<int64_t>("process_cols")
        .Attr<int64_t>("m")
        .Attr<int64_t>("n")
        .Attr<int64_t>("tile_size")
        .Attr<int64_t>("full_matrices")
        .Attr<absl::Span<const int64_t>>("partition_slots")
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ctx<ffi::Extension<Collectives>>());

// Registers GESVD with right singular vectors only.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpGesvdVhFFI, XlaCusolverMpGesvdVhDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("process_rows")
        .Attr<int64_t>("process_cols")
        .Attr<int64_t>("m")
        .Attr<int64_t>("n")
        .Attr<int64_t>("tile_size")
        .Attr<int64_t>("full_matrices")
        .Attr<absl::Span<const int64_t>>("partition_slots")
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ctx<ffi::Extension<Collectives>>());

// Registers values-only GESVD. The shorter result list prevents allocation of
// either singular-vector matrix.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpGesvdValuesFFI, XlaCusolverMpGesvdValuesDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("process_rows")
        .Attr<int64_t>("process_cols")
        .Attr<int64_t>("m")
        .Attr<int64_t>("n")
        .Attr<int64_t>("tile_size")
        .Attr<int64_t>("full_matrices")
        .Attr<absl::Span<const int64_t>>("partition_slots")
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ctx<ffi::Extension<Collectives>>());

// Registers polar decomposition with both Up and H outputs.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpPolarUhFFI, XlaCusolverMpPolarUhDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("process_rows")
        .Attr<int64_t>("process_cols")
        .Attr<int64_t>("m")
        .Attr<int64_t>("n")
        .Attr<int64_t>("tile_size")
        .Attr<absl::Span<const int64_t>>("partition_slots")
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ctx<ffi::Extension<Collectives>>());

// Registers polar decomposition without the optional H output.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaCusolverMpPolarUFFI, XlaCusolverMpPolarUDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("process_rows")
        .Attr<int64_t>("process_cols")
        .Attr<int64_t>("m")
        .Attr<int64_t>("n")
        .Attr<int64_t>("tile_size")
        .Attr<absl::Span<const int64_t>>("partition_slots")
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ctx<ffi::Extension<Collectives>>());

}  // namespace jaxmg
