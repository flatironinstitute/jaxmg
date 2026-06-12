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
// Shared declarations for the XLA communicator native backend.
//
// This backend is built by Bazel inside the pinned OpenXLA source tree. XLA
// collective FFI contexts depend on generated XLA protobufs and runtime
// libraries that are already described by XLA's Bazel build graph.
//
// Header layout:
//   1. Error and dtype helpers used by every translation unit.
//   2. Small process-local state structs used to bridge concurrent FFI calls
//      into one cuSolverMg host invocation in SPMD mode.
//   3. XLA communicator clique helpers and the rank-0 broadcast utility.
//   4. 1D redistribution entry points.
//   5. 2D rectangle/redistribution entry points used by the cuSOLVERMp path.
//   6. cuSOLVERMp diagnostic and production potrs entry points.
//   7. Fused cuSolverMg production solver handlers registered under the
//      historical JAXMg FFI target names.
//
// The 1D cuSolverMg production path is:
//   Python wrapper -> FFI handler -> XLA communicator lookup -> 1D cyclic
//   reshuffle -> cuSolverMg host call -> optional broadcast/reverse reshuffle.
//
// The first cuSOLVERMp production path is:
//   Python wrapper -> local shard padding -> native 2D redistribution ->
//   cuSOLVERMp potrf/potrs over borrowed NCCL communicator -> reverse native
//   2D redistribution -> local unpadding.

#ifndef JAXMG_XLA_COMM_BACKEND_H_
#define JAXMG_XLA_COMM_BACKEND_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include <cuComplex.h>
#include <cuda_runtime_api.h>
#include <cusolverMg.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_execution.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/ffi/ffi.h"
#include "xla/future.h"
#include "xla/runtime/device_id.h"
#include "xla/service/computation_placer.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace ffi = ::xla::ffi;

// Convert CUDA/cuSolverMg errors into absl::Status so all FFI handlers can
// return failures through the same mechanism used by XLA.
absl::Status CudaToStatus(cudaError_t err, const char* file, int line);
absl::Status CusolverToStatus(cusolverStatus_t err, const char* file,
                              int line);

#define JAXMG_RETURN_IF_CUDA_ERROR(expr)                              \
  do {                                                                \
    absl::Status _jaxmg_cuda_status =                                 \
        CudaToStatus((expr), __FILE__, __LINE__);                     \
    if (!_jaxmg_cuda_status.ok()) return _jaxmg_cuda_status;          \
  } while (0)

#define JAXMG_RETURN_IF_CUSOLVER_ERROR(expr)                          \
  do {                                                                \
    absl::Status _jaxmg_cusolver_status =                             \
        CusolverToStatus((expr), __FILE__, __LINE__);                 \
    if (!_jaxmg_cusolver_status.ok()) return _jaxmg_cusolver_status;  \
  } while (0)

#define JAXMG_RETURN_IF_ERROR(expr)                         \
  do {                                                      \
    absl::Status _jaxmg_status = (expr);                    \
    if (!_jaxmg_status.ok()) return _jaxmg_status;          \
  } while (0)

// cuSolverMg APIs are typed by cudaDataType at runtime, while XLA FFI exposes
// primitive element types. The solver traits keep the C++ template type, the
// cuSolverMg data type, the eigenvalue data type, and the failure fill value in
// one place for each supported dtype.
template <typename T>
struct SolverTraits;

template <>
struct SolverTraits<float> {
  using HostNanType = float;
  using EigenvalueType = float;
  static constexpr cudaDataType cuda_data_type = CUDA_R_32F;
  static constexpr cudaDataType eigenvalue_cuda_data_type = CUDA_R_32F;
  static float Nan() { return NAN; }
  static EigenvalueType EigenvalueNan() { return NAN; }
};

template <>
struct SolverTraits<double> {
  using HostNanType = double;
  using EigenvalueType = double;
  static constexpr cudaDataType cuda_data_type = CUDA_R_64F;
  static constexpr cudaDataType eigenvalue_cuda_data_type = CUDA_R_64F;
  static double Nan() { return NAN; }
  static EigenvalueType EigenvalueNan() { return NAN; }
};

template <>
struct SolverTraits<cuFloatComplex> {
  using HostNanType = cuFloatComplex;
  using EigenvalueType = float;
  static constexpr cudaDataType cuda_data_type = CUDA_C_32F;
  static constexpr cudaDataType eigenvalue_cuda_data_type = CUDA_R_32F;
  static cuFloatComplex Nan() { return make_cuFloatComplex(NAN, NAN); }
  static EigenvalueType EigenvalueNan() { return NAN; }
};

template <>
struct SolverTraits<cuDoubleComplex> {
  using HostNanType = cuDoubleComplex;
  using EigenvalueType = double;
  static constexpr cudaDataType cuda_data_type = CUDA_C_64F;
  static constexpr cudaDataType eigenvalue_cuda_data_type = CUDA_R_64F;
  static cuDoubleComplex Nan() { return make_cuDoubleComplex(NAN, NAN); }
  static EigenvalueType EigenvalueNan() { return NAN; }
};

class ReusableHostBarrier {
 public:
  // Process-local barrier for the FFI invocations participating in one
  // single-node solver call. This is not a multi-node synchronization
  // primitive.
  void ArriveAndWait(int participants) {
    std::unique_lock<std::mutex> lock(mu_);
    if (participants_ != participants) {
      participants_ = participants;
      arrived_ = 0;
      generation_ = 0;
    }
    const int generation = generation_;
    if (++arrived_ == participants_) {
      arrived_ = 0;
      ++generation_;
      cv_.notify_all();
      return;
    }
    cv_.wait(lock, [&] { return generation != generation_; });
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  int participants_ = 0;
  int arrived_ = 0;
  int generation_ = 0;
};

// Per-solver pointer exchange state. cuSolverMg consumes host arrays of device
// pointers from the rank-0 host invocation, while each local FFI invocation owns
// only its current device's JAX buffers and scratch.
//
// These structs are used only in SPMD mode. MPMD uses MpmdSolverExchange in
// mpmd_ipc.h because the participating ranks are separate host processes.
struct FusedPotrsState {
  std::array<void*, 16> a{};
  std::array<void*, 16> b{};
  std::array<void*, 16> work{};
  std::array<int64_t, 16> lwork{};
  std::array<int32_t, 16> solver_status{};
};

struct FusedMatrixState {
  std::array<void*, 16> a{};
  std::array<void*, 16> work{};
  std::array<int64_t, 16> lwork{};
  std::array<int32_t, 16> solver_status{};
};

struct FusedSyevdState {
  std::array<void*, 16> a{};
  std::array<void*, 16> eigenvalues{};
  std::array<void*, 16> work{};
  std::array<int64_t, 16> lwork{};
  std::array<int32_t, 16> solver_status{};
};

extern ReusableHostBarrier fused_potrs_barrier;
extern FusedPotrsState fused_potrs_state;
extern ReusableHostBarrier fused_potri_barrier;
extern FusedMatrixState fused_potri_state;
extern ReusableHostBarrier fused_syevd_barrier;
extern FusedSyevdState fused_syevd_state;
extern ReusableHostBarrier fused_syevd_no_v_barrier;
extern FusedSyevdState fused_syevd_no_v_state;

// Lightweight opt-in timing helper shared by solver handlers. It is named after
// the first path it was built for, but is now used by potrs, potri, and syevd.
class PotrsPhaseTimer {
 public:
  PotrsPhaseTimer(const char* path, int rank, int ranks, int64_t n,
                  int64_t nrhs, int64_t tile_size, int64_t start)
      : enabled_(rank == 0 && TimingEnabled()),
        path_(path),
        rank_(rank),
        ranks_(ranks),
        n_(n),
        nrhs_(nrhs),
        tile_size_(tile_size),
        start_(start),
        last_(start) {}

  void Mark(const char* phase) {
    if (!enabled_) {
      return;
    }
    const int64_t now = NowNanos();
    const double delta_ms = static_cast<double>(now - last_) / 1.0e6;
    const double total_ms = static_cast<double>(now - start_) / 1.0e6;
    std::fprintf(
        stderr,
        "JAXMG_POTRS_TIMING path=%s rank=%d ranks=%d n=%lld nrhs=%lld "
        "tile=%lld phase=%s delta_ms=%.3f total_ms=%.3f\n",
        path_, rank_, ranks_, static_cast<long long>(n_),
        static_cast<long long>(nrhs_), static_cast<long long>(tile_size_),
        phase, delta_ms, total_ms);
    std::fflush(stderr);
    last_ = now;
  }

  static int64_t NowNanos() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
  }

 private:
  static bool TimingEnabled() {
    const char* env = std::getenv("JAXMG_POTRS_TIMING");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
  }

  bool enabled_;
  const char* path_;
  int rank_;
  int ranks_;
  int64_t n_;
  int64_t nrhs_;
  int64_t tile_size_;
  int64_t start_;
  int64_t last_;
};

absl::StatusOr<void*> AllocateFfiScratch(se::ScratchAllocator& scratch,
                                         size_t bytes, const char* name);

// Build the XLA collective groups used by this backend. The current production
// path is single-process/single-node and therefore uses the local device set.
// The node-scoped helpers are the MPMD migration point: they keep cuSolverMg
// single-node by grouping ranks in chunks of JAXMG_NUMBER_OF_DEVICES, while
// still using XLA communicator contexts for collectives inside that node.
// The all-assigned helpers are for the cuSOLVERMp investigation path, where
// the communicator must eventually span every rank in the distributed
// cuSOLVERMp process grid rather than just the current node.
// LocalDevicesCliqueKey is for ordinary collectives; LocalDevicesP2PCliqueKey
// uses a communication id for point-to-point style CollectivePermute calls.
ReplicaGroup LocalDevicesReplicaGroup(const CollectiveParams& params);
std::vector<GlobalDeviceId> LocalGlobalDeviceGroup(
    const CollectiveParams& params);
ReplicaGroup AllAssignedDevicesReplicaGroup(const CollectiveParams& params);
std::vector<GlobalDeviceId> AllAssignedGlobalDeviceGroup(
    const CollectiveParams& params);
absl::StatusOr<std::vector<GlobalDeviceId>> NodeScopedGlobalDeviceGroup(
    const CollectiveParams& params);
absl::StatusOr<ReplicaGroup> NodeScopedReplicaGroup(
    const CollectiveParams& params);
absl::StatusOr<int> NodeScopedGroupOrdinal(const CollectiveParams& params);
absl::StatusOr<GpuCliqueKey> LocalDevicesCliqueKey(
    const CollectiveParams& params);
absl::StatusOr<GpuCliqueKey> LocalDevicesP2PCliqueKey(
    const CollectiveParams& params);
absl::StatusOr<GpuCliqueKey> AllAssignedDevicesCliqueKey(
    const CollectiveParams& params);
absl::StatusOr<GpuCliqueKey> AllAssignedDevicesP2PCliqueKey(
    const CollectiveParams& params);
absl::StatusOr<GpuCliqueKey> NodeScopedCliqueKey(
    const CollectiveParams& params);
absl::StatusOr<GpuCliqueKey> NodeScopedP2PCliqueKey(
    const CollectiveParams& params);

absl::Status BroadcastBufferFromRank0(
    const char* caller, se::Stream* stream, se::Stream* comm_stream,
    Communicator* comm, se::DeviceAddressBase buffer, PrimitiveType element_type,
    size_t element_count, int64_t num_ranks, int64_t rank_value);

// Execute the native 1D row-sharded <-> cuSolverMg block-cyclic reshuffle.
// matrix_base and matrix_out_base may alias. scratch_base must be at least one
// column when local_scratch_slots == 1, which is enough for the closed-cycle
// permutation algorithm.
absl::Status ExecuteMatrixColumnNativePlanRaw(
    const char* caller, se::Stream* stream, se::Stream* comm_stream,
    Communicator* comm, ffi::AnyBuffer matrix,
    se::DeviceAddressBase matrix_base, se::DeviceAddressBase matrix_out_base,
    se::DeviceAddressBase scratch_base, se::DeviceAddressBase scratch_out_base,
    int64_t local_slots, int64_t column_elements, uint64_t column_bytes,
    int64_t local_scratch_slots, int64_t num_ranks, int64_t rank_value,
    int64_t tile_size, bool reverse);

// Diagnostic handlers. These are useful when changing the XLA integration
// because they isolate clique construction, communicator lookup, all-reduce,
// CollectivePermute, and local rectangle addressing without involving
// cuSolverMg.
absl::Status XlaCommCollectiveProbePrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests);
absl::Status XlaCommCollectiveProbeDispatch(
    se::Stream* stream, ffi::AnyBuffer token,
    ffi::Result<ffi::BufferR1<S32>> out,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques);
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
absl::Status XlaRectPackUnpackProbePrepare();
absl::Status XlaRectPackUnpackProbeDispatch(
    cudaStream_t cuda_stream, int64_t row_start, int64_t col_start,
    int64_t row_count, int64_t col_count,
    int64_t target_row, int64_t target_col, ffi::AnyBuffer matrix,
    ffi::AnyBuffer scratch,
    ffi::Result<ffi::AnyBuffer> matrix_out,
    ffi::Result<ffi::AnyBuffer> scratch_out);
absl::Status XlaRectTransferProbePrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests);
absl::Status XlaRectTransferProbeDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    absl::Span<const int64_t> targets,
    absl::Span<const int64_t> src_row_starts,
    absl::Span<const int64_t> src_col_starts,
    absl::Span<const int64_t> dst_row_starts,
    absl::Span<const int64_t> dst_col_starts, int64_t row_count,
    int64_t col_count, ffi::AnyBuffer matrix, ffi::AnyBuffer scratch,
    ffi::Result<ffi::AnyBuffer> matrix_out,
    ffi::Result<ffi::AnyBuffer> scratch_out,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques);
absl::Status XlaRect2DNativePlanPrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests);
absl::Status XlaRect2DNativePlanDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    int64_t process_rows, int64_t process_cols, int64_t tile_rows,
    int64_t tile_cols, ffi::AnyBuffer matrix,
    ffi::AnyBuffer scratch, ffi::Result<ffi::AnyBuffer> matrix_out,
    ffi::Result<ffi::AnyBuffer> scratch_out,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques);
absl::Status XlaRectPadded2DNativePlanPrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests);
absl::Status XlaRectPadded2DNativePlanDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    int64_t process_rows, int64_t process_cols, int64_t tile_rows,
    int64_t tile_cols, int64_t logical_rows, int64_t logical_cols,
    int64_t reverse, ffi::AnyBuffer matrix, ffi::AnyBuffer scratch,
    ffi::Result<ffi::AnyBuffer> matrix_out,
    ffi::Result<ffi::AnyBuffer> scratch_out,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques);
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
absl::Status XlaCusolverMpPotrsPrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests);
absl::Status XlaCusolverMpPotrsDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    int64_t process_rows, int64_t process_cols, int64_t n, int64_t nrhs,
    int64_t tile_size, ffi::AnyBuffer a, ffi::AnyBuffer b,
    ffi::Result<ffi::AnyBuffer> a_out, ffi::Result<ffi::AnyBuffer> b_out,
    ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques);

// Redistribution handlers. The step/batch variants are retained for focused
// testing and diagnostics. The production solvers use the native-plan variant
// so the full permutation schedule is constructed and executed in C++.
absl::Status XlaCommMatrixColumnStepPrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests);
absl::Status XlaCommMatrixColumnStepDispatch(
    se::Stream* stream, se::Stream* comm_stream, int64_t kind,
    int64_t source_rank_attr, int64_t target_rank_attr, int64_t source_col,
    int64_t target_col, ffi::AnyBuffer matrix, ffi::AnyBuffer scratch,
    ffi::Result<ffi::AnyBuffer> matrix_out,
    ffi::Result<ffi::AnyBuffer> scratch_out,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques);
absl::Status XlaCommMatrixColumnBatchPrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests);
absl::Status XlaCommMatrixColumnBatchDispatch(
    se::Stream* stream, se::Stream* comm_stream,
    absl::Span<const int64_t> kinds,
    absl::Span<const int64_t> source_ranks,
    absl::Span<const int64_t> target_ranks,
    absl::Span<const int64_t> source_cols,
    absl::Span<const int64_t> target_cols,
    absl::Span<const int64_t> scratch_slots, ffi::AnyBuffer matrix,
    ffi::AnyBuffer scratch, ffi::Result<ffi::AnyBuffer> matrix_out,
    ffi::Result<ffi::AnyBuffer> scratch_out,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques);
absl::Status XlaCommMatrixColumnNativePlanDispatch(
    se::Stream* stream, se::Stream* comm_stream, int64_t tile_size,
    ffi::AnyBuffer matrix, ffi::AnyBuffer scratch,
    ffi::Result<ffi::AnyBuffer> matrix_out,
    ffi::Result<ffi::AnyBuffer> scratch_out,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques);

// Fused solver handlers. These keep the old Python FFI target names but use the
// XLA communicator reshuffler internally before and, where needed, after the
// cuSolverMg call.
absl::Status XlaCommPotrsMgNativePlanDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    se::OwningScratchAllocator<> scratch, int64_t tile_size, ffi::AnyBuffer a,
    ffi::AnyBuffer b, ffi::Result<ffi::AnyBuffer> out_a,
    ffi::Result<ffi::AnyBuffer> out_b,
    ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques);
absl::Status XlaCommPotriMgNativePlanDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    se::OwningScratchAllocator<> scratch, int64_t tile_size, ffi::AnyBuffer a,
    ffi::Result<ffi::AnyBuffer> out,
    ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques);
absl::Status XlaCommSyevdMgNativePlanDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    se::OwningScratchAllocator<> scratch, int64_t tile_size, ffi::AnyBuffer a,
    ffi::Result<ffi::AnyBuffer> eigenvalues,
    ffi::Result<ffi::AnyBuffer> vectors,
    ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques);
absl::Status XlaCommSyevdNoVMgNativePlanDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    se::OwningScratchAllocator<> scratch, int64_t tile_size, ffi::AnyBuffer a,
    ffi::Result<ffi::AnyBuffer> eigenvalues,
    ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques);

}  // namespace xla::gpu

#endif  // JAXMG_XLA_COMM_BACKEND_H_
