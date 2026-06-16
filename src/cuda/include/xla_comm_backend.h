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
//   2. XLA communicator clique helpers.
//   3. Shared 2D rectangle schedule types used by memory_redist/*.cc.
//   4. 2D rectangle/redistribution entry points used by diagnostics and
//      cuSOLVERMp solvers.
//   5. cuSOLVERMp diagnostic probes, shared solve helpers, and production FFI
//      entry points.
//
// The production cuSOLVERMp path is:
//   Python wrapper -> local shard padding -> native 2D redistribution ->
//   cuSOLVERMp solver over the borrowed XLA/NCCL communicator -> reverse
//   native 2D redistribution -> local unpadding.

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
#include <cusolver_common.h>
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

// Convert CUDA/cuSOLVER/cuSOLVERMp errors into absl::Status so all FFI handlers can
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

// cuSOLVERMp APIs are typed by cudaDataType at runtime, while XLA FFI exposes
// primitive element types. The solver traits keep the C++ template type, the
// cuSOLVERMp data type, the eigenvalue data type, and the failure fill value in
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

// Lightweight opt-in timing helper shared by solver handlers. It is named after
// the first path it was built for, but is now used by potrs and syevd.
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

// Build the XLA collective groups used by this backend. Production cuSOLVERMp
// solvers use the all-assigned helpers so the borrowed NCCL communicator spans
// every rank in the process grid. Node-scoped helpers remain for focused
// diagnostics that intentionally restrict communication to one host group.
ReplicaGroup AllAssignedDevicesReplicaGroup(const CollectiveParams& params);
std::vector<GlobalDeviceId> AllAssignedGlobalDeviceGroup(
    const CollectiveParams& params);
absl::StatusOr<std::vector<GlobalDeviceId>> NodeScopedGlobalDeviceGroup(
    const CollectiveParams& params);
absl::StatusOr<ReplicaGroup> NodeScopedReplicaGroup(
    const CollectiveParams& params);
absl::StatusOr<int> NodeScopedGroupOrdinal(const CollectiveParams& params);
absl::StatusOr<GpuCliqueKey> AllAssignedDevicesCliqueKey(
    const CollectiveParams& params);
absl::StatusOr<GpuCliqueKey> AllAssignedDevicesP2PCliqueKey(
    const CollectiveParams& params);
absl::StatusOr<GpuCliqueKey> NodeScopedCliqueKey(
    const CollectiveParams& params);
absl::StatusOr<GpuCliqueKey> NodeScopedP2PCliqueKey(
    const CollectiveParams& params);

// Shared 2D redistribution schedule.
//
// cuSOLVERMp needs a column-major local 2D block-cyclic layout. JAX users start
// from ordinary 2D block-sharded buffers. The native implementation represents
// both the edge-padding compaction and the block-cyclic redistribution as a
// sequence of rectangle moves over local buffers. These structs are deliberately
// small value types so they can be constructed in the planning files and then
// executed by rectangle_pack.cc without sharing ownership or lifetime state.
struct NativeLocalRect {
  int64_t row_start;
  int64_t col_start;
  int64_t row_count;
  int64_t col_count;
};

enum class Native2DStepKind : int64_t {
  kMove = 0,
  kSaveScratch = 1,
  kRestoreScratch = 2,
};

struct Native2DStep {
  int64_t phase;
  int64_t sequence;
  Native2DStepKind kind;
  int64_t source_rank;
  int64_t target_rank;
  NativeLocalRect source;
  NativeLocalRect target;
};

struct Native2DStepBatch {
  int64_t phase;
  Native2DStepKind kind;
  std::vector<Native2DStep> steps;
};

absl::Status CopyMatrixIfNeeded(cudaStream_t cuda_stream,
                                ffi::AnyBuffer matrix,
                                ffi::Result<ffi::AnyBuffer> matrix_out);
absl::Status CopyScratchIfNeeded(cudaStream_t cuda_stream,
                                 ffi::AnyBuffer scratch,
                                 ffi::Result<ffi::AnyBuffer> scratch_out);
int64_t MaxStepElementCount(const std::vector<Native2DStep>& steps);
std::vector<Native2DStepBatch> BatchNative2DSteps(
    const std::vector<Native2DStep>& steps);
absl::Status ExecuteNative2DStepBatches(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    const std::vector<Native2DStepBatch>& batches, int64_t local_rows,
    int64_t local_cols, int64_t rank_value, int64_t num_ranks,
    size_t element_bytes, int64_t slot_elements, ffi::AnyBuffer matrix,
    se::DeviceAddressBase matrix_out_base,
    se::DeviceAddressBase scratch_out_base, GpuCommunicator* comm);
absl::StatusOr<std::vector<Native2DStep>> BuildEdgePaddingNative2DSteps(
    int64_t process_rows, int64_t process_cols, int64_t tile_rows,
    int64_t tile_cols, int64_t logical_rows, int64_t logical_cols,
    int64_t local_rows, int64_t local_cols,
    absl::Span<const int64_t> rank_map);
std::vector<Native2DStep> ReverseEdgePaddingSteps(
    const std::vector<Native2DStep>& forward_steps);
absl::StatusOr<std::vector<Native2DStep>> BuildSlabNative2DSteps(
    int64_t process_rows, int64_t process_cols, int64_t tile_rows,
    int64_t tile_cols, int64_t local_rows, int64_t local_cols,
    absl::Span<const int64_t> rank_map, bool reverse = false);
absl::StatusOr<int64_t> RequiredPadded2DNativePlanScratchElements(
    int64_t process_rows, int64_t process_cols, int64_t tile_rows,
    int64_t tile_cols, int64_t logical_rows, int64_t logical_cols,
    int64_t local_rows, int64_t local_cols,
    absl::Span<const int64_t> rank_map);
absl::Status ExecutePadded2DNativePlanRaw(
    const char* caller, se::Stream* stream, se::Stream* comm_stream,
    cudaStream_t cuda_stream, int64_t process_rows, int64_t process_cols,
    int64_t tile_rows, int64_t tile_cols, int64_t logical_rows,
    int64_t logical_cols, int64_t reverse,
    absl::Span<const int64_t> rank_map, ffi::AnyBuffer matrix,
    se::DeviceAddressBase matrix_out_base,
    se::DeviceAddressBase scratch_base, int64_t scratch_elements,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques);

// Diagnostic handlers. These are useful when changing the XLA integration
// because they isolate clique construction, communicator lookup, all-reduce,
// CollectivePermute, and local rectangle addressing without involving
// cuSOLVERMp.
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
    int64_t tile_cols, absl::Span<const int64_t> rank_map,
    ffi::AnyBuffer matrix, ffi::AnyBuffer scratch,
    ffi::Result<ffi::AnyBuffer> matrix_out,
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
    int64_t reverse, absl::Span<const int64_t> rank_map,
    ffi::AnyBuffer matrix, ffi::AnyBuffer scratch,
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
// cusolvermp_routines/cusolvermp.cc owns the dynamic dlopen/dlsym boundary and
// the low-level cuSOLVERMp handle/grid/descriptor lifecycle. Production
// wrappers in cusolvermp_potrs.cc and cusolvermp_syevd.cc call these helpers
// after they have redistributed JAX buffers into local 2D block-cyclic form.
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

#endif  // JAXMG_XLA_COMM_BACKEND_H_
