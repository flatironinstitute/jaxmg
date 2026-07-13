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
// Common declarations for the XLA FFI native backend.
//
// This header contains only cross-cutting utilities that are shared by memory
// redistribution and cuSOLVERMp routines: error/status conversion, dtype
// traits, XLA scratch allocation, and communicator clique construction.
// Solver-specific and redistribution-specific declarations live in their own
// headers.

#ifndef JAXMG_XLA_COMM_COMMON_H_
#define JAXMG_XLA_COMM_COMMON_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
#include "third_party/nccl/nccl.h"

namespace xla::gpu {
namespace ffi = ::xla::ffi;

// Convert CUDA/cuSOLVER errors into absl::Status so all FFI handlers can return
// failures through the same mechanism used by XLA.
absl::Status CudaToStatus(cudaError_t err, const char* file, int line);
absl::Status CusolverToStatus(cusolverStatus_t err, const char* file,
                              int line);
absl::Status NcclToStatus(ncclResult_t err, const char* file, int line);

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

#define JAXMG_RETURN_IF_NCCL_ERROR(expr)                         \
  do {                                                           \
    absl::Status _jaxmg_nccl_status =                            \
        NcclToStatus((expr), __FILE__, __LINE__);                \
    if (!_jaxmg_nccl_status.ok()) return _jaxmg_nccl_status;     \
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

// Solver metadata for float32 real matrices.
template <>
struct SolverTraits<float> {
  using HostNanType = float;
  using EigenvalueType = float;
  static constexpr cudaDataType cuda_data_type = CUDA_R_32F;
  static constexpr cudaDataType eigenvalue_cuda_data_type = CUDA_R_32F;
  // Returns a matrix fill value used only for failure/debug fallback paths.
  static float Nan() { return NAN; }
  // Returns the eigenvalue fill value for float32 real matrices.
  static EigenvalueType EigenvalueNan() { return NAN; }
};

// Solver metadata for float64 real matrices.
template <>
struct SolverTraits<double> {
  using HostNanType = double;
  using EigenvalueType = double;
  static constexpr cudaDataType cuda_data_type = CUDA_R_64F;
  static constexpr cudaDataType eigenvalue_cuda_data_type = CUDA_R_64F;
  // Returns a matrix fill value used only for failure/debug fallback paths.
  static double Nan() { return NAN; }
  // Returns the eigenvalue fill value for float64 real matrices.
  static EigenvalueType EigenvalueNan() { return NAN; }
};

// Solver metadata for complex64 Hermitian matrices and float32 eigenvalues.
template <>
struct SolverTraits<cuFloatComplex> {
  using HostNanType = cuFloatComplex;
  using EigenvalueType = float;
  static constexpr cudaDataType cuda_data_type = CUDA_C_32F;
  static constexpr cudaDataType eigenvalue_cuda_data_type = CUDA_R_32F;
  // Returns a complex matrix fill value used only for failure/debug fallback paths.
  static cuFloatComplex Nan() { return make_cuFloatComplex(NAN, NAN); }
  // Returns the real eigenvalue fill value for complex64 Hermitian matrices.
  static EigenvalueType EigenvalueNan() { return NAN; }
};

// Solver metadata for complex128 Hermitian matrices and float64 eigenvalues.
template <>
struct SolverTraits<cuDoubleComplex> {
  using HostNanType = cuDoubleComplex;
  using EigenvalueType = double;
  static constexpr cudaDataType cuda_data_type = CUDA_C_64F;
  static constexpr cudaDataType eigenvalue_cuda_data_type = CUDA_R_64F;
  // Returns a complex matrix fill value used only for failure/debug fallback paths.
  static cuDoubleComplex Nan() { return make_cuDoubleComplex(NAN, NAN); }
  // Returns the real eigenvalue fill value for complex128 Hermitian matrices.
  static EigenvalueType EigenvalueNan() { return NAN; }
};

// Allocates a device scratch pointer from XLA's per-call scratch allocator.
absl::StatusOr<void*> AllocateFfiScratch(se::ScratchAllocator& scratch,
                                         size_t bytes, const char* name);

// Build the XLA collective groups used by this backend. Production cuSOLVERMp
// solvers use the all-assigned helpers so the borrowed NCCL communicator spans
// every rank in the process grid.
ReplicaGroup AllAssignedDevicesReplicaGroup(const CollectiveParams& params);
std::vector<GlobalDeviceId> AllAssignedGlobalDeviceGroup(
    const CollectiveParams& params);
absl::StatusOr<GpuCliqueKey> AllAssignedDevicesCliqueKey(
    const CollectiveParams& params);
absl::StatusOr<GpuCliqueKey> AllAssignedDevicesP2PCliqueKey(
    const CollectiveParams& params);

// Shared prepare helper. It requests the all-assigned P2P communicator that
// backs cuSOLVERMp calls and native redistribution.
absl::Status RequestAllAssignedP2PCommunicator(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests, const char* caller);

// Borrows and validates the raw NCCL handle owned by XLA for one FFI call.
// The returned handle remains owned by XLA and must never be destroyed by
// JAXMg.
absl::StatusOr<ncclComm_t> BorrowNcclComm(const char* caller,
                                          GpuCommunicator* comm);
absl::Status ValidateBorrowedNcclComm(const char* caller, ncclComm_t comm,
                                      int64_t expected_rank,
                                      int64_t expected_count);

// Records the CUDA stream selected for raw NCCL work. Production calls prefer
// XLA's communication stream and fall back to the ordinary platform stream in
// contexts where XLA does not expose a separate communication stream.
struct NcclStreamChoice {
  cudaStream_t stream;
  bool uses_comm_stream;
};

absl::StatusOr<NcclStreamChoice> ChooseNcclStream(
    const char* caller, se::Stream* comm_stream, cudaStream_t cuda_stream);

// Reduces one device-resident real scalar across the borrowed communicator and
// leaves the global sum in the same buffer on every rank. `dtype` must be
// ncclFloat or ncclDouble.
absl::Status RunRawNcclAllReduceReal(
    const char* caller, se::Stream* stream, se::Stream* comm_stream,
    cudaStream_t cuda_stream, ncclComm_t comm, ncclDataType_t dtype,
    void* value);

}  // namespace xla::gpu

#endif  // JAXMG_XLA_COMM_COMMON_H_
