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
// traits, and access to XLA's public Collectives FFI extension.
// Solver-specific and redistribution-specific declarations live in their own
// headers.

#ifndef JAXMG_XLA_COMM_COMMON_H_
#define JAXMG_XLA_COMM_COMMON_H_

#include <cuComplex.h>
#include <cuda_runtime_api.h>
#include <cusolver_common.h>
#include <nccl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/ffi/api/collectives_api.h"
#include "xla/ffi/api/ffi.h"

// Abseil stays private to this library. Only C FFI errors and array views cross
// the boundary; no XLA C++ object is passed into the backend.
namespace xla::ffi {
template <ExecutionStage stage>
struct ResultEncoding<stage, absl::Status> {
  static XLA_FFI_Error* Encode(const XLA_FFI_Api* api,
                               XLA_FFI_InvokeContext* ctx,
                               absl::Status status) {
    return ResultEncoding<stage, Error>::Encode(
        api, ctx,
        Error(static_cast<ErrorCode>(status.code()),
              std::string(status.message())));
  }
};

template <>
struct AttrDecoding<absl::Span<const int64_t>> {
  using Type = absl::Span<const int64_t>;
  static std::optional<absl::Span<const int64_t>> Decode(
      XLA_FFI_AttrType type, void* attr, DiagnosticEngine& diagnostic) {
    auto values =
        AttrDecoding<Span<const int64_t>>::Decode(type, attr, diagnostic);
    if (!values) return std::nullopt;
    return absl::Span<const int64_t>(values->begin(), values->size());
  }
};
}  // namespace xla::ffi

namespace jaxmg {
namespace ffi = ::xla::ffi;

// The minimum supported FFI headers expose a policy-based Collectives wrapper.
// This policy keeps existing native status propagation without linking XLA.
struct CollectiveErrorPolicy {
  using Status = absl::Status;
  template <typename T>
  using StatusOr = absl::StatusOr<T>;
  static Status Ok() { return absl::OkStatus(); }
  static Status TakeError(const XLA_FFI_Api* api, XLA_FFI_Error* error);
};
using CollectiveContext =
    ffi::internal::CommunicatorContextBase<CollectiveErrorPolicy>;
struct Collectives
    : ffi::internal::CollectivesExtensionBase<CollectiveContext> {
  static bool Support(int32_t major, int32_t minor) {
    return major == kMajorVersion && minor >= kMinorVersion;
  }
};

// Convert CUDA/cuSOLVER errors into absl::Status so all FFI handlers can return
// failures through the same mechanism used by XLA.
absl::Status CudaToStatus(cudaError_t err, const char* file, int line);
absl::Status NcclToStatus(ncclResult_t err, const char* file, int line);

#define JAXMG_RETURN_IF_CUDA_ERROR(expr)                     \
  do {                                                       \
    absl::Status _jaxmg_cuda_status =                        \
        CudaToStatus((expr), __FILE__, __LINE__);            \
    if (!_jaxmg_cuda_status.ok()) return _jaxmg_cuda_status; \
  } while (0)

#define JAXMG_RETURN_IF_NCCL_ERROR(expr)                     \
  do {                                                       \
    absl::Status _jaxmg_nccl_status =                        \
        NcclToStatus((expr), __FILE__, __LINE__);            \
    if (!_jaxmg_nccl_status.ok()) return _jaxmg_nccl_status; \
  } while (0)

#define JAXMG_RETURN_IF_ERROR(expr)                \
  do {                                             \
    absl::Status _jaxmg_status = (expr);           \
    if (!_jaxmg_status.ok()) return _jaxmg_status; \
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
  // Returns a complex matrix fill value used only for failure/debug fallback
  // paths.
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
  // Returns a complex matrix fill value used only for failure/debug fallback
  // paths.
  static cuDoubleComplex Nan() { return make_cuDoubleComplex(NAN, NAN); }
  // Returns the real eigenvalue fill value for complex128 Hermitian matrices.
  static EigenvalueType EigenvalueNan() { return NAN; }
};

// Prepare and execution request the same ordered SPMD group. Its NCCL ranks
// follow logical partition IDs, independently of physical device placement.
absl::Status XlaCusolverMpPrepare(ffi::Dictionary attributes,
                                  CollectiveContext collectives);

// Borrows and validates the raw NCCL handle owned by XLA for one FFI call.
// The returned handle remains owned by XLA and must never be destroyed by
// JAXMg.
absl::StatusOr<ncclComm_t> BorrowNcclComm(const char* caller,
                                          CollectiveContext collectives,
                                          int64_t num_ranks);
absl::Status ValidateBorrowedNcclComm(const char* caller, ncclComm_t comm,
                                      int64_t expected_rank,
                                      int64_t expected_count);

// Reduces one device-resident real scalar across the borrowed communicator and
// leaves the global sum in the same buffer on every rank. `dtype` must be
// ncclFloat or ncclDouble.
absl::Status RunRawNcclAllReduceReal(const char* caller,
                                     cudaStream_t cuda_stream, ncclComm_t comm,
                                     ncclDataType_t dtype, void* value);

}  // namespace jaxmg

#endif  // JAXMG_XLA_COMM_COMMON_H_
