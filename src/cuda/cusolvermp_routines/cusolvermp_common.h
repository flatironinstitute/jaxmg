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
// Shared cuSOLVERMp runtime helpers.
//
// This header is the private boundary between the fused POTRS/SYEVD FFI
// handlers and NVIDIA's cuSOLVERMp runtime. It deliberately contains only
// production concerns: status-vector schema, grid/rank validation, direct
// cuSOLVERMp symbol access, and small CUDA/XLA utility helpers. Solver-specific
// orchestration stays in cusolvermp_potrs.cc and cusolvermp_syevd.cc.

#ifndef JAXMG_CUSOLVERMP_COMMON_H_
#define JAXMG_CUSOLVERMP_COMMON_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

#include <cusolverMp.h>

#include "../include/xla_comm_common.h"
#include "third_party/nccl/nccl.h"

namespace xla::gpu {

struct CusolverMpApi {
  decltype(&cusolverMpCreate) create = &cusolverMpCreate;
  decltype(&cusolverMpDestroy) destroy = &cusolverMpDestroy;
  decltype(&cusolverMpGetVersion) get_version = &cusolverMpGetVersion;
  decltype(&cusolverMpCreateDeviceGrid) create_grid =
      &cusolverMpCreateDeviceGrid;
  decltype(&cusolverMpDestroyGrid) destroy_grid = &cusolverMpDestroyGrid;
  decltype(&cusolverMpCreateMatrixDesc) create_matrix_desc =
      &cusolverMpCreateMatrixDesc;
  decltype(&cusolverMpDestroyMatrixDesc) destroy_matrix_desc =
      &cusolverMpDestroyMatrixDesc;
  decltype(&cusolverMpPotrf_bufferSize) potrf_buffer_size =
      &cusolverMpPotrf_bufferSize;
  decltype(&cusolverMpPotrf) potrf = &cusolverMpPotrf;
  decltype(&cusolverMpPotrs_bufferSize) potrs_buffer_size =
      &cusolverMpPotrs_bufferSize;
  decltype(&cusolverMpPotrs) potrs = &cusolverMpPotrs;
  decltype(&cusolverMpSyevd_bufferSize) syevd_buffer_size =
      &cusolverMpSyevd_bufferSize;
  decltype(&cusolverMpSyevd) syevd = &cusolverMpSyevd;
};

enum CusolverMpStatusCode : int32_t {
  kStatusOk = 0,
  kLibraryMissing = 1,
  kSymbolMissing = 2,
  kCudaDeviceFailed = 3,
  kCollectiveContextMissing = 4,
  kCliqueKeyFailed = 5,
  kCommunicatorMissing = 6,
  kNcclHandleMissing = 7,
  kNcclRankMismatch = 8,
  kGridShapeMismatch = 9,
  kCreateHandleFailed = 10,
  kGetVersionFailed = 11,
  kCreateGridFailed = 12,
  kCreateMatrixDescFailed = 13,
  kDestroyMatrixDescFailed = 14,
  kDestroyGridFailed = 15,
  kDestroyHandleFailed = 16,
  kScatterSymbolMissing = 17,
  kUnsupportedDtype = 18,
  kOutputShapeMismatch = 19,
  kScatterFailed = 20,
  kSolverSymbolMissing = 21,
  kPotrfWorkspaceFailed = 22,
  kPotrsWorkspaceFailed = 23,
  kDeviceAllocFailed = 24,
  kHostAllocFailed = 25,
  kPotrfFailed = 26,
  kPotrfInfoNonzero = 27,
  kPotrsFailed = 28,
  kPotrsInfoNonzero = 29,
  kGatherSymbolMissing = 30,
  kGatherFailed = 31,
  kResidualTooLarge = 32,
  kSyevdWorkspaceFailed = 33,
  kSyevdFailed = 34,
  kSyevdInfoNonzero = 35,
};

inline constexpr int kPotrsStatusSize = 40;
inline constexpr int kSyevdStatusSize = 36;
inline constexpr cusolverMpGridMapping_t kCusolverMpGridMappingRowMajor =
    CUSOLVERMP_GRID_MAPPING_ROW_MAJOR;
inline constexpr cusolverMpGridMapping_t kCusolverMpGridMappingColMajor =
    CUSOLVERMP_GRID_MAPPING_COL_MAJOR;

cusolverMpGridMapping_t ToCusolverMpGridMapping(int64_t grid_mapping);
absl::Status ValidateCusolverMpGridMapping(const char* caller,
                                           int64_t grid_mapping);
absl::Status ValidateStandardRankMapForGridMapping(
    const char* caller, absl::Span<const int64_t> rank_map,
    int64_t process_rows, int64_t process_cols, int64_t grid_mapping);
std::pair<int32_t, int32_t> ProcessCoordFromRank(int nccl_rank,
                                                 int64_t process_rows,
                                                 int64_t process_cols,
                                                 int64_t grid_mapping);

template <size_t StatusSize>
CusolverMpApi LinkedCusolverMpApi(std::array<int32_t, StatusSize>* status) {
  static_assert(StatusSize > 7, "cuSOLVERMp status vector is too small");
  (*status)[7] = 1;
  return CusolverMpApi{};
}

int64_t LocalNumroc(int64_t n, int64_t block, int32_t process,
                    int32_t process_count);
absl::Status CopyPotrsStatusToDevice(
    se::Stream* stream, const std::array<int32_t, kPotrsStatusSize>& status,
    ffi::Result<ffi::BufferR1<S32>> out);
absl::Status CopySyevdStatusToDevice(
    se::Stream* stream, const std::array<int32_t, kSyevdStatusSize>& status,
    ffi::Result<ffi::BufferR1<S32>> out);
int32_t SizeToKiBForStatus(size_t bytes);
absl::Status CopyAnyBufferToOutputIfNeeded(cudaStream_t cuda_stream,
                                           ffi::AnyBuffer input,
                                           ffi::Result<ffi::AnyBuffer> output);
bool CusolverMpDebugEnabled();
void CusolverMpDebug(int rank, const char* format, ...);
absl::StatusOr<int> DeviceForCudaPointer(const void* ptr);

}  // namespace xla::gpu

#endif  // JAXMG_CUSOLVERMP_COMMON_H_
