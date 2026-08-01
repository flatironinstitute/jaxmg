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
// This header defines the shared boundary between the fused POTRS, LU, SYEVD,
// and GESVD handlers and the cuSOLVERMp runtime. It contains the status
// schemas, process-grid validation, typed cuSOLVERMp entry points, and CUDA/XLA
// utility declarations used by each solver. Solver orchestration remains in
// the corresponding solver source file.

#ifndef JAXMG_CUSOLVERMP_COMMON_H_
#define JAXMG_CUSOLVERMP_COMMON_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

#include <cusolverMp.h>

#include "../xla_ffi/runtime.h"
#include "third_party/nccl/nccl.h"

namespace xla::gpu {

// Typed table of the cuSOLVERMp entry points used by the native solver handlers.
// Grouping the functions here keeps runtime invocation consistent across the
// supported solver routines.
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
  decltype(&cusolverMpGetrf_bufferSize) getrf_buffer_size =
      &cusolverMpGetrf_bufferSize;
  decltype(&cusolverMpGetrf) getrf = &cusolverMpGetrf;
  decltype(&cusolverMpGetrs_bufferSize) getrs_buffer_size =
      &cusolverMpGetrs_bufferSize;
  decltype(&cusolverMpGetrs) getrs = &cusolverMpGetrs;
  decltype(&cusolverMpSyevd_bufferSize) syevd_buffer_size =
      &cusolverMpSyevd_bufferSize;
  decltype(&cusolverMpSyevd) syevd = &cusolverMpSyevd;
  decltype(&cusolverMpGesvdDescriptorCreate) gesvd_descriptor_create =
      &cusolverMpGesvdDescriptorCreate;
  decltype(&cusolverMpGesvdDescriptorDestroy) gesvd_descriptor_destroy =
      &cusolverMpGesvdDescriptorDestroy;
  decltype(&cusolverMpGesvdDescriptorSetAttribute)
      gesvd_descriptor_set_attribute = &cusolverMpGesvdDescriptorSetAttribute;
  decltype(&cusolverMpGesvdDescriptorGetAttribute)
      gesvd_descriptor_get_attribute = &cusolverMpGesvdDescriptorGetAttribute;
  decltype(&cusolverMpGesvd_bufferSize) gesvd_buffer_size =
      &cusolverMpGesvd_bufferSize;
  decltype(&cusolverMpGesvd) gesvd = &cusolverMpGesvd;
};

// Native status codes returned through the small device status vectors.
//
// Python reads these integer codes after an FFI call to distinguish ordinary
// solver failures from communicator, grid, allocation, or descriptor setup
// failures without parsing stderr from a distributed Slurm run.
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
  kGetrfWorkspaceFailed = 36,
  kGetrsWorkspaceFailed = 37,
  kGetrfFailed = 38,
  kGetrfInfoNonzero = 39,
  kGetrsFailed = 40,
  kGetrsInfoNonzero = 41,
  kGesvdDescriptorCreateFailed = 42,
  kGesvdDescriptorAttributeFailed = 43,
  kGesvdWorkspaceFailed = 44,
  kGesvdFailed = 45,
  kGesvdInfoNonzero = 46,
  kGesvdDescriptorDestroyFailed = 47,
  kGesvdDescriptorQueryFailed = 48,
};

inline constexpr int kPotrsStatusSize = 40;
inline constexpr int kLuSolveStatusSize = 41;
inline constexpr int kSyevdStatusSize = 36;
inline constexpr int kGesvdStatusSize = 42;
inline constexpr cusolverMpGridMapping_t kCusolverMpGridMappingRowMajor =
    CUSOLVERMP_GRID_MAPPING_ROW_MAJOR;
inline constexpr cusolverMpGridMapping_t kCusolverMpGridMappingColMajor =
    CUSOLVERMP_GRID_MAPPING_COL_MAJOR;

// Converts the Python-provided grid mapping attribute into cuSOLVERMp's enum.
cusolverMpGridMapping_t ToCusolverMpGridMapping(int64_t grid_mapping);

// Checks that the requested grid mapping is one of cuSOLVERMp's supported
// dense row-major or column-major process-grid mappings.
absl::Status ValidateCusolverMpGridMapping(const char* caller,
                                           int64_t grid_mapping);

// Validates that a JAX mesh rank map matches the selected dense row-major or
// column-major cuSOLVERMp process-grid mapping.
absl::Status ValidateStandardRankMapForGridMapping(
    const char* caller, absl::Span<const int64_t> rank_map,
    int64_t process_rows, int64_t process_cols, int64_t grid_mapping);

// Converts an NCCL communicator rank into the process-row/process-column
// coordinate used by cuSOLVERMp descriptors.
std::pair<int32_t, int32_t> ProcessCoordFromRank(int nccl_rank,
                                                 int64_t process_rows,
                                                 int64_t process_cols,
                                                 int64_t grid_mapping);

// Returns a linked cuSOLVERMp API table and marks the status vector to show
// that the direct cuSOLVERMp runtime dependency was available.
template <size_t StatusSize>
CusolverMpApi LinkedCusolverMpApi(std::array<int32_t, StatusSize>* status) {
  static_assert(StatusSize > 7, "cuSOLVERMp status vector is too small");
  (*status)[7] = 1;
  return CusolverMpApi{};
}

// Computes the ScaLAPACK/NUMROC local length owned by one process coordinate
// for a block-cyclic axis.
int64_t LocalNumroc(int64_t n, int64_t block, int32_t process,
                    int32_t process_count);

// Copies a POTRS status vector from host memory into the JAX-visible device
// status output.
absl::Status CopyPotrsStatusToDevice(
    se::Stream* stream, const std::array<int32_t, kPotrsStatusSize>& status,
    ffi::Result<ffi::BufferR1<S32>> out);

// Copies an LU-solve status vector from host memory into the JAX-visible device
// status output.
absl::Status CopyLuSolveStatusToDevice(
    se::Stream* stream, const std::array<int32_t, kLuSolveStatusSize>& status,
    ffi::Result<ffi::BufferR1<S32>> out);

// Copies an SYEVD status vector from host memory into the JAX-visible device
// status output.
absl::Status CopySyevdStatusToDevice(
    se::Stream* stream, const std::array<int32_t, kSyevdStatusSize>& status,
    ffi::Result<ffi::BufferR1<S32>> out);

// Copies a GESVD status vector from host memory into the JAX-visible device
// status output.
absl::Status CopyGesvdStatusToDevice(
    se::Stream* stream, const std::array<int32_t, kGesvdStatusSize>& status,
    ffi::Result<ffi::BufferR1<S32>> out);

// Encodes workspace byte sizes compactly in status vectors as KiB.
int32_t SizeToKiBForStatus(size_t bytes);

// Copies an input buffer into an output/work buffer only when XLA did not alias
// the two buffers for the current FFI call.
absl::Status CopyAnyBufferToOutputIfNeeded(cudaStream_t cuda_stream,
                                           ffi::AnyBuffer input,
                                           ffi::Result<ffi::AnyBuffer> output);

// Returns whether verbose cuSOLVERMp debug logging is enabled for native runs.
bool CusolverMpDebugEnabled();

// Emits rank-tagged debug logging for hard-to-debug distributed cuSOLVERMp
// failures when JAXMG_CUSOLVERMP_DEBUG is set.
void CusolverMpDebug(int rank, const char* format, ...);

// Resolves the CUDA device that owns a donated JAX buffer pointer so each FFI
// rank binds cuSOLVERMp to the correct local GPU.
absl::StatusOr<int> DeviceForCudaPointer(const void* ptr);

}  // namespace xla::gpu

#endif  // JAXMG_CUSOLVERMP_COMMON_H_
