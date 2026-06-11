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
// cuSOLVERMp dynamic boundary and first production potrs implementation.
//
// This file owns the native cuSOLVERMp integration layer.  It deliberately
// keeps the cuSOLVERMp dependency behind dlopen/dlsym for now: ordinary CUDA
// toolkit installs do not consistently ship cusolverMp.h/libcusolverMp.so, but
// the rest of the JAXMg XLA-communicator backend should still build and run its
// diagnostics when cuSOLVERMp is absent from the loader path.
//
// File workflow:
//   1. Request an all-assigned XLA GPU communicator for the active FFI call.
//   2. Borrow the raw NCCL handle stored inside XLA's GpuCommunicator.
//   3. Create a cuSOLVERMp handle and row-major cuSOLVERMp process grid.
//   4. Create column-major local matrix descriptors over JAXMg device buffers.
//   5. Run diagnostic probes that isolate initialization, scatter layout, and
//      host-generated potrs inputs.
//   6. Run the production `cusolvermp_potrs` handler on buffers that have
//      already been redistributed by rectangle_pack.cc into 2D block-cyclic
//      cuSOLVERMp layout.
//
// The probe entry points intentionally return device status vectors instead of
// failing hard.  That makes them useful in CI and on systems where the NVIDIA
// HPC SDK is not installed.  The production path uses the same status vector
// convention so Python can distinguish a missing cuSOLVERMp runtime from a
// numerical cuSOLVERMp failure.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <limits>
#include <vector>

#include "include/xla_comm_backend.h"
#include "third_party/nccl/nccl.h"

namespace xla::gpu {
namespace {

using CusolverMpOpaqueHandle = void*;
using CusolverMpOpaqueGrid = void*;
using CusolverMpOpaqueMatrixDesc = void*;

using CusolverMpCreateFn =
    cusolverStatus_t (*)(CusolverMpOpaqueHandle*, int, cudaStream_t);
using CusolverMpDestroyFn = cusolverStatus_t (*)(CusolverMpOpaqueHandle);
using CusolverMpGetVersionFn =
    cusolverStatus_t (*)(CusolverMpOpaqueHandle, int*);
using CusolverMpCreateDeviceGridFn =
    cusolverStatus_t (*)(CusolverMpOpaqueHandle, CusolverMpOpaqueGrid*,
                         ncclComm_t, int32_t, int32_t, int);
using CusolverMpDestroyGridFn = cusolverStatus_t (*)(CusolverMpOpaqueGrid);
using CusolverMpCreateMatrixDescFn =
    cusolverStatus_t (*)(CusolverMpOpaqueMatrixDesc*, CusolverMpOpaqueGrid,
                         cudaDataType, int64_t, int64_t, int64_t, int64_t,
                         uint32_t, uint32_t, int64_t);
using CusolverMpDestroyMatrixDescFn =
    cusolverStatus_t (*)(CusolverMpOpaqueMatrixDesc);
using CusolverMpMatrixScatterH2DFn =
    cusolverStatus_t (*)(CusolverMpOpaqueHandle, int64_t, int64_t, void*,
                         int64_t, int64_t, CusolverMpOpaqueMatrixDesc, int,
                         const void*, int64_t);
using CusolverMpMatrixGatherD2HFn =
    cusolverStatus_t (*)(CusolverMpOpaqueHandle, int64_t, int64_t,
                         const void*, int64_t, int64_t,
                         CusolverMpOpaqueMatrixDesc, int, void*, int64_t);
using CusolverMpPotrfBufferSizeFn =
    cusolverStatus_t (*)(CusolverMpOpaqueHandle, cublasFillMode_t, int64_t,
                         const void*, int64_t, int64_t,
                         CusolverMpOpaqueMatrixDesc, cudaDataType, size_t*,
                         size_t*);
using CusolverMpPotrfFn =
    cusolverStatus_t (*)(CusolverMpOpaqueHandle, cublasFillMode_t, int64_t,
                         void*, int64_t, int64_t, CusolverMpOpaqueMatrixDesc,
                         cudaDataType, void*, size_t, void*, size_t, int*);
using CusolverMpPotrsBufferSizeFn =
    cusolverStatus_t (*)(CusolverMpOpaqueHandle, cublasFillMode_t, int64_t,
                         int64_t, const void*, int64_t, int64_t,
                         CusolverMpOpaqueMatrixDesc, const void*, int64_t,
                         int64_t, CusolverMpOpaqueMatrixDesc, cudaDataType,
                         size_t*, size_t*);
using CusolverMpPotrsFn =
    cusolverStatus_t (*)(CusolverMpOpaqueHandle, cublasFillMode_t, int64_t,
                         int64_t, const void*, int64_t, int64_t,
                         CusolverMpOpaqueMatrixDesc, void*, int64_t, int64_t,
                         CusolverMpOpaqueMatrixDesc, cudaDataType, void*,
                         size_t, void*, size_t, int*);

struct CusolverMpApi {
  void* library = nullptr;
  CusolverMpCreateFn create = nullptr;
  CusolverMpDestroyFn destroy = nullptr;
  CusolverMpGetVersionFn get_version = nullptr;
  CusolverMpCreateDeviceGridFn create_grid = nullptr;
  CusolverMpDestroyGridFn destroy_grid = nullptr;
  CusolverMpCreateMatrixDescFn create_matrix_desc = nullptr;
  CusolverMpDestroyMatrixDescFn destroy_matrix_desc = nullptr;
  CusolverMpMatrixScatterH2DFn scatter_h2d = nullptr;
  CusolverMpMatrixGatherD2HFn gather_d2h = nullptr;
  CusolverMpPotrfBufferSizeFn potrf_buffer_size = nullptr;
  CusolverMpPotrfFn potrf = nullptr;
  CusolverMpPotrsBufferSizeFn potrs_buffer_size = nullptr;
  CusolverMpPotrsFn potrs = nullptr;
};

enum ProbeStatus : int32_t {
  kProbeOk = 0,
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
};

constexpr int kProbeSize = 16;
constexpr int kScatterProbeSize = 24;
constexpr int kPotrsProbeSize = 40;
constexpr int kResidualScale = 1000000;

// The dynamic probe cannot include cusolverMp.h because ordinary CUDA/cuSolver
// installs do not ship it. Mirror the enum values from the NVIDIA HPC SDK 26.3
// cuSOLVERMp 0.7.2 header:
//   CUSOLVERMP_GRID_MAPPING_ROW_MAJOR = 1
//   CUSOLVERMP_GRID_MAPPING_COL_MAJOR = 0
//
// JAXMg's 2D planner currently uses row-major rank mapping:
// rank = process_row * process_cols + process_col.
constexpr int kCusolverMpGridMappingRowMajor = 1;

template <typename Fn>
Fn LoadRequiredSymbol(void* library, const char* name) {
  dlerror();
  void* symbol = dlsym(library, name);
  if (symbol == nullptr || dlerror() != nullptr) {
    return nullptr;
  }
  return reinterpret_cast<Fn>(symbol);
}

template <size_t ProbeSize>
CusolverMpApi LoadCusolverMpApi(std::array<int32_t, ProbeSize>* probe) {
  static_assert(ProbeSize > 7, "cuSOLVERMp probe status vector is too small");
  CusolverMpApi api;
  const char* candidates[] = {
      "libcusolverMp.so",
      "libcusolverMp.so.0",
      "libcusolverMp.so.1",
  };
  for (const char* candidate : candidates) {
    api.library = dlopen(candidate, RTLD_NOW | RTLD_LOCAL);
    if (api.library != nullptr) {
      break;
    }
  }
  if (api.library == nullptr) {
    (*probe)[0] = kLibraryMissing;
    return api;
  }
  (*probe)[7] = 1;

  api.create =
      LoadRequiredSymbol<CusolverMpCreateFn>(api.library, "cusolverMpCreate");
  api.destroy =
      LoadRequiredSymbol<CusolverMpDestroyFn>(api.library, "cusolverMpDestroy");
  api.get_version = LoadRequiredSymbol<CusolverMpGetVersionFn>(
      api.library, "cusolverMpGetVersion");
  api.create_grid = LoadRequiredSymbol<CusolverMpCreateDeviceGridFn>(
      api.library, "cusolverMpCreateDeviceGrid");
  api.destroy_grid = LoadRequiredSymbol<CusolverMpDestroyGridFn>(
      api.library, "cusolverMpDestroyGrid");
  api.create_matrix_desc = LoadRequiredSymbol<CusolverMpCreateMatrixDescFn>(
      api.library, "cusolverMpCreateMatrixDesc");
  api.destroy_matrix_desc = LoadRequiredSymbol<CusolverMpDestroyMatrixDescFn>(
      api.library, "cusolverMpDestroyMatrixDesc");
  api.scatter_h2d = LoadRequiredSymbol<CusolverMpMatrixScatterH2DFn>(
      api.library, "cusolverMpMatrixScatterH2D");
  api.gather_d2h = LoadRequiredSymbol<CusolverMpMatrixGatherD2HFn>(
      api.library, "cusolverMpMatrixGatherD2H");
  api.potrf_buffer_size = LoadRequiredSymbol<CusolverMpPotrfBufferSizeFn>(
      api.library, "cusolverMpPotrf_bufferSize");
  api.potrf =
      LoadRequiredSymbol<CusolverMpPotrfFn>(api.library, "cusolverMpPotrf");
  api.potrs_buffer_size = LoadRequiredSymbol<CusolverMpPotrsBufferSizeFn>(
      api.library, "cusolverMpPotrs_bufferSize");
  api.potrs =
      LoadRequiredSymbol<CusolverMpPotrsFn>(api.library, "cusolverMpPotrs");

  if (api.create == nullptr || api.destroy == nullptr ||
      api.get_version == nullptr || api.create_grid == nullptr ||
      api.destroy_grid == nullptr || api.create_matrix_desc == nullptr ||
      api.destroy_matrix_desc == nullptr) {
    (*probe)[0] = kSymbolMissing;
  }
  return api;
}

int64_t LocalNumroc(int64_t n, int64_t block, int32_t process,
                    int32_t process_count) {
  int64_t owned = 0;
  const int64_t tile_count = (n + block - 1) / block;
  for (int64_t tile = 0; tile < tile_count; ++tile) {
    if (tile % process_count != process) {
      continue;
    }
    const int64_t start = tile * block;
    owned += std::min(block, n - start);
  }
  return owned;
}

absl::Status CopyProbeToDevice(se::Stream* stream,
                               const std::array<int32_t, kProbeSize>& probe,
                               ffi::Result<ffi::BufferR1<S32>> out) {
  se::DeviceAddress<int32_t> dst = out->device_memory();
  return stream->MemcpyH2D(absl::MakeConstSpan(probe), &dst);
}

bool HasRequiredSymbols(const CusolverMpApi& api) {
  return api.create != nullptr && api.destroy != nullptr &&
         api.get_version != nullptr && api.create_grid != nullptr &&
         api.destroy_grid != nullptr && api.create_matrix_desc != nullptr &&
         api.destroy_matrix_desc != nullptr;
}

bool HasPotrsSymbols(const CusolverMpApi& api) {
  return HasRequiredSymbols(api) && api.scatter_h2d != nullptr &&
         api.gather_d2h != nullptr && api.potrf_buffer_size != nullptr &&
         api.potrf != nullptr && api.potrs_buffer_size != nullptr &&
         api.potrs != nullptr;
}

bool HasDistributedPotrsSymbols(const CusolverMpApi& api) {
  return HasRequiredSymbols(api) && api.gather_d2h != nullptr &&
         api.potrf_buffer_size != nullptr && api.potrf != nullptr &&
         api.potrs_buffer_size != nullptr && api.potrs != nullptr;
}

bool HasDistributedPotrsSolveSymbols(const CusolverMpApi& api) {
  return HasRequiredSymbols(api) && api.potrf_buffer_size != nullptr &&
         api.potrf != nullptr && api.potrs_buffer_size != nullptr &&
         api.potrs != nullptr;
}

absl::Status CopyScatterProbeToDevice(
    se::Stream* stream, const std::array<int32_t, kScatterProbeSize>& probe,
    ffi::Result<ffi::BufferR1<S32>> out) {
  se::DeviceAddress<int32_t> dst = out->device_memory();
  return stream->MemcpyH2D(absl::MakeConstSpan(probe), &dst);
}

absl::Status CopyPotrsProbeToDevice(
    se::Stream* stream, const std::array<int32_t, kPotrsProbeSize>& probe,
    ffi::Result<ffi::BufferR1<S32>> out) {
  se::DeviceAddress<int32_t> dst = out->device_memory();
  return stream->MemcpyH2D(absl::MakeConstSpan(probe), &dst);
}

int32_t SizeToKiBForProbe(size_t bytes) {
  const size_t kib = (bytes + 1023) / 1024;
  if (kib > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return std::numeric_limits<int32_t>::max();
  }
  return static_cast<int32_t>(kib);
}

template <typename DataType>
DataType ScatterHostValue(int64_t row, int64_t col, int64_t cols);

template <>
float ScatterHostValue<float>(int64_t row, int64_t col, int64_t cols) {
  return static_cast<float>(row * cols + col + 1);
}

template <>
double ScatterHostValue<double>(int64_t row, int64_t col, int64_t cols) {
  return static_cast<double>(row * cols + col + 1);
}

template <>
cuFloatComplex ScatterHostValue<cuFloatComplex>(int64_t row, int64_t col,
                                                int64_t cols) {
  const float real = static_cast<float>(row * cols + col + 1);
  return make_cuFloatComplex(real, -real);
}

template <>
cuDoubleComplex ScatterHostValue<cuDoubleComplex>(int64_t row, int64_t col,
                                                  int64_t cols) {
  const double real = static_cast<double>(row * cols + col + 1);
  return make_cuDoubleComplex(real, -real);
}

template <typename DataType>
std::vector<DataType> MakeScatterHostMatrix(int64_t rows, int64_t cols) {
  std::vector<DataType> host(static_cast<size_t>(rows * cols));
  for (int64_t col = 0; col < cols; ++col) {
    for (int64_t row = 0; row < rows; ++row) {
      host[static_cast<size_t>(row + col * rows)] =
          ScatterHostValue<DataType>(row, col, cols);
    }
  }
  return host;
}

template <typename DataType>
DataType RealScalar(double value);

template <>
float RealScalar<float>(double value) {
  return static_cast<float>(value);
}

template <>
double RealScalar<double>(double value) {
  return value;
}

template <>
cuFloatComplex RealScalar<cuFloatComplex>(double value) {
  return make_cuFloatComplex(static_cast<float>(value), 0.0f);
}

template <>
cuDoubleComplex RealScalar<cuDoubleComplex>(double value) {
  return make_cuDoubleComplex(value, 0.0);
}

template <typename DataType>
DataType ProbeRhsValue();

template <>
float ProbeRhsValue<float>() {
  return 1.0f;
}

template <>
double ProbeRhsValue<double>() {
  return 1.0;
}

template <>
cuFloatComplex ProbeRhsValue<cuFloatComplex>() {
  return make_cuFloatComplex(1.0f, -1.0f);
}

template <>
cuDoubleComplex ProbeRhsValue<cuDoubleComplex>() {
  return make_cuDoubleComplex(1.0, -1.0);
}

template <typename DataType>
DataType ExpectedProbeSolution(int64_t row, int64_t n);

template <>
float ExpectedProbeSolution<float>(int64_t row, int64_t n) {
  return static_cast<float>(1.0 / static_cast<double>(n + row + 1));
}

template <>
double ExpectedProbeSolution<double>(int64_t row, int64_t n) {
  return 1.0 / static_cast<double>(n + row + 1);
}

template <>
cuFloatComplex ExpectedProbeSolution<cuFloatComplex>(int64_t row, int64_t n) {
  const float scale = static_cast<float>(1.0 / static_cast<double>(n + row + 1));
  return make_cuFloatComplex(scale, -scale);
}

template <>
cuDoubleComplex ExpectedProbeSolution<cuDoubleComplex>(int64_t row, int64_t n) {
  const double scale = 1.0 / static_cast<double>(n + row + 1);
  return make_cuDoubleComplex(scale, -scale);
}

template <typename DataType>
double AbsoluteDifference(DataType lhs, DataType rhs);

template <>
double AbsoluteDifference<float>(float lhs, float rhs) {
  return std::abs(static_cast<double>(lhs) - static_cast<double>(rhs));
}

template <>
double AbsoluteDifference<double>(double lhs, double rhs) {
  return std::abs(lhs - rhs);
}

template <>
double AbsoluteDifference<cuFloatComplex>(cuFloatComplex lhs,
                                          cuFloatComplex rhs) {
  const double real = static_cast<double>(cuCrealf(lhs) - cuCrealf(rhs));
  const double imag = static_cast<double>(cuCimagf(lhs) - cuCimagf(rhs));
  return std::hypot(real, imag);
}

template <>
double AbsoluteDifference<cuDoubleComplex>(cuDoubleComplex lhs,
                                           cuDoubleComplex rhs) {
  const double real = cuCreal(lhs) - cuCreal(rhs);
  const double imag = cuCimag(lhs) - cuCimag(rhs);
  return std::hypot(real, imag);
}

template <typename DataType>
std::vector<DataType> MakePotrsHostA(int64_t n) {
  std::vector<DataType> host(static_cast<size_t>(n * n));
  for (int64_t col = 0; col < n; ++col) {
    for (int64_t row = 0; row < n; ++row) {
      host[static_cast<size_t>(row + col * n)] =
          (row == col) ? RealScalar<DataType>(n + row + 1)
                       : RealScalar<DataType>(0.0);
    }
  }
  return host;
}

template <typename DataType>
std::vector<DataType> MakePotrsHostB(int64_t n, int64_t nrhs) {
  std::vector<DataType> host(static_cast<size_t>(n * nrhs));
  for (int64_t col = 0; col < nrhs; ++col) {
    for (int64_t row = 0; row < n; ++row) {
      host[static_cast<size_t>(row + col * n)] = ProbeRhsValue<DataType>();
    }
  }
  return host;
}

template <typename DataType>
double MaxPotrsSolutionError(const std::vector<DataType>& solution, int64_t n,
                             int64_t nrhs) {
  double max_error = 0.0;
  for (int64_t col = 0; col < nrhs; ++col) {
    for (int64_t row = 0; row < n; ++row) {
      max_error = std::max(
          max_error,
          AbsoluteDifference<DataType>(
              solution[static_cast<size_t>(row + col * n)],
              ExpectedProbeSolution<DataType>(row, n)));
    }
  }
  return max_error;
}

absl::Status CopyAnyBufferToOutputIfNeeded(cudaStream_t cuda_stream,
                                           ffi::AnyBuffer input,
                                           ffi::Result<ffi::AnyBuffer> output) {
  if (input.untyped_data() == output->untyped_data()) {
    return absl::OkStatus();
  }
  if (input.size_bytes() != output->size_bytes()) {
    return absl::InvalidArgumentError(
        "cuSOLVERMp distributed-input probe received mismatched input/output "
        "buffer sizes");
  }
  JAXMG_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
      output->untyped_data(), input.untyped_data(), input.size_bytes(),
      cudaMemcpyDeviceToDevice, cuda_stream));
  return absl::OkStatus();
}

template <typename DataType>
absl::Status RunCusolverMpScatterLayoutProbe(
    const CusolverMpApi& api, CusolverMpOpaqueHandle handle,
    CusolverMpOpaqueGrid grid, cudaStream_t cuda_stream, int64_t logical_rows,
    int64_t logical_cols, int64_t tile_rows, int64_t tile_cols,
    int64_t local_physical_rows, int64_t local_physical_cols,
    int32_t process_row, int32_t process_col, int32_t nccl_rank,
    ffi::Result<ffi::AnyBuffer> matrix_out,
    std::array<int32_t, kScatterProbeSize>* probe) {
  const int64_t local_rows =
      LocalNumroc(logical_rows, tile_rows, process_row,
                  static_cast<int32_t>((*probe)[4]));
  const int64_t local_cols =
      LocalNumroc(logical_cols, tile_cols, process_col,
                  static_cast<int32_t>((*probe)[5]));
  (*probe)[19] = static_cast<int32_t>(local_rows);
  (*probe)[20] = static_cast<int32_t>(local_cols);
  (*probe)[21] = static_cast<int32_t>(local_physical_rows);
  if (local_physical_rows < local_rows || local_physical_cols < local_cols) {
    (*probe)[0] = kOutputShapeMismatch;
    return absl::OkStatus();
  }

  CusolverMpOpaqueMatrixDesc desc = nullptr;
  cusolverStatus_t status = api.create_matrix_desc(
      &desc, grid, SolverTraits<DataType>::cuda_data_type, logical_rows,
      logical_cols, tile_rows, tile_cols, /*RSRC_A=*/0, /*CSRC_A=*/0,
      local_physical_rows);
  if (status != CUSOLVER_STATUS_SUCCESS || desc == nullptr) {
    (*probe)[0] = kCreateMatrixDescFailed;
    (*probe)[11] = static_cast<int32_t>(status);
    return absl::OkStatus();
  }
  (*probe)[10] = 1;

  JAXMG_RETURN_IF_CUDA_ERROR(cudaMemsetAsync(
      matrix_out->untyped_data(), 0, matrix_out->size_bytes(), cuda_stream));

  std::vector<DataType> host_source;
  const void* host_ptr = nullptr;
  if (nccl_rank == 0) {
    host_source = MakeScatterHostMatrix<DataType>(logical_rows, logical_cols);
    host_ptr = host_source.data();
  }

  status = api.scatter_h2d(handle, logical_rows, logical_cols,
                           matrix_out->untyped_data(), /*IA=*/1, /*JA=*/1,
                           desc, /*root=*/0, host_ptr,
                           /*h_ldsrc=*/logical_rows);
  if (status != CUSOLVER_STATUS_SUCCESS) {
    (*probe)[0] = kScatterFailed;
    (*probe)[11] = static_cast<int32_t>(status);
    api.destroy_matrix_desc(desc);
    return absl::OkStatus();
  }
  (*probe)[22] = 1;

  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));

  status = api.destroy_matrix_desc(desc);
  if (status != CUSOLVER_STATUS_SUCCESS) {
    (*probe)[0] = kDestroyMatrixDescFailed;
    (*probe)[11] = static_cast<int32_t>(status);
  }
  return absl::OkStatus();
}

template <typename DataType>
absl::Status RunCusolverMpPotrsProbe(
    const CusolverMpApi& api, CusolverMpOpaqueHandle handle,
    CusolverMpOpaqueGrid grid, cudaStream_t cuda_stream, int64_t n,
    int64_t tile_size, int64_t local_physical_rows,
    int64_t local_physical_cols_a, int64_t local_physical_cols_b,
    int32_t process_row, int32_t process_col, int32_t nccl_rank,
    ffi::Result<ffi::AnyBuffer> a_out, ffi::Result<ffi::AnyBuffer> b_out,
    std::array<int32_t, kPotrsProbeSize>* probe) {
  constexpr int64_t kNrhs = 1;

  const int32_t process_rows = (*probe)[4];
  const int32_t process_cols = (*probe)[5];
  const int64_t local_rows = LocalNumroc(n, tile_size, process_row,
                                         static_cast<int32_t>(process_rows));
  const int64_t local_cols_a = LocalNumroc(n, tile_size, process_col,
                                           static_cast<int32_t>(process_cols));
  const int64_t local_cols_b = LocalNumroc(kNrhs, tile_size, process_col,
                                           static_cast<int32_t>(process_cols));
  (*probe)[18] = static_cast<int32_t>(local_rows);
  (*probe)[19] = static_cast<int32_t>(local_cols_a);
  (*probe)[20] = static_cast<int32_t>(local_rows);
  (*probe)[21] = static_cast<int32_t>(local_cols_b);
  if (local_physical_rows < local_rows ||
      local_physical_cols_a < local_cols_a ||
      local_physical_cols_b < local_cols_b) {
    (*probe)[0] = kOutputShapeMismatch;
    return absl::OkStatus();
  }

  CusolverMpOpaqueMatrixDesc desc_a = nullptr;
  CusolverMpOpaqueMatrixDesc desc_b = nullptr;
  cusolverStatus_t status = api.create_matrix_desc(
      &desc_a, grid, SolverTraits<DataType>::cuda_data_type, n, n, tile_size,
      tile_size, /*RSRC_A=*/0, /*CSRC_A=*/0, local_physical_rows);
  if (status != CUSOLVER_STATUS_SUCCESS || desc_a == nullptr) {
    (*probe)[0] = kCreateMatrixDescFailed;
    (*probe)[11] = static_cast<int32_t>(status);
    return absl::OkStatus();
  }
  (*probe)[10] = 1;

  status = api.create_matrix_desc(
      &desc_b, grid, SolverTraits<DataType>::cuda_data_type, n, kNrhs,
      tile_size, tile_size, /*RSRC_B=*/0, /*CSRC_B=*/0,
      local_physical_rows);
  if (status != CUSOLVER_STATUS_SUCCESS || desc_b == nullptr) {
    (*probe)[0] = kCreateMatrixDescFailed;
    (*probe)[11] = static_cast<int32_t>(status);
    api.destroy_matrix_desc(desc_a);
    return absl::OkStatus();
  }

  void* d_potrf_work = nullptr;
  void* d_potrs_work = nullptr;
  void* h_potrf_work = nullptr;
  void* h_potrs_work = nullptr;
  int* d_potrf_info = nullptr;
  int* d_potrs_info = nullptr;
  auto cleanup = [&]() {
    if (d_potrf_work != nullptr) cudaFree(d_potrf_work);
    if (d_potrs_work != nullptr) cudaFree(d_potrs_work);
    if (d_potrf_info != nullptr) cudaFree(d_potrf_info);
    if (d_potrs_info != nullptr) cudaFree(d_potrs_info);
    if (h_potrf_work != nullptr) std::free(h_potrf_work);
    if (h_potrs_work != nullptr) std::free(h_potrs_work);
    api.destroy_matrix_desc(desc_b);
    api.destroy_matrix_desc(desc_a);
  };

  size_t potrf_workspace_device = 0;
  size_t potrf_workspace_host = 0;
  size_t potrs_workspace_device = 0;
  size_t potrs_workspace_host = 0;
  status = api.potrf_buffer_size(
      handle, CUBLAS_FILL_MODE_LOWER, n, a_out->untyped_data(), /*ia=*/1,
      /*ja=*/1, desc_a, SolverTraits<DataType>::cuda_data_type,
      &potrf_workspace_device, &potrf_workspace_host);
  if (status != CUSOLVER_STATUS_SUCCESS) {
    (*probe)[0] = kPotrfWorkspaceFailed;
    (*probe)[11] = static_cast<int32_t>(status);
    cleanup();
    return absl::OkStatus();
  }
  (*probe)[22] = SizeToKiBForProbe(potrf_workspace_device);
  (*probe)[23] = SizeToKiBForProbe(potrf_workspace_host);

  status = api.potrs_buffer_size(
      handle, CUBLAS_FILL_MODE_LOWER, n, kNrhs, a_out->untyped_data(),
      /*ia=*/1, /*ja=*/1, desc_a, b_out->untyped_data(), /*ib=*/1,
      /*jb=*/1, desc_b, SolverTraits<DataType>::cuda_data_type,
      &potrs_workspace_device, &potrs_workspace_host);
  if (status != CUSOLVER_STATUS_SUCCESS) {
    (*probe)[0] = kPotrsWorkspaceFailed;
    (*probe)[11] = static_cast<int32_t>(status);
    cleanup();
    return absl::OkStatus();
  }
  (*probe)[24] = SizeToKiBForProbe(potrs_workspace_device);
  (*probe)[25] = SizeToKiBForProbe(potrs_workspace_host);

  cudaError_t cuda_status = cudaSuccess;
  if (potrf_workspace_device > 0) {
    cuda_status = cudaMalloc(&d_potrf_work, potrf_workspace_device);
    if (cuda_status != cudaSuccess) {
      (*probe)[0] = kDeviceAllocFailed;
      cleanup();
      return absl::OkStatus();
    }
  }
  if (potrs_workspace_device > 0) {
    cuda_status = cudaMalloc(&d_potrs_work, potrs_workspace_device);
    if (cuda_status != cudaSuccess) {
      (*probe)[0] = kDeviceAllocFailed;
      cleanup();
      return absl::OkStatus();
    }
  }
  cuda_status = cudaMalloc(reinterpret_cast<void**>(&d_potrf_info),
                           sizeof(int));
  if (cuda_status != cudaSuccess) {
    (*probe)[0] = kDeviceAllocFailed;
    cleanup();
    return absl::OkStatus();
  }
  cuda_status = cudaMalloc(reinterpret_cast<void**>(&d_potrs_info),
                           sizeof(int));
  if (cuda_status != cudaSuccess) {
    (*probe)[0] = kDeviceAllocFailed;
    cleanup();
    return absl::OkStatus();
  }
  if (potrf_workspace_host > 0) {
    h_potrf_work = std::malloc(potrf_workspace_host);
    if (h_potrf_work == nullptr) {
      (*probe)[0] = kHostAllocFailed;
      cleanup();
      return absl::OkStatus();
    }
  }
  if (potrs_workspace_host > 0) {
    h_potrs_work = std::malloc(potrs_workspace_host);
    if (h_potrs_work == nullptr) {
      (*probe)[0] = kHostAllocFailed;
      cleanup();
      return absl::OkStatus();
    }
  }

  JAXMG_RETURN_IF_CUDA_ERROR(cudaMemsetAsync(
      a_out->untyped_data(), 0, a_out->size_bytes(), cuda_stream));
  JAXMG_RETURN_IF_CUDA_ERROR(cudaMemsetAsync(
      b_out->untyped_data(), 0, b_out->size_bytes(), cuda_stream));
  JAXMG_RETURN_IF_CUDA_ERROR(
      cudaMemsetAsync(d_potrf_info, 0, sizeof(int), cuda_stream));
  JAXMG_RETURN_IF_CUDA_ERROR(
      cudaMemsetAsync(d_potrs_info, 0, sizeof(int), cuda_stream));

  std::vector<DataType> host_a;
  std::vector<DataType> host_b;
  const void* host_a_ptr = nullptr;
  const void* host_b_ptr = nullptr;
  if (nccl_rank == 0) {
    host_a = MakePotrsHostA<DataType>(n);
    host_b = MakePotrsHostB<DataType>(n, kNrhs);
    host_a_ptr = host_a.data();
    host_b_ptr = host_b.data();
  }

  status = api.scatter_h2d(handle, n, n, a_out->untyped_data(), /*IA=*/1,
                           /*JA=*/1, desc_a, /*root=*/0, host_a_ptr,
                           /*h_ldsrc=*/n);
  if (status != CUSOLVER_STATUS_SUCCESS) {
    (*probe)[0] = kScatterFailed;
    (*probe)[11] = static_cast<int32_t>(status);
    cleanup();
    return absl::OkStatus();
  }
  (*probe)[30] = 1;

  status = api.scatter_h2d(handle, n, kNrhs, b_out->untyped_data(), /*IA=*/1,
                           /*JA=*/1, desc_b, /*root=*/0, host_b_ptr,
                           /*h_ldsrc=*/n);
  if (status != CUSOLVER_STATUS_SUCCESS) {
    (*probe)[0] = kScatterFailed;
    (*probe)[11] = static_cast<int32_t>(status);
    cleanup();
    return absl::OkStatus();
  }
  (*probe)[31] = 1;

  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));

  status = api.potrf(handle, CUBLAS_FILL_MODE_LOWER, n, a_out->untyped_data(),
                     /*ia=*/1, /*ja=*/1, desc_a,
                     SolverTraits<DataType>::cuda_data_type, d_potrf_work,
                     potrf_workspace_device, h_potrf_work,
                     potrf_workspace_host, d_potrf_info);
  if (status != CUSOLVER_STATUS_SUCCESS) {
    (*probe)[0] = kPotrfFailed;
    (*probe)[11] = static_cast<int32_t>(status);
    cleanup();
    return absl::OkStatus();
  }
  (*probe)[26] = 1;
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));

  int h_potrf_info = -1;
  JAXMG_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
      &h_potrf_info, d_potrf_info, sizeof(int), cudaMemcpyDeviceToHost,
      cuda_stream));
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));
  (*probe)[27] = h_potrf_info;
  if (h_potrf_info != 0) {
    (*probe)[0] = kPotrfInfoNonzero;
    cleanup();
    return absl::OkStatus();
  }

  status = api.potrs(handle, CUBLAS_FILL_MODE_LOWER, n, kNrhs,
                     a_out->untyped_data(), /*ia=*/1, /*ja=*/1, desc_a,
                     b_out->untyped_data(), /*ib=*/1, /*jb=*/1, desc_b,
                     SolverTraits<DataType>::cuda_data_type, d_potrs_work,
                     potrs_workspace_device, h_potrs_work,
                     potrs_workspace_host, d_potrs_info);
  if (status != CUSOLVER_STATUS_SUCCESS) {
    (*probe)[0] = kPotrsFailed;
    (*probe)[11] = static_cast<int32_t>(status);
    cleanup();
    return absl::OkStatus();
  }
  (*probe)[28] = 1;
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));

  int h_potrs_info = -1;
  JAXMG_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
      &h_potrs_info, d_potrs_info, sizeof(int), cudaMemcpyDeviceToHost,
      cuda_stream));
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));
  (*probe)[29] = h_potrs_info;
  if (h_potrs_info != 0) {
    (*probe)[0] = kPotrsInfoNonzero;
    cleanup();
    return absl::OkStatus();
  }

  std::vector<DataType> host_solution;
  void* host_solution_ptr = nullptr;
  if (nccl_rank == 0) {
    host_solution.resize(static_cast<size_t>(n));
    host_solution_ptr = host_solution.data();
  }
  status = api.gather_d2h(handle, n, kNrhs, b_out->untyped_data(), /*IA=*/1,
                          /*JA=*/1, desc_b, /*root=*/0, host_solution_ptr,
                          /*h_lddst=*/n);
  if (status != CUSOLVER_STATUS_SUCCESS) {
    (*probe)[0] = kGatherFailed;
    (*probe)[11] = static_cast<int32_t>(status);
    cleanup();
    return absl::OkStatus();
  }
  (*probe)[32] = 1;
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));

  if (nccl_rank == 0) {
    const double max_error =
        MaxPotrsSolutionError<DataType>(host_solution, n, kNrhs);
    const double scaled = std::ceil(max_error * kResidualScale);
    (*probe)[33] = static_cast<int32_t>(
        std::min<double>(scaled, std::numeric_limits<int32_t>::max()));
    if ((*probe)[33] > 10) {
      (*probe)[0] = kResidualTooLarge;
    }
  }

  cleanup();
  return absl::OkStatus();
}

template <typename DataType>
absl::Status RunCusolverMpDistributedPotrsProbe(
    const CusolverMpApi& api, CusolverMpOpaqueHandle handle,
    CusolverMpOpaqueGrid grid, cudaStream_t cuda_stream, int64_t n,
    int64_t nrhs, int64_t tile_size, int64_t local_physical_rows,
    int64_t local_physical_cols_a, int64_t local_physical_cols_b,
    int32_t process_row, int32_t process_col, int32_t nccl_rank,
    ffi::AnyBuffer a, ffi::AnyBuffer b, ffi::Result<ffi::AnyBuffer> a_out,
    ffi::Result<ffi::AnyBuffer> b_out,
    std::array<int32_t, kPotrsProbeSize>* probe,
    bool validate_solution = true) {
  const int32_t process_rows = (*probe)[4];
  const int32_t process_cols = (*probe)[5];
  const int64_t local_rows = LocalNumroc(n, tile_size, process_row,
                                         static_cast<int32_t>(process_rows));
  const int64_t local_cols_a = LocalNumroc(n, tile_size, process_col,
                                           static_cast<int32_t>(process_cols));
  const int64_t local_cols_b = LocalNumroc(nrhs, tile_size, process_col,
                                           static_cast<int32_t>(process_cols));
  (*probe)[18] = static_cast<int32_t>(local_rows);
  (*probe)[19] = static_cast<int32_t>(local_cols_a);
  (*probe)[20] = static_cast<int32_t>(local_rows);
  (*probe)[21] = static_cast<int32_t>(local_cols_b);
  (*probe)[36] = static_cast<int32_t>(nrhs);
  (*probe)[37] = 1;  // A came from a JAX/native redistribution, not scatter.
  (*probe)[38] = 1;  // B came from a JAX/native redistribution, not scatter.
  if (local_physical_rows < local_rows ||
      local_physical_cols_a < local_cols_a ||
      local_physical_cols_b < local_cols_b) {
    (*probe)[0] = kOutputShapeMismatch;
    return absl::OkStatus();
  }

  CusolverMpOpaqueMatrixDesc desc_a = nullptr;
  CusolverMpOpaqueMatrixDesc desc_b = nullptr;
  cusolverStatus_t status = api.create_matrix_desc(
      &desc_a, grid, SolverTraits<DataType>::cuda_data_type, n, n, tile_size,
      tile_size, /*RSRC_A=*/0, /*CSRC_A=*/0, local_physical_rows);
  if (status != CUSOLVER_STATUS_SUCCESS || desc_a == nullptr) {
    (*probe)[0] = kCreateMatrixDescFailed;
    (*probe)[11] = static_cast<int32_t>(status);
    return absl::OkStatus();
  }
  (*probe)[10] = 1;

  status = api.create_matrix_desc(
      &desc_b, grid, SolverTraits<DataType>::cuda_data_type, n, nrhs,
      tile_size, tile_size, /*RSRC_B=*/0, /*CSRC_B=*/0,
      local_physical_rows);
  if (status != CUSOLVER_STATUS_SUCCESS || desc_b == nullptr) {
    (*probe)[0] = kCreateMatrixDescFailed;
    (*probe)[11] = static_cast<int32_t>(status);
    api.destroy_matrix_desc(desc_a);
    return absl::OkStatus();
  }

  void* d_potrf_work = nullptr;
  void* d_potrs_work = nullptr;
  void* h_potrf_work = nullptr;
  void* h_potrs_work = nullptr;
  int* d_potrf_info = nullptr;
  int* d_potrs_info = nullptr;
  auto cleanup = [&]() {
    if (d_potrf_work != nullptr) cudaFree(d_potrf_work);
    if (d_potrs_work != nullptr) cudaFree(d_potrs_work);
    if (d_potrf_info != nullptr) cudaFree(d_potrf_info);
    if (d_potrs_info != nullptr) cudaFree(d_potrs_info);
    if (h_potrf_work != nullptr) std::free(h_potrf_work);
    if (h_potrs_work != nullptr) std::free(h_potrs_work);
    api.destroy_matrix_desc(desc_b);
    api.destroy_matrix_desc(desc_a);
  };

  size_t potrf_workspace_device = 0;
  size_t potrf_workspace_host = 0;
  size_t potrs_workspace_device = 0;
  size_t potrs_workspace_host = 0;
  status = api.potrf_buffer_size(
      handle, CUBLAS_FILL_MODE_LOWER, n, a_out->untyped_data(), /*ia=*/1,
      /*ja=*/1, desc_a, SolverTraits<DataType>::cuda_data_type,
      &potrf_workspace_device, &potrf_workspace_host);
  if (status != CUSOLVER_STATUS_SUCCESS) {
    (*probe)[0] = kPotrfWorkspaceFailed;
    (*probe)[11] = static_cast<int32_t>(status);
    cleanup();
    return absl::OkStatus();
  }
  (*probe)[22] = SizeToKiBForProbe(potrf_workspace_device);
  (*probe)[23] = SizeToKiBForProbe(potrf_workspace_host);

  status = api.potrs_buffer_size(
      handle, CUBLAS_FILL_MODE_LOWER, n, nrhs, a_out->untyped_data(),
      /*ia=*/1, /*ja=*/1, desc_a, b_out->untyped_data(), /*ib=*/1,
      /*jb=*/1, desc_b, SolverTraits<DataType>::cuda_data_type,
      &potrs_workspace_device, &potrs_workspace_host);
  if (status != CUSOLVER_STATUS_SUCCESS) {
    (*probe)[0] = kPotrsWorkspaceFailed;
    (*probe)[11] = static_cast<int32_t>(status);
    cleanup();
    return absl::OkStatus();
  }
  (*probe)[24] = SizeToKiBForProbe(potrs_workspace_device);
  (*probe)[25] = SizeToKiBForProbe(potrs_workspace_host);

  cudaError_t cuda_status = cudaSuccess;
  if (potrf_workspace_device > 0) {
    cuda_status = cudaMalloc(&d_potrf_work, potrf_workspace_device);
    if (cuda_status != cudaSuccess) {
      (*probe)[0] = kDeviceAllocFailed;
      cleanup();
      return absl::OkStatus();
    }
  }
  if (potrs_workspace_device > 0) {
    cuda_status = cudaMalloc(&d_potrs_work, potrs_workspace_device);
    if (cuda_status != cudaSuccess) {
      (*probe)[0] = kDeviceAllocFailed;
      cleanup();
      return absl::OkStatus();
    }
  }
  cuda_status = cudaMalloc(reinterpret_cast<void**>(&d_potrf_info),
                           sizeof(int));
  if (cuda_status != cudaSuccess) {
    (*probe)[0] = kDeviceAllocFailed;
    cleanup();
    return absl::OkStatus();
  }
  cuda_status = cudaMalloc(reinterpret_cast<void**>(&d_potrs_info),
                           sizeof(int));
  if (cuda_status != cudaSuccess) {
    (*probe)[0] = kDeviceAllocFailed;
    cleanup();
    return absl::OkStatus();
  }
  if (potrf_workspace_host > 0) {
    h_potrf_work = std::malloc(potrf_workspace_host);
    if (h_potrf_work == nullptr) {
      (*probe)[0] = kHostAllocFailed;
      cleanup();
      return absl::OkStatus();
    }
  }
  if (potrs_workspace_host > 0) {
    h_potrs_work = std::malloc(potrs_workspace_host);
    if (h_potrs_work == nullptr) {
      (*probe)[0] = kHostAllocFailed;
      cleanup();
      return absl::OkStatus();
    }
  }

  JAXMG_RETURN_IF_ERROR(CopyAnyBufferToOutputIfNeeded(cuda_stream, a, a_out));
  JAXMG_RETURN_IF_ERROR(CopyAnyBufferToOutputIfNeeded(cuda_stream, b, b_out));
  JAXMG_RETURN_IF_CUDA_ERROR(
      cudaMemsetAsync(d_potrf_info, 0, sizeof(int), cuda_stream));
  JAXMG_RETURN_IF_CUDA_ERROR(
      cudaMemsetAsync(d_potrs_info, 0, sizeof(int), cuda_stream));
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));

  status = api.potrf(handle, CUBLAS_FILL_MODE_LOWER, n, a_out->untyped_data(),
                     /*ia=*/1, /*ja=*/1, desc_a,
                     SolverTraits<DataType>::cuda_data_type, d_potrf_work,
                     potrf_workspace_device, h_potrf_work,
                     potrf_workspace_host, d_potrf_info);
  if (status != CUSOLVER_STATUS_SUCCESS) {
    (*probe)[0] = kPotrfFailed;
    (*probe)[11] = static_cast<int32_t>(status);
    cleanup();
    return absl::OkStatus();
  }
  (*probe)[26] = 1;
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));

  int h_potrf_info = -1;
  JAXMG_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
      &h_potrf_info, d_potrf_info, sizeof(int), cudaMemcpyDeviceToHost,
      cuda_stream));
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));
  (*probe)[27] = h_potrf_info;
  if (h_potrf_info != 0) {
    (*probe)[0] = kPotrfInfoNonzero;
    cleanup();
    return absl::OkStatus();
  }

  status = api.potrs(handle, CUBLAS_FILL_MODE_LOWER, n, nrhs,
                     a_out->untyped_data(), /*ia=*/1, /*ja=*/1, desc_a,
                     b_out->untyped_data(), /*ib=*/1, /*jb=*/1, desc_b,
                     SolverTraits<DataType>::cuda_data_type, d_potrs_work,
                     potrs_workspace_device, h_potrs_work,
                     potrs_workspace_host, d_potrs_info);
  if (status != CUSOLVER_STATUS_SUCCESS) {
    (*probe)[0] = kPotrsFailed;
    (*probe)[11] = static_cast<int32_t>(status);
    cleanup();
    return absl::OkStatus();
  }
  (*probe)[28] = 1;
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));

  int h_potrs_info = -1;
  JAXMG_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
      &h_potrs_info, d_potrs_info, sizeof(int), cudaMemcpyDeviceToHost,
      cuda_stream));
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));
  (*probe)[29] = h_potrs_info;
  if (h_potrs_info != 0) {
    (*probe)[0] = kPotrsInfoNonzero;
    cleanup();
    return absl::OkStatus();
  }

  if (validate_solution) {
    if (api.gather_d2h == nullptr) {
      (*probe)[0] = kGatherSymbolMissing;
      cleanup();
      return absl::OkStatus();
    }
    std::vector<DataType> host_solution;
    void* host_solution_ptr = nullptr;
    if (nccl_rank == 0) {
      host_solution.resize(static_cast<size_t>(n * nrhs));
      host_solution_ptr = host_solution.data();
    }
    status = api.gather_d2h(handle, n, nrhs, b_out->untyped_data(), /*IA=*/1,
                            /*JA=*/1, desc_b, /*root=*/0, host_solution_ptr,
                            /*h_lddst=*/n);
    if (status != CUSOLVER_STATUS_SUCCESS) {
      (*probe)[0] = kGatherFailed;
      (*probe)[11] = static_cast<int32_t>(status);
      cleanup();
      return absl::OkStatus();
    }
    (*probe)[32] = 1;
    JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));

    if (nccl_rank == 0) {
      const double max_error =
          MaxPotrsSolutionError<DataType>(host_solution, n, nrhs);
      const double scaled = std::ceil(max_error * kResidualScale);
      (*probe)[33] = static_cast<int32_t>(
          std::min<double>(scaled, std::numeric_limits<int32_t>::max()));
      if ((*probe)[33] > 10) {
        (*probe)[0] = kResidualTooLarge;
      }
    }
  }

  cleanup();
  return absl::OkStatus();
}

}  // namespace

absl::Status XlaCusolverMpInitProbePrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests) {
  if (collective_params == nullptr || clique_requests == nullptr) {
    return absl::InvalidArgumentError(
        "cusolvermp_init_probe requires XLA collective prepare contexts");
  }

  absl::StatusOr<GpuCliqueKey> clique_key =
      AllAssignedDevicesP2PCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    return clique_key.status();
  }
  return clique_requests->RequestClique(
      *clique_key, {AllAssignedGlobalDeviceGroup(*collective_params)});
}

absl::Status XlaCusolverMpScatterLayoutProbePrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests) {
  return XlaCusolverMpInitProbePrepare(collective_params, clique_requests);
}

absl::Status XlaCusolverMpPotrsProbePrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests) {
  return XlaCusolverMpInitProbePrepare(collective_params, clique_requests);
}

absl::Status XlaCusolverMpDistributedPotrsProbePrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests) {
  return XlaCusolverMpInitProbePrepare(collective_params, clique_requests);
}

absl::Status XlaCusolverMpInitProbeDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    int64_t process_rows, int64_t process_cols, int64_t matrix_rows,
    int64_t matrix_cols, int64_t tile_rows, int64_t tile_cols,
    ffi::AnyBuffer token, ffi::Result<ffi::BufferR1<S32>> out,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  if (stream == nullptr) {
    return absl::InvalidArgumentError(
        "cusolvermp_init_probe requires an XLA stream context");
  }
  if (out->dimensions().size() != 1 || out->dimensions()[0] != kProbeSize) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "cusolvermp_init_probe expects output shape (%d,), got rank %d",
        kProbeSize, out->dimensions().size()));
  }

  std::array<int32_t, kProbeSize> probe = {
      kProbeOk,
      -1,  // CUDA device selected for this FFI invocation.
      -1,  // NCCL rank reported by the borrowed communicator.
      -1,  // NCCL communicator size.
      static_cast<int32_t>(process_rows),
      static_cast<int32_t>(process_cols),
      -1,  // cuSOLVERMp version, if the library is present.
      0,   // libcusolverMp loaded.
      0,   // cusolverMpHandle_t created.
      0,   // cusolverMpGrid_t created.
      0,   // cusolverMpMatrixDescriptor_t created.
      0,   // raw cuSOLVER status from the failing call, if any.
      static_cast<int32_t>(token.size_bytes()),
      static_cast<int32_t>(matrix_rows),
      static_cast<int32_t>(matrix_cols),
      static_cast<int32_t>(tile_rows),
  };

  int cuda_device = -1;
  cudaError_t cuda_status = cudaGetDevice(&cuda_device);
  if (cuda_status != cudaSuccess) {
    probe[0] = kCudaDeviceFailed;
    return CopyProbeToDevice(stream, probe, out);
  }
  probe[1] = cuda_device;

  if (collective_params == nullptr || collective_cliques == nullptr) {
    probe[0] = kCollectiveContextMissing;
    return CopyProbeToDevice(stream, probe, out);
  }

  absl::StatusOr<GpuCliqueKey> clique_key =
      AllAssignedDevicesP2PCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    probe[0] = kCliqueKeyFailed;
    return CopyProbeToDevice(stream, probe, out);
  }

  absl::StatusOr<GpuCommunicator*> gpu_comm = collective_cliques->GetComm(
      *clique_key, collective_params->global_device_id);
  if (!gpu_comm.ok() || *gpu_comm == nullptr) {
    probe[0] = kCommunicatorMissing;
    return CopyProbeToDevice(stream, probe, out);
  }

  void* platform_handle = (*gpu_comm)->platform_comm().handle;
  if (platform_handle == nullptr) {
    probe[0] = kNcclHandleMissing;
    return CopyProbeToDevice(stream, probe, out);
  }
  ncclComm_t nccl_comm = reinterpret_cast<ncclComm_t>(platform_handle);

  int nccl_rank = -1;
  int nccl_count = -1;
  ncclResult_t rank_status = ncclCommUserRank(nccl_comm, &nccl_rank);
  ncclResult_t count_status = ncclCommCount(nccl_comm, &nccl_count);
  if (rank_status != ncclSuccess || count_status != ncclSuccess) {
    probe[0] = kNcclRankMismatch;
    return CopyProbeToDevice(stream, probe, out);
  }
  probe[2] = nccl_rank;
  probe[3] = nccl_count;

  if (process_rows <= 0 || process_cols <= 0 ||
      process_rows * process_cols != nccl_count) {
    probe[0] = kGridShapeMismatch;
    return CopyProbeToDevice(stream, probe, out);
  }

  CusolverMpApi api = LoadCusolverMpApi(&probe);
  if (probe[0] != kProbeOk || !HasRequiredSymbols(api)) {
    if (api.library != nullptr) {
      dlclose(api.library);
    }
    return CopyProbeToDevice(stream, probe, out);
  }

  CusolverMpOpaqueHandle handle = nullptr;
  cusolverStatus_t status = api.create(&handle, cuda_device, cuda_stream);
  if (status != CUSOLVER_STATUS_SUCCESS || handle == nullptr) {
    probe[0] = kCreateHandleFailed;
    probe[11] = static_cast<int32_t>(status);
    dlclose(api.library);
    return CopyProbeToDevice(stream, probe, out);
  }
  probe[8] = 1;

  int version = -1;
  status = api.get_version(handle, &version);
  if (status != CUSOLVER_STATUS_SUCCESS) {
    probe[0] = kGetVersionFailed;
    probe[11] = static_cast<int32_t>(status);
    api.destroy(handle);
    dlclose(api.library);
    return CopyProbeToDevice(stream, probe, out);
  }
  probe[6] = version;

  CusolverMpOpaqueGrid grid = nullptr;
  status = api.create_grid(
      handle, &grid, nccl_comm, static_cast<int32_t>(process_rows),
      static_cast<int32_t>(process_cols), kCusolverMpGridMappingRowMajor);
  if (status != CUSOLVER_STATUS_SUCCESS || grid == nullptr) {
    probe[0] = kCreateGridFailed;
    probe[11] = static_cast<int32_t>(status);
    api.destroy(handle);
    dlclose(api.library);
    return CopyProbeToDevice(stream, probe, out);
  }
  probe[9] = 1;

  const int32_t process_row = nccl_rank / static_cast<int32_t>(process_cols);
  const int64_t local_rows =
      LocalNumroc(matrix_rows, tile_rows, process_row,
                  static_cast<int32_t>(process_rows));
  const int64_t lld = std::max<int64_t>(1, local_rows);

  CusolverMpOpaqueMatrixDesc desc = nullptr;
  status = api.create_matrix_desc(
      &desc, grid, CUDA_R_32F, matrix_rows, matrix_cols, tile_rows, tile_cols,
      /*RSRC_A=*/0, /*CSRC_A=*/0, lld);
  if (status != CUSOLVER_STATUS_SUCCESS || desc == nullptr) {
    probe[0] = kCreateMatrixDescFailed;
    probe[11] = static_cast<int32_t>(status);
    api.destroy_grid(grid);
    api.destroy(handle);
    dlclose(api.library);
    return CopyProbeToDevice(stream, probe, out);
  }
  probe[10] = 1;

  status = api.destroy_matrix_desc(desc);
  if (status != CUSOLVER_STATUS_SUCCESS) {
    probe[0] = kDestroyMatrixDescFailed;
    probe[11] = static_cast<int32_t>(status);
    api.destroy_grid(grid);
    api.destroy(handle);
    dlclose(api.library);
    return CopyProbeToDevice(stream, probe, out);
  }

  status = api.destroy_grid(grid);
  if (status != CUSOLVER_STATUS_SUCCESS) {
    probe[0] = kDestroyGridFailed;
    probe[11] = static_cast<int32_t>(status);
    api.destroy(handle);
    dlclose(api.library);
    return CopyProbeToDevice(stream, probe, out);
  }

  status = api.destroy(handle);
  if (status != CUSOLVER_STATUS_SUCCESS) {
    probe[0] = kDestroyHandleFailed;
    probe[11] = static_cast<int32_t>(status);
    dlclose(api.library);
    return CopyProbeToDevice(stream, probe, out);
  }

  dlclose(api.library);
  return CopyProbeToDevice(stream, probe, out);
}

absl::Status XlaCusolverMpScatterLayoutProbeDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    int64_t process_rows, int64_t process_cols, int64_t logical_rows,
    int64_t logical_cols, int64_t tile_rows, int64_t tile_cols,
    ffi::AnyBuffer matrix, ffi::Result<ffi::AnyBuffer> matrix_out,
    ffi::Result<ffi::BufferR1<S32>> status_out,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  if (stream == nullptr || cuda_stream == nullptr) {
    return absl::InvalidArgumentError(
        "cusolvermp_scatter_layout_probe requires XLA and CUDA streams");
  }
  if (matrix.dimensions().size() != 2 ||
      matrix_out->dimensions().size() != 2 ||
      matrix.dimensions()[0] != matrix_out->dimensions()[0] ||
      matrix.dimensions()[1] != matrix_out->dimensions()[1]) {
    return absl::InvalidArgumentError(
        "cusolvermp_scatter_layout_probe expects matching rank-2 matrix "
        "input/output");
  }
  if (matrix.element_type() != matrix_out->element_type()) {
    return absl::InvalidArgumentError(
        "cusolvermp_scatter_layout_probe requires matching matrix dtypes");
  }
  if (status_out->dimensions().size() != 1 ||
      status_out->dimensions()[0] != kScatterProbeSize) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "cusolvermp_scatter_layout_probe expects status shape (%d,)",
        kScatterProbeSize));
  }

  std::array<int32_t, kScatterProbeSize> probe = {
      kProbeOk,
      -1,  // CUDA device selected for this FFI invocation.
      -1,  // NCCL rank reported by the borrowed communicator.
      -1,  // NCCL communicator size.
      static_cast<int32_t>(process_rows),
      static_cast<int32_t>(process_cols),
      -1,  // cuSOLVERMp version, if available.
      0,   // libcusolverMp loaded.
      0,   // cusolverMpHandle_t created.
      0,   // cusolverMpGrid_t created.
      0,   // cusolverMpMatrixDescriptor_t created.
      0,   // raw cuSOLVER status from the failing call, if any.
      static_cast<int32_t>(matrix.size_bytes()),
      static_cast<int32_t>(logical_rows),
      static_cast<int32_t>(logical_cols),
      static_cast<int32_t>(tile_rows),
      static_cast<int32_t>(tile_cols),
      static_cast<int32_t>(matrix.dimensions()[0]),
      static_cast<int32_t>(matrix.dimensions()[1]),
      -1,  // local NUMROC rows.
      -1,  // local NUMROC cols.
      -1,  // LLD_A used in the descriptor.
      0,   // cusolverMpMatrixScatterH2D called.
      -1,  // dtype code.
  };

  int cuda_device = -1;
  cudaError_t cuda_status = cudaGetDevice(&cuda_device);
  if (cuda_status != cudaSuccess) {
    probe[0] = kCudaDeviceFailed;
    return CopyScatterProbeToDevice(stream, probe, status_out);
  }
  probe[1] = cuda_device;

  if (collective_params == nullptr || collective_cliques == nullptr) {
    probe[0] = kCollectiveContextMissing;
    return CopyScatterProbeToDevice(stream, probe, status_out);
  }

  absl::StatusOr<GpuCliqueKey> clique_key =
      AllAssignedDevicesP2PCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    probe[0] = kCliqueKeyFailed;
    return CopyScatterProbeToDevice(stream, probe, status_out);
  }

  absl::StatusOr<GpuCommunicator*> gpu_comm = collective_cliques->GetComm(
      *clique_key, collective_params->global_device_id);
  if (!gpu_comm.ok() || *gpu_comm == nullptr) {
    probe[0] = kCommunicatorMissing;
    return CopyScatterProbeToDevice(stream, probe, status_out);
  }

  void* platform_handle = (*gpu_comm)->platform_comm().handle;
  if (platform_handle == nullptr) {
    probe[0] = kNcclHandleMissing;
    return CopyScatterProbeToDevice(stream, probe, status_out);
  }
  ncclComm_t nccl_comm = reinterpret_cast<ncclComm_t>(platform_handle);

  int nccl_rank = -1;
  int nccl_count = -1;
  ncclResult_t rank_status = ncclCommUserRank(nccl_comm, &nccl_rank);
  ncclResult_t count_status = ncclCommCount(nccl_comm, &nccl_count);
  if (rank_status != ncclSuccess || count_status != ncclSuccess) {
    probe[0] = kNcclRankMismatch;
    return CopyScatterProbeToDevice(stream, probe, status_out);
  }
  probe[2] = nccl_rank;
  probe[3] = nccl_count;

  if (process_rows <= 0 || process_cols <= 0 ||
      process_rows * process_cols != nccl_count || logical_rows <= 0 ||
      logical_cols <= 0 || tile_rows <= 0 || tile_cols <= 0) {
    probe[0] = kGridShapeMismatch;
    return CopyScatterProbeToDevice(stream, probe, status_out);
  }

  CusolverMpApi api = LoadCusolverMpApi(&probe);
  if (probe[0] != kProbeOk || !HasRequiredSymbols(api)) {
    if (api.library != nullptr) {
      dlclose(api.library);
    }
    return CopyScatterProbeToDevice(stream, probe, status_out);
  }
  if (api.scatter_h2d == nullptr) {
    probe[0] = kScatterSymbolMissing;
    dlclose(api.library);
    return CopyScatterProbeToDevice(stream, probe, status_out);
  }

  CusolverMpOpaqueHandle handle = nullptr;
  cusolverStatus_t cusolver_status =
      api.create(&handle, cuda_device, cuda_stream);
  if (cusolver_status != CUSOLVER_STATUS_SUCCESS || handle == nullptr) {
    probe[0] = kCreateHandleFailed;
    probe[11] = static_cast<int32_t>(cusolver_status);
    dlclose(api.library);
    return CopyScatterProbeToDevice(stream, probe, status_out);
  }
  probe[8] = 1;

  int version = -1;
  cusolver_status = api.get_version(handle, &version);
  if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
    probe[0] = kGetVersionFailed;
    probe[11] = static_cast<int32_t>(cusolver_status);
    api.destroy(handle);
    dlclose(api.library);
    return CopyScatterProbeToDevice(stream, probe, status_out);
  }
  probe[6] = version;

  CusolverMpOpaqueGrid grid = nullptr;
  cusolver_status = api.create_grid(
      handle, &grid, nccl_comm, static_cast<int32_t>(process_rows),
      static_cast<int32_t>(process_cols), kCusolverMpGridMappingRowMajor);
  if (cusolver_status != CUSOLVER_STATUS_SUCCESS || grid == nullptr) {
    probe[0] = kCreateGridFailed;
    probe[11] = static_cast<int32_t>(cusolver_status);
    api.destroy(handle);
    dlclose(api.library);
    return CopyScatterProbeToDevice(stream, probe, status_out);
  }
  probe[9] = 1;

  const int32_t process_row = nccl_rank / static_cast<int32_t>(process_cols);
  const int32_t process_col = nccl_rank % static_cast<int32_t>(process_cols);
  probe[21] = static_cast<int32_t>(matrix.dimensions()[0]);

  absl::Status scatter_status;
  switch (matrix.element_type()) {
    case F32:
      probe[23] = 1;
      scatter_status = RunCusolverMpScatterLayoutProbe<float>(
          api, handle, grid, cuda_stream, logical_rows, logical_cols,
          tile_rows, tile_cols, matrix.dimensions()[0],
          matrix.dimensions()[1], process_row, process_col, nccl_rank,
          matrix_out, &probe);
      break;
    case F64:
      probe[23] = 2;
      scatter_status = RunCusolverMpScatterLayoutProbe<double>(
          api, handle, grid, cuda_stream, logical_rows, logical_cols,
          tile_rows, tile_cols, matrix.dimensions()[0],
          matrix.dimensions()[1], process_row, process_col, nccl_rank,
          matrix_out, &probe);
      break;
    case C64:
      probe[23] = 3;
      scatter_status = RunCusolverMpScatterLayoutProbe<cuFloatComplex>(
          api, handle, grid, cuda_stream, logical_rows, logical_cols,
          tile_rows, tile_cols, matrix.dimensions()[0],
          matrix.dimensions()[1], process_row, process_col, nccl_rank,
          matrix_out, &probe);
      break;
    case C128:
      probe[23] = 4;
      scatter_status = RunCusolverMpScatterLayoutProbe<cuDoubleComplex>(
          api, handle, grid, cuda_stream, logical_rows, logical_cols,
          tile_rows, tile_cols, matrix.dimensions()[0],
          matrix.dimensions()[1], process_row, process_col, nccl_rank,
          matrix_out, &probe);
      break;
    default:
      probe[0] = kUnsupportedDtype;
      break;
  }
  if (!scatter_status.ok()) {
    api.destroy_grid(grid);
    api.destroy(handle);
    dlclose(api.library);
    return scatter_status;
  }

  if (probe[0] == kProbeOk) {
    cusolver_status = api.destroy_grid(grid);
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
      probe[0] = kDestroyGridFailed;
      probe[11] = static_cast<int32_t>(cusolver_status);
    }
  } else {
    api.destroy_grid(grid);
  }

  if (probe[0] == kProbeOk) {
    cusolver_status = api.destroy(handle);
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
      probe[0] = kDestroyHandleFailed;
      probe[11] = static_cast<int32_t>(cusolver_status);
    }
  } else {
    api.destroy(handle);
  }

  dlclose(api.library);
  return CopyScatterProbeToDevice(stream, probe, status_out);
}

absl::Status XlaCusolverMpPotrsProbeDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    int64_t process_rows, int64_t process_cols, int64_t n, int64_t tile_size,
    ffi::AnyBuffer a, ffi::AnyBuffer b, ffi::Result<ffi::AnyBuffer> a_out,
    ffi::Result<ffi::AnyBuffer> b_out,
    ffi::Result<ffi::BufferR1<S32>> status_out,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  if (stream == nullptr || cuda_stream == nullptr) {
    return absl::InvalidArgumentError(
        "cusolvermp_potrs_probe requires XLA and CUDA streams");
  }
  if (a.dimensions().size() != 2 || b.dimensions().size() != 2 ||
      a_out->dimensions().size() != 2 || b_out->dimensions().size() != 2) {
    return absl::InvalidArgumentError(
        "cusolvermp_potrs_probe expects rank-2 A and B buffers");
  }
  if (a.element_type() != b.element_type() ||
      a.element_type() != a_out->element_type() ||
      b.element_type() != b_out->element_type()) {
    return absl::InvalidArgumentError(
        "cusolvermp_potrs_probe requires matching A/B dtypes");
  }
  if (a.dimensions()[0] != a_out->dimensions()[0] ||
      a.dimensions()[1] != a_out->dimensions()[1] ||
      b.dimensions()[0] != b_out->dimensions()[0] ||
      b.dimensions()[1] != b_out->dimensions()[1]) {
    return absl::InvalidArgumentError(
        "cusolvermp_potrs_probe input/output shapes must match");
  }
  if (a.dimensions()[0] != b.dimensions()[0]) {
    return absl::InvalidArgumentError(
        "cusolvermp_potrs_probe expects A and B to have the same local row "
        "capacity");
  }
  if (status_out->dimensions().size() != 1 ||
      status_out->dimensions()[0] != kPotrsProbeSize) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "cusolvermp_potrs_probe expects status shape (%d,)",
        kPotrsProbeSize));
  }

  std::array<int32_t, kPotrsProbeSize> probe = {
      kProbeOk,
      -1,  // CUDA device selected for this FFI invocation.
      -1,  // NCCL rank reported by the borrowed communicator.
      -1,  // NCCL communicator size.
      static_cast<int32_t>(process_rows),
      static_cast<int32_t>(process_cols),
      -1,  // cuSOLVERMp version, if available.
      0,   // libcusolverMp loaded.
      0,   // cusolverMpHandle_t created.
      0,   // cusolverMpGrid_t created.
      0,   // cusolverMpMatrixDescriptor_t for A created.
      0,   // raw cuSOLVER status from the failing call, if any.
      static_cast<int32_t>(a.size_bytes()),
      static_cast<int32_t>(n),
      static_cast<int32_t>(tile_size),
      static_cast<int32_t>(a.dimensions()[0]),
      static_cast<int32_t>(a.dimensions()[1]),
      static_cast<int32_t>(b.dimensions()[0]),
      -1,  // local NUMROC rows for A.
      -1,  // local NUMROC cols for A.
      -1,  // local NUMROC rows for B.
      -1,  // local NUMROC cols for B.
      -1,  // potrf device workspace, KiB.
      -1,  // potrf host workspace, KiB.
      -1,  // potrs device workspace, KiB.
      -1,  // potrs host workspace, KiB.
      0,   // cusolverMpPotrf called.
      -1,  // potrf info value copied from device.
      0,   // cusolverMpPotrs called.
      -1,  // potrs info value copied from device.
      0,   // A scatter called.
      0,   // B scatter called.
      0,   // B gather called.
      -1,  // rank-0 residual error scaled by 1e6.
      -1,  // dtype code.
      static_cast<int32_t>(b.dimensions()[1]),
      -1,
      -1,
      -1,
      -1,
  };

  int cuda_device = -1;
  cudaError_t cuda_status = cudaGetDevice(&cuda_device);
  if (cuda_status != cudaSuccess) {
    probe[0] = kCudaDeviceFailed;
    return CopyPotrsProbeToDevice(stream, probe, status_out);
  }
  probe[1] = cuda_device;

  if (collective_params == nullptr || collective_cliques == nullptr) {
    probe[0] = kCollectiveContextMissing;
    return CopyPotrsProbeToDevice(stream, probe, status_out);
  }

  absl::StatusOr<GpuCliqueKey> clique_key =
      AllAssignedDevicesP2PCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    probe[0] = kCliqueKeyFailed;
    return CopyPotrsProbeToDevice(stream, probe, status_out);
  }

  absl::StatusOr<GpuCommunicator*> gpu_comm = collective_cliques->GetComm(
      *clique_key, collective_params->global_device_id);
  if (!gpu_comm.ok() || *gpu_comm == nullptr) {
    probe[0] = kCommunicatorMissing;
    return CopyPotrsProbeToDevice(stream, probe, status_out);
  }

  void* platform_handle = (*gpu_comm)->platform_comm().handle;
  if (platform_handle == nullptr) {
    probe[0] = kNcclHandleMissing;
    return CopyPotrsProbeToDevice(stream, probe, status_out);
  }
  ncclComm_t nccl_comm = reinterpret_cast<ncclComm_t>(platform_handle);

  int nccl_rank = -1;
  int nccl_count = -1;
  ncclResult_t rank_status = ncclCommUserRank(nccl_comm, &nccl_rank);
  ncclResult_t count_status = ncclCommCount(nccl_comm, &nccl_count);
  if (rank_status != ncclSuccess || count_status != ncclSuccess) {
    probe[0] = kNcclRankMismatch;
    return CopyPotrsProbeToDevice(stream, probe, status_out);
  }
  probe[2] = nccl_rank;
  probe[3] = nccl_count;

  if (process_rows <= 0 || process_cols <= 0 ||
      process_rows * process_cols != nccl_count || n <= 0 || tile_size <= 0) {
    probe[0] = kGridShapeMismatch;
    return CopyPotrsProbeToDevice(stream, probe, status_out);
  }

  CusolverMpApi api = LoadCusolverMpApi(&probe);
  if (probe[0] != kProbeOk || !HasRequiredSymbols(api)) {
    if (api.library != nullptr) {
      dlclose(api.library);
    }
    return CopyPotrsProbeToDevice(stream, probe, status_out);
  }
  if (api.scatter_h2d == nullptr) {
    probe[0] = kScatterSymbolMissing;
    dlclose(api.library);
    return CopyPotrsProbeToDevice(stream, probe, status_out);
  }
  if (api.gather_d2h == nullptr) {
    probe[0] = kGatherSymbolMissing;
    dlclose(api.library);
    return CopyPotrsProbeToDevice(stream, probe, status_out);
  }
  if (!HasPotrsSymbols(api)) {
    probe[0] = kSolverSymbolMissing;
    dlclose(api.library);
    return CopyPotrsProbeToDevice(stream, probe, status_out);
  }

  CusolverMpOpaqueHandle handle = nullptr;
  cusolverStatus_t cusolver_status =
      api.create(&handle, cuda_device, cuda_stream);
  if (cusolver_status != CUSOLVER_STATUS_SUCCESS || handle == nullptr) {
    probe[0] = kCreateHandleFailed;
    probe[11] = static_cast<int32_t>(cusolver_status);
    dlclose(api.library);
    return CopyPotrsProbeToDevice(stream, probe, status_out);
  }
  probe[8] = 1;

  int version = -1;
  cusolver_status = api.get_version(handle, &version);
  if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
    probe[0] = kGetVersionFailed;
    probe[11] = static_cast<int32_t>(cusolver_status);
    api.destroy(handle);
    dlclose(api.library);
    return CopyPotrsProbeToDevice(stream, probe, status_out);
  }
  probe[6] = version;

  CusolverMpOpaqueGrid grid = nullptr;
  cusolver_status = api.create_grid(
      handle, &grid, nccl_comm, static_cast<int32_t>(process_rows),
      static_cast<int32_t>(process_cols), kCusolverMpGridMappingRowMajor);
  if (cusolver_status != CUSOLVER_STATUS_SUCCESS || grid == nullptr) {
    probe[0] = kCreateGridFailed;
    probe[11] = static_cast<int32_t>(cusolver_status);
    api.destroy(handle);
    dlclose(api.library);
    return CopyPotrsProbeToDevice(stream, probe, status_out);
  }
  probe[9] = 1;

  const int32_t process_row = nccl_rank / static_cast<int32_t>(process_cols);
  const int32_t process_col = nccl_rank % static_cast<int32_t>(process_cols);

  absl::Status potrs_status;
  switch (a.element_type()) {
    case F32:
      probe[34] = 1;
      potrs_status = RunCusolverMpPotrsProbe<float>(
          api, handle, grid, cuda_stream, n, tile_size, a.dimensions()[0],
          a.dimensions()[1], b.dimensions()[1], process_row, process_col,
          nccl_rank, a_out, b_out, &probe);
      break;
    case F64:
      probe[34] = 2;
      potrs_status = RunCusolverMpPotrsProbe<double>(
          api, handle, grid, cuda_stream, n, tile_size, a.dimensions()[0],
          a.dimensions()[1], b.dimensions()[1], process_row, process_col,
          nccl_rank, a_out, b_out, &probe);
      break;
    case C64:
      probe[34] = 3;
      potrs_status = RunCusolverMpPotrsProbe<cuFloatComplex>(
          api, handle, grid, cuda_stream, n, tile_size, a.dimensions()[0],
          a.dimensions()[1], b.dimensions()[1], process_row, process_col,
          nccl_rank, a_out, b_out, &probe);
      break;
    case C128:
      probe[34] = 4;
      potrs_status = RunCusolverMpPotrsProbe<cuDoubleComplex>(
          api, handle, grid, cuda_stream, n, tile_size, a.dimensions()[0],
          a.dimensions()[1], b.dimensions()[1], process_row, process_col,
          nccl_rank, a_out, b_out, &probe);
      break;
    default:
      probe[0] = kUnsupportedDtype;
      break;
  }
  if (!potrs_status.ok()) {
    api.destroy_grid(grid);
    api.destroy(handle);
    dlclose(api.library);
    return potrs_status;
  }

  if (probe[0] == kProbeOk) {
    cusolver_status = api.destroy_grid(grid);
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
      probe[0] = kDestroyGridFailed;
      probe[11] = static_cast<int32_t>(cusolver_status);
    }
  } else {
    api.destroy_grid(grid);
  }

  if (probe[0] == kProbeOk) {
    cusolver_status = api.destroy(handle);
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
      probe[0] = kDestroyHandleFailed;
      probe[11] = static_cast<int32_t>(cusolver_status);
    }
  } else {
    api.destroy(handle);
  }

  dlclose(api.library);
  return CopyPotrsProbeToDevice(stream, probe, status_out);
}

absl::Status CusolverMpDistributedPotrsDispatchImpl(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    int64_t process_rows, int64_t process_cols, int64_t n, int64_t nrhs,
    int64_t tile_size, ffi::AnyBuffer a, ffi::AnyBuffer b,
    ffi::Result<ffi::AnyBuffer> a_out, ffi::Result<ffi::AnyBuffer> b_out,
    ffi::Result<ffi::BufferR1<S32>> status_out,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques, bool validate_solution) {
  if (stream == nullptr || cuda_stream == nullptr) {
    return absl::InvalidArgumentError(
        "cusolvermp_distributed_potrs_probe requires XLA and CUDA streams");
  }
  if (a.dimensions().size() != 2 || b.dimensions().size() != 2 ||
      a_out->dimensions().size() != 2 || b_out->dimensions().size() != 2) {
    return absl::InvalidArgumentError(
        "cusolvermp_distributed_potrs_probe expects rank-2 A and B buffers");
  }
  if (a.element_type() != b.element_type() ||
      a.element_type() != a_out->element_type() ||
      b.element_type() != b_out->element_type()) {
    return absl::InvalidArgumentError(
        "cusolvermp_distributed_potrs_probe requires matching A/B dtypes");
  }
  if (a.dimensions()[0] != a_out->dimensions()[0] ||
      a.dimensions()[1] != a_out->dimensions()[1] ||
      b.dimensions()[0] != b_out->dimensions()[0] ||
      b.dimensions()[1] != b_out->dimensions()[1]) {
    return absl::InvalidArgumentError(
        "cusolvermp_distributed_potrs_probe input/output shapes must match");
  }
  if (a.dimensions()[0] != b.dimensions()[0]) {
    return absl::InvalidArgumentError(
        "cusolvermp_distributed_potrs_probe expects A and B to have the same "
        "local row capacity");
  }
  if (status_out->dimensions().size() != 1 ||
      status_out->dimensions()[0] != kPotrsProbeSize) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "cusolvermp_distributed_potrs_probe expects status shape (%d,)",
        kPotrsProbeSize));
  }

  std::array<int32_t, kPotrsProbeSize> probe = {
      kProbeOk,
      -1,  // CUDA device selected for this FFI invocation.
      -1,  // NCCL rank reported by the borrowed communicator.
      -1,  // NCCL communicator size.
      static_cast<int32_t>(process_rows),
      static_cast<int32_t>(process_cols),
      -1,  // cuSOLVERMp version, if available.
      0,   // libcusolverMp loaded.
      0,   // cusolverMpHandle_t created.
      0,   // cusolverMpGrid_t created.
      0,   // cusolverMpMatrixDescriptor_t for A created.
      0,   // raw cuSOLVER status from the failing call, if any.
      static_cast<int32_t>(a.size_bytes()),
      static_cast<int32_t>(n),
      static_cast<int32_t>(tile_size),
      static_cast<int32_t>(a.dimensions()[0]),
      static_cast<int32_t>(a.dimensions()[1]),
      static_cast<int32_t>(b.dimensions()[0]),
      -1,  // local NUMROC rows for A.
      -1,  // local NUMROC cols for A.
      -1,  // local NUMROC rows for B.
      -1,  // local NUMROC cols for B.
      -1,  // potrf device workspace, KiB.
      -1,  // potrf host workspace, KiB.
      -1,  // potrs device workspace, KiB.
      -1,  // potrs host workspace, KiB.
      0,   // cusolverMpPotrf called.
      -1,  // potrf info value copied from device.
      0,   // cusolverMpPotrs called.
      -1,  // potrs info value copied from device.
      0,   // A scatter called.
      0,   // B scatter called.
      0,   // B gather called.
      -1,  // rank-0 residual error scaled by 1e6.
      -1,  // dtype code.
      static_cast<int32_t>(b.dimensions()[1]),
      static_cast<int32_t>(nrhs),
      0,  // A came from native JAX-buffer redistribution.
      0,  // B came from native JAX-buffer redistribution.
      -1,
  };

  int cuda_device = -1;
  cudaError_t cuda_status = cudaGetDevice(&cuda_device);
  if (cuda_status != cudaSuccess) {
    probe[0] = kCudaDeviceFailed;
    return CopyPotrsProbeToDevice(stream, probe, status_out);
  }
  probe[1] = cuda_device;

  if (collective_params == nullptr || collective_cliques == nullptr) {
    probe[0] = kCollectiveContextMissing;
    return CopyPotrsProbeToDevice(stream, probe, status_out);
  }

  absl::StatusOr<GpuCliqueKey> clique_key =
      AllAssignedDevicesP2PCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    probe[0] = kCliqueKeyFailed;
    return CopyPotrsProbeToDevice(stream, probe, status_out);
  }

  absl::StatusOr<GpuCommunicator*> gpu_comm = collective_cliques->GetComm(
      *clique_key, collective_params->global_device_id);
  if (!gpu_comm.ok() || *gpu_comm == nullptr) {
    probe[0] = kCommunicatorMissing;
    return CopyPotrsProbeToDevice(stream, probe, status_out);
  }

  void* platform_handle = (*gpu_comm)->platform_comm().handle;
  if (platform_handle == nullptr) {
    probe[0] = kNcclHandleMissing;
    return CopyPotrsProbeToDevice(stream, probe, status_out);
  }
  ncclComm_t nccl_comm = reinterpret_cast<ncclComm_t>(platform_handle);

  int nccl_rank = -1;
  int nccl_count = -1;
  ncclResult_t rank_status = ncclCommUserRank(nccl_comm, &nccl_rank);
  ncclResult_t count_status = ncclCommCount(nccl_comm, &nccl_count);
  if (rank_status != ncclSuccess || count_status != ncclSuccess) {
    probe[0] = kNcclRankMismatch;
    return CopyPotrsProbeToDevice(stream, probe, status_out);
  }
  probe[2] = nccl_rank;
  probe[3] = nccl_count;

  if (process_rows <= 0 || process_cols <= 0 ||
      process_rows * process_cols != nccl_count || n <= 0 || nrhs <= 0 ||
      tile_size <= 0) {
    probe[0] = kGridShapeMismatch;
    return CopyPotrsProbeToDevice(stream, probe, status_out);
  }

  CusolverMpApi api = LoadCusolverMpApi(&probe);
  if (probe[0] != kProbeOk || !HasRequiredSymbols(api)) {
    if (api.library != nullptr) {
      dlclose(api.library);
    }
    return CopyPotrsProbeToDevice(stream, probe, status_out);
  }
  if (validate_solution && api.gather_d2h == nullptr) {
    probe[0] = kGatherSymbolMissing;
    dlclose(api.library);
    return CopyPotrsProbeToDevice(stream, probe, status_out);
  }
  if (validate_solution ? !HasDistributedPotrsSymbols(api)
                        : !HasDistributedPotrsSolveSymbols(api)) {
    probe[0] = kSolverSymbolMissing;
    dlclose(api.library);
    return CopyPotrsProbeToDevice(stream, probe, status_out);
  }

  CusolverMpOpaqueHandle handle = nullptr;
  cusolverStatus_t cusolver_status =
      api.create(&handle, cuda_device, cuda_stream);
  if (cusolver_status != CUSOLVER_STATUS_SUCCESS || handle == nullptr) {
    probe[0] = kCreateHandleFailed;
    probe[11] = static_cast<int32_t>(cusolver_status);
    dlclose(api.library);
    return CopyPotrsProbeToDevice(stream, probe, status_out);
  }
  probe[8] = 1;

  int version = -1;
  cusolver_status = api.get_version(handle, &version);
  if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
    probe[0] = kGetVersionFailed;
    probe[11] = static_cast<int32_t>(cusolver_status);
    api.destroy(handle);
    dlclose(api.library);
    return CopyPotrsProbeToDevice(stream, probe, status_out);
  }
  probe[6] = version;

  CusolverMpOpaqueGrid grid = nullptr;
  cusolver_status = api.create_grid(
      handle, &grid, nccl_comm, static_cast<int32_t>(process_rows),
      static_cast<int32_t>(process_cols), kCusolverMpGridMappingRowMajor);
  if (cusolver_status != CUSOLVER_STATUS_SUCCESS || grid == nullptr) {
    probe[0] = kCreateGridFailed;
    probe[11] = static_cast<int32_t>(cusolver_status);
    api.destroy(handle);
    dlclose(api.library);
    return CopyPotrsProbeToDevice(stream, probe, status_out);
  }
  probe[9] = 1;

  const int32_t process_row = nccl_rank / static_cast<int32_t>(process_cols);
  const int32_t process_col = nccl_rank % static_cast<int32_t>(process_cols);

  absl::Status potrs_status;
  switch (a.element_type()) {
    case F32:
      probe[34] = 1;
      potrs_status = RunCusolverMpDistributedPotrsProbe<float>(
          api, handle, grid, cuda_stream, n, nrhs, tile_size,
          a.dimensions()[0], a.dimensions()[1], b.dimensions()[1],
          process_row, process_col, nccl_rank, a, b, a_out, b_out, &probe,
          validate_solution);
      break;
    case F64:
      probe[34] = 2;
      potrs_status = RunCusolverMpDistributedPotrsProbe<double>(
          api, handle, grid, cuda_stream, n, nrhs, tile_size,
          a.dimensions()[0], a.dimensions()[1], b.dimensions()[1],
          process_row, process_col, nccl_rank, a, b, a_out, b_out, &probe,
          validate_solution);
      break;
    case C64:
      probe[34] = 3;
      potrs_status = RunCusolverMpDistributedPotrsProbe<cuFloatComplex>(
          api, handle, grid, cuda_stream, n, nrhs, tile_size,
          a.dimensions()[0], a.dimensions()[1], b.dimensions()[1],
          process_row, process_col, nccl_rank, a, b, a_out, b_out, &probe,
          validate_solution);
      break;
    case C128:
      probe[34] = 4;
      potrs_status = RunCusolverMpDistributedPotrsProbe<cuDoubleComplex>(
          api, handle, grid, cuda_stream, n, nrhs, tile_size,
          a.dimensions()[0], a.dimensions()[1], b.dimensions()[1],
          process_row, process_col, nccl_rank, a, b, a_out, b_out, &probe,
          validate_solution);
      break;
    default:
      probe[0] = kUnsupportedDtype;
      break;
  }
  if (!potrs_status.ok()) {
    api.destroy_grid(grid);
    api.destroy(handle);
    dlclose(api.library);
    return potrs_status;
  }

  if (probe[0] == kProbeOk) {
    cusolver_status = api.destroy_grid(grid);
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
      probe[0] = kDestroyGridFailed;
      probe[11] = static_cast<int32_t>(cusolver_status);
    }
  } else {
    api.destroy_grid(grid);
  }

  if (probe[0] == kProbeOk) {
    cusolver_status = api.destroy(handle);
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
      probe[0] = kDestroyHandleFailed;
      probe[11] = static_cast<int32_t>(cusolver_status);
    }
  } else {
    api.destroy(handle);
  }

  dlclose(api.library);
  return CopyPotrsProbeToDevice(stream, probe, status_out);
}

absl::Status XlaCusolverMpDistributedPotrsProbeDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    int64_t process_rows, int64_t process_cols, int64_t n, int64_t nrhs,
    int64_t tile_size, ffi::AnyBuffer a, ffi::AnyBuffer b,
    ffi::Result<ffi::AnyBuffer> a_out, ffi::Result<ffi::AnyBuffer> b_out,
    ffi::Result<ffi::BufferR1<S32>> status_out,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  return CusolverMpDistributedPotrsDispatchImpl(
      stream, comm_stream, cuda_stream, process_rows, process_cols, n, nrhs,
      tile_size, a, b, a_out, b_out, status_out, collective_params,
      collective_cliques, /*validate_solution=*/true);
}

absl::Status XlaCusolverMpPotrsPrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests) {
  return XlaCusolverMpDistributedPotrsProbePrepare(collective_params,
                                                  clique_requests);
}

absl::Status XlaCusolverMpPotrsDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    int64_t process_rows, int64_t process_cols, int64_t n, int64_t nrhs,
    int64_t tile_size, ffi::AnyBuffer a, ffi::AnyBuffer b,
    ffi::Result<ffi::AnyBuffer> a_out, ffi::Result<ffi::AnyBuffer> b_out,
    ffi::Result<ffi::BufferR1<S32>> status_out,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  return CusolverMpDistributedPotrsDispatchImpl(
      stream, comm_stream, cuda_stream, process_rows, process_cols, n, nrhs,
      tile_size, a, b, a_out, b_out, status_out, collective_params,
      collective_cliques, /*validate_solution=*/false);
}

}  // namespace xla::gpu
