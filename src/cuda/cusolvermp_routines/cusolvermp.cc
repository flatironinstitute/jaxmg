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
// cuSOLVERMp dynamic boundary, diagnostics, and shared solver helpers.
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
//   3. Create a cuSOLVERMp handle and a row- or column-major cuSOLVERMp
//      process grid matching the validated JAX mesh order.
//   4. Create column-major local matrix descriptors over JAXMg device buffers.
//   5. Run diagnostic probes that isolate initialization, scatter layout, and
//      host-generated potrs inputs.
//   6. Expose shared cuSOLVERMp solve helpers used by the production
//      `cusolvermp_potrs` and `cusolvermp_syevd` handlers. Those production
//      handlers live in cusolvermp_potrs.cc and cusolvermp_syevd.cc so each
//      user-facing solver has a compact native entry-point file.
//
// The probe entry points intentionally return device status vectors instead of
// failing hard.  That makes them useful in CI and on systems where the NVIDIA
// HPC SDK is not installed.  The production path uses the same status vector
// convention so Python can distinguish a missing cuSOLVERMp runtime from a
// numerical cuSOLVERMp failure.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <limits>
#include <utility>
#include <vector>

#include <cublas_v2.h>

#include "../include/xla_comm_backend.h"
#include "third_party/nccl/nccl.h"

namespace xla::gpu {
namespace {

// ---------------------------------------------------------------------------
// Dynamic cuSOLVERMp ABI boundary
// ---------------------------------------------------------------------------
//
// cuSOLVERMp is currently loaded dynamically so that the rest of the XLA
// communicator backend can still build on machines that have CUDA/JAX but not
// the NVIDIA HPC SDK.  These typedefs mirror only the cuSOLVERMp calls that
// JAXMg needs.  The rest of the file talks through CusolverMpApi instead of
// including cusolverMp.h directly.

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
using CusolverMpSyevdBufferSizeFn =
    cusolverStatus_t (*)(CusolverMpOpaqueHandle, char*, cublasFillMode_t,
                         int64_t, void*, int64_t, int64_t,
                         CusolverMpOpaqueMatrixDesc, void*, void*, int64_t,
                         int64_t, CusolverMpOpaqueMatrixDesc, cudaDataType,
                         size_t*, size_t*);
using CusolverMpSyevdFn =
    cusolverStatus_t (*)(CusolverMpOpaqueHandle, char*, cublasFillMode_t,
                         int64_t, void*, int64_t, int64_t,
                         CusolverMpOpaqueMatrixDesc, void*, void*, int64_t,
                         int64_t, CusolverMpOpaqueMatrixDesc, cudaDataType,
                         void*, size_t, void*, size_t, int*);

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
  CusolverMpSyevdBufferSizeFn syevd_buffer_size = nullptr;
  CusolverMpSyevdFn syevd = nullptr;
};

// ---------------------------------------------------------------------------
// Status vectors and process-grid validation
// ---------------------------------------------------------------------------
//
// The diagnostic and production FFI handlers return compact int32 status
// vectors.  They are not a public API; they are a debugging contract between
// the native backend and the private Python FFI adapter.  Production callers
// still get useful failure information without crashing the whole JAX program
// when a cluster node is missing cuSOLVERMp or a cuSOLVERMp call returns a
// status code.

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
  kSyevdWorkspaceFailed = 33,
  kSyevdFailed = 34,
  kSyevdInfoNonzero = 35,
};

// Device status vector schemas returned by the diagnostic and production FFI
// calls. Python allocates these output buffers with matching constants in
// `_cusolvermp_ffi.py`; update both sides together when adding fields.
//
// Shared prefix:
//   [0] status code from ProbeStatus
//   [1] selected CUDA device
//   [2] NCCL rank
//   [3] NCCL rank count
//   [4] process rows
//   [5] process columns
//   [6] cuSOLVERMp version
//   [7] cuSOLVERMp library present
//
// The remaining slots are probe-specific and store cuSOLVERMp status codes,
// local matrix sizes, workspace sizes in KiB, info values, and validation flags.
constexpr int kProbeSize = 16;
constexpr int kScatterProbeSize = 24;
constexpr int kPotrsProbeSize = 40;
constexpr int kSyevdProbeSize = 36;
constexpr int kResidualScale = 1000000;

// The dynamic probe cannot include cusolverMp.h because ordinary CUDA/cuSolver
// installs do not ship it. Mirror the enum values from the NVIDIA HPC SDK 26.3
// cuSOLVERMp 0.7.2 header:
//   CUSOLVERMP_GRID_MAPPING_ROW_MAJOR = 1
//   CUSOLVERMP_GRID_MAPPING_COL_MAJOR = 0
//
// JAXMg accepts the two dense grid mappings that cuSOLVERMp itself can
// describe.  The rank_map FFI attribute is indexed by row-major process-grid
// slot and stores the communicator rank at that coordinate.
constexpr int kCusolverMpGridMappingRowMajor = 1;
constexpr int kCusolverMpGridMappingColMajor = 0;

// cuSOLVERMp supports only row-major and column-major mappings from
// communicator ranks onto the process grid.  JAXMg deliberately rejects exotic
// mesh permutations here rather than carrying a separate remapping layer
// through the solver path.
absl::Status ValidateCusolverMpGridMapping(const char* caller,
                                           int64_t grid_mapping) {
  if (grid_mapping != kCusolverMpGridMappingColMajor &&
      grid_mapping != kCusolverMpGridMappingRowMajor) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s grid_mapping must be 0 (column-major) or 1 (row-major), got %d",
        caller, grid_mapping));
  }
  return absl::OkStatus();
}

absl::Status ValidateStandardRankMapForGridMapping(
    const char* caller, absl::Span<const int64_t> rank_map,
    int64_t process_rows, int64_t process_cols, int64_t grid_mapping) {
  JAXMG_RETURN_IF_ERROR(ValidateCusolverMpGridMapping(caller, grid_mapping));
  const int64_t num_ranks = process_rows * process_cols;
  if (rank_map.size() != static_cast<size_t>(num_ranks)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s expected rank_map length %d, got %d", caller, num_ranks,
        rank_map.size()));
  }
  std::vector<bool> seen(num_ranks, false);
  for (int64_t process_rank = 0; process_rank < num_ranks; ++process_rank) {
    const int64_t communicator_rank = rank_map[process_rank];
    if (communicator_rank < 0 || communicator_rank >= num_ranks) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "%s rank_map[%d]=%d is outside [0, %d)", caller, process_rank,
          communicator_rank, num_ranks));
    }
    if (seen[communicator_rank]) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "%s rank_map contains duplicate communicator rank %d", caller,
          communicator_rank));
    }
    seen[communicator_rank] = true;
    const int64_t process_row = process_rank / process_cols;
    const int64_t process_col = process_rank % process_cols;
    const int64_t expected =
        grid_mapping == kCusolverMpGridMappingRowMajor
            ? process_rank
            : process_col * process_rows + process_row;
    if (communicator_rank != expected) {
      return absl::UnimplementedError(absl::StrFormat(
          "%s supports only rank maps matching the requested cuSOLVERMp "
          "grid_mapping; rank_map[%d]=%d, expected %d",
          caller, process_rank, communicator_rank, expected));
    }
  }
  return absl::OkStatus();
}

std::pair<int32_t, int32_t> ProcessCoordFromRank(int nccl_rank,
                                                 int64_t process_rows,
                                                 int64_t process_cols,
                                                 int64_t grid_mapping) {
  if (grid_mapping == kCusolverMpGridMappingRowMajor) {
    return {
        nccl_rank / static_cast<int32_t>(process_cols),
        nccl_rank % static_cast<int32_t>(process_cols),
    };
  }
  return {
      nccl_rank % static_cast<int32_t>(process_rows),
      nccl_rank / static_cast<int32_t>(process_rows),
  };
}

// ---------------------------------------------------------------------------
// Dynamic symbol loading and low-level size helpers
// ---------------------------------------------------------------------------

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
  api.syevd_buffer_size = LoadRequiredSymbol<CusolverMpSyevdBufferSizeFn>(
      api.library, "cusolverMpSyevd_bufferSize");
  api.syevd =
      LoadRequiredSymbol<CusolverMpSyevdFn>(api.library, "cusolverMpSyevd");

  if (api.create == nullptr || api.destroy == nullptr ||
      api.get_version == nullptr || api.create_grid == nullptr ||
      api.destroy_grid == nullptr || api.create_matrix_desc == nullptr ||
      api.destroy_matrix_desc == nullptr) {
    (*probe)[0] = kSymbolMissing;
  }
  return api;
}

// LocalNumroc mirrors ScaLAPACK/BLACS NUMROC for the simple 2D block-cyclic
// ownership pattern used by cuSOLVERMp: count how many global rows or columns
// this process coordinate owns for a given tile size.
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

// ---------------------------------------------------------------------------
// Status-vector copy helpers and cuSOLVERMp capability checks
// ---------------------------------------------------------------------------
//
// Probe status arrays are built on the host and copied to a JAX result buffer
// at the end of each FFI call.  The Has* functions make it explicit which
// symbol groups are required for each diagnostic or production path.

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

bool HasSyevdSymbols(const CusolverMpApi& api) {
  return HasRequiredSymbols(api) && api.syevd_buffer_size != nullptr &&
         api.syevd != nullptr;
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

absl::Status CopySyevdProbeToDevice(
    se::Stream* stream, const std::array<int32_t, kSyevdProbeSize>& probe,
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

// ---------------------------------------------------------------------------
// Host-side diagnostic matrix generators
// ---------------------------------------------------------------------------
//
// These helpers are used only by the probe handlers.  They build tiny,
// deterministic host matrices that cuSOLVERMp scatters internally, allowing us
// to separate "cuSOLVERMp itself works on this machine" from "JAXMg's native
// redistribution produced the expected local buffers".

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
std::vector<DataType> MakeSyevdHostA(int64_t n) {
  std::vector<DataType> host(static_cast<size_t>(n * n));
  for (int64_t col = 0; col < n; ++col) {
    for (int64_t row = 0; row < n; ++row) {
      double value = 0.0;
      if (row == col) {
        value = static_cast<double>(n + row + 1);
      } else {
        value = 0.01 / static_cast<double>(1 + std::abs(row - col));
      }
      host[static_cast<size_t>(row + col * n)] = RealScalar<DataType>(value);
    }
  }
  return host;
}

template <typename DataType>
std::vector<DataType> MakeSyevdHostQ(int64_t n) {
  std::vector<DataType> host(static_cast<size_t>(n * n));
  std::fill(host.begin(), host.end(), RealScalar<DataType>(-1.0));
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

// ---------------------------------------------------------------------------
// Runtime utility helpers
// ---------------------------------------------------------------------------
//
// The production fused handlers usually alias input and output buffers through
// JAX FFI donation.  The copy helper is used by diagnostic paths and shared
// solver helpers where an output buffer may or may not already alias the input.
// DeviceForCudaPointer is important for multi-device JAX callbacks: the host
// thread's current CUDA device is not always the owner of the donated buffer.

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

bool CusolverMpDebugEnabled() {
  const char* value = std::getenv("JAXMG_CUSOLVERMP_DEBUG");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

void CusolverMpDebug(int rank, const char* format, ...) {
  if (!CusolverMpDebugEnabled()) {
    return;
  }
  std::fprintf(stderr, "JAXMG_CUSOLVERMP_DEBUG rank=%d ", rank);
  va_list args;
  va_start(args, format);
  std::vfprintf(stderr, format, args);
  va_end(args);
  std::fprintf(stderr, "\n");
  std::fflush(stderr);
}

absl::StatusOr<int> DeviceForCudaPointer(const void* ptr) {
  cudaPointerAttributes attrs;
  cudaError_t status = cudaPointerGetAttributes(&attrs, ptr);
  if (status != cudaSuccess) {
    return absl::InternalError(absl::StrFormat(
        "cudaPointerGetAttributes failed while resolving the cuSOLVERMp "
        "buffer device: %s",
        cudaGetErrorString(status)));
  }
  return attrs.device;
}

// ---------------------------------------------------------------------------
// Diagnostic probe runners
// ---------------------------------------------------------------------------
//
// The next group of templated helpers runs isolated cuSOLVERMp operations on
// generated inputs.  They do not exercise JAXMg's native 2D redistribution; the
// goal is to prove the borrowed XLA/NCCL communicator can be consumed by
// cuSOLVERMp and that a given CSD3/NVIDIA runtime supports the requested
// solver routine and dtype.

// Scatter-layout probe: have cuSOLVERMp scatter a host column-major matrix
// into local device memory.  This verifies descriptor leading dimensions and
// process-grid mapping independently of JAXMg's own redistribution kernels.
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

// POTRS sample probe: generate a diagonal SPD matrix and RHS on rank 0, use
// cuSOLVERMp scatter, run potrf/potrs, gather the result, and check the residual
// on rank 0.  This is a cuSOLVERMp-only smoke test.
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

// Shared POTRS runner for buffers that have already been redistributed by
// JAXMg.  Production potrs calls this with validate_solution=false so it does
// not gather a full solution back to host.  Diagnostic distributed-input probes
// can enable validation to compare against the known generated RHS.
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

// SYEVD sample probe: like the POTRS sample probe, but for eigensolves.  It
// uses cuSOLVERMp's scatter path for both A and Q so failures here usually mean
// a cuSOLVERMp/runtime issue rather than a JAXMg redistribution issue.
template <typename DataType>
absl::Status RunCusolverMpSyevdSampleProbe(
    const CusolverMpApi& api, CusolverMpOpaqueHandle handle,
    CusolverMpOpaqueGrid grid_a, CusolverMpOpaqueGrid grid_q,
    cudaStream_t cuda_stream, int64_t n, int64_t tile_size,
    int32_t process_row, int32_t process_col,
    std::array<int32_t, kSyevdProbeSize>* probe) {
  using EigenvalueType = typename SolverTraits<DataType>::EigenvalueType;

  const int debug_rank = (*probe)[2];
  const int32_t process_rows = (*probe)[4];
  const int32_t process_cols = (*probe)[5];
  const int64_t local_rows = LocalNumroc(n, tile_size, process_row,
                                         static_cast<int32_t>(process_rows));
  const int64_t local_cols = LocalNumroc(n, tile_size, process_col,
                                         static_cast<int32_t>(process_cols));
  const int64_t lld = std::max<int64_t>(1, local_rows);
  const int64_t allocated_cols = std::max<int64_t>(1, local_cols);
  const size_t matrix_bytes =
      static_cast<size_t>(lld * allocated_cols) * sizeof(DataType);
  const size_t eigenvalue_bytes =
      static_cast<size_t>(std::max<int64_t>(1, n)) * sizeof(EigenvalueType);
  (*probe)[17] = static_cast<int32_t>(local_rows);
  (*probe)[18] = static_cast<int32_t>(local_cols);
  (*probe)[19] = SizeToKiBForProbe(eigenvalue_bytes);
  (*probe)[28] = 1;  // Matrix input is produced by cuSOLVERMp scatter.

  CusolverMpDebug(
      debug_rank,
      "syevd sample enter n=%lld tile=%lld process_coord=(%d,%d) "
      "local=%lldx%lld lld=%lld dtype=%d",
      static_cast<long long>(n), static_cast<long long>(tile_size),
      process_row, process_col, static_cast<long long>(local_rows),
      static_cast<long long>(local_cols), static_cast<long long>(lld),
      static_cast<int>(SolverTraits<DataType>::cuda_data_type));

  CusolverMpOpaqueMatrixDesc desc_a = nullptr;
  CusolverMpOpaqueMatrixDesc desc_q = nullptr;
  cusolverStatus_t status = api.create_matrix_desc(
      &desc_a, grid_a, SolverTraits<DataType>::cuda_data_type, n, n,
      tile_size, tile_size, /*RSRC_A=*/0, /*CSRC_A=*/0, lld);
  CusolverMpDebug(debug_rank, "syevd sample create desc_a status=%d desc=%p",
                  static_cast<int>(status), desc_a);
  if (status != CUSOLVER_STATUS_SUCCESS || desc_a == nullptr) {
    (*probe)[0] = kCreateMatrixDescFailed;
    (*probe)[11] = static_cast<int32_t>(status);
    return absl::OkStatus();
  }
  (*probe)[10] = 1;

  status = api.create_matrix_desc(
      &desc_q, grid_q, SolverTraits<DataType>::cuda_data_type, n, n,
      tile_size, tile_size, /*RSRC_Q=*/0, /*CSRC_Q=*/0, lld);
  CusolverMpDebug(debug_rank, "syevd sample create desc_q status=%d desc=%p",
                  static_cast<int>(status), desc_q);
  if (status != CUSOLVER_STATUS_SUCCESS || desc_q == nullptr) {
    (*probe)[0] = kCreateMatrixDescFailed;
    (*probe)[11] = static_cast<int32_t>(status);
    api.destroy_matrix_desc(desc_a);
    return absl::OkStatus();
  }
  (*probe)[27] = 1;

  void* d_a = nullptr;
  void* d_q = nullptr;
  void* d_eigenvalues = nullptr;
  void* d_work = nullptr;
  void* h_work = nullptr;
  int* d_info = nullptr;
  auto cleanup = [&]() {
    if (d_a != nullptr) cudaFree(d_a);
    if (d_q != nullptr) cudaFree(d_q);
    if (d_eigenvalues != nullptr) cudaFree(d_eigenvalues);
    if (d_work != nullptr) cudaFree(d_work);
    if (d_info != nullptr) cudaFree(d_info);
    if (h_work != nullptr) std::free(h_work);
    api.destroy_matrix_desc(desc_q);
    api.destroy_matrix_desc(desc_a);
  };

  cudaError_t cuda_status = cudaMalloc(&d_a, matrix_bytes);
  if (cuda_status != cudaSuccess) {
    (*probe)[0] = kDeviceAllocFailed;
    cleanup();
    return absl::OkStatus();
  }
  cuda_status = cudaMalloc(&d_q, matrix_bytes);
  if (cuda_status != cudaSuccess) {
    (*probe)[0] = kDeviceAllocFailed;
    cleanup();
    return absl::OkStatus();
  }
  cuda_status = cudaMalloc(&d_eigenvalues, eigenvalue_bytes);
  if (cuda_status != cudaSuccess) {
    (*probe)[0] = kDeviceAllocFailed;
    cleanup();
    return absl::OkStatus();
  }
  JAXMG_RETURN_IF_CUDA_ERROR(
      cudaMemsetAsync(d_a, 0, matrix_bytes, cuda_stream));
  JAXMG_RETURN_IF_CUDA_ERROR(
      cudaMemsetAsync(d_q, 0, matrix_bytes, cuda_stream));
  JAXMG_RETURN_IF_CUDA_ERROR(cudaMemsetAsync(
      d_eigenvalues, 0, eigenvalue_bytes, cuda_stream));

  size_t workspace_device = 0;
  size_t workspace_host = 0;
  // cuSOLVERMp 0.7.x exposes this selector as `compz` and NVIDIA's
  // distributed SYEVD sample uses 'Z' for the eigenvector-producing path.  The
  // newer documentation names the argument `jobz`, but the installed header
  // and runtime follow the Stedc-style convention.
  char compz[] = {'Z', '\0'};
  CusolverMpDebug(debug_rank,
                  "syevd sample buffer_size begin compz=%c a=%p d=%p q=%p",
                  compz[0], d_a, d_eigenvalues, d_q);
  status = api.syevd_buffer_size(
      handle, compz, CUBLAS_FILL_MODE_LOWER, n, d_a, /*IA=*/1, /*JA=*/1,
      desc_a, d_eigenvalues, d_q, /*IQ=*/1, /*JQ=*/1, desc_q,
      SolverTraits<DataType>::cuda_data_type, &workspace_device,
      &workspace_host);
  CusolverMpDebug(debug_rank,
                  "syevd sample buffer_size end status=%d device=%zu host=%zu",
                  static_cast<int>(status), workspace_device, workspace_host);
  if (status != CUSOLVER_STATUS_SUCCESS) {
    (*probe)[0] = kSyevdWorkspaceFailed;
    (*probe)[11] = static_cast<int32_t>(status);
    cleanup();
    return absl::OkStatus();
  }
  (*probe)[21] = SizeToKiBForProbe(workspace_device);
  (*probe)[22] = SizeToKiBForProbe(workspace_host);

  if (workspace_device > 0) {
    cuda_status = cudaMalloc(&d_work, workspace_device);
    if (cuda_status != cudaSuccess) {
      (*probe)[0] = kDeviceAllocFailed;
      cleanup();
      return absl::OkStatus();
    }
  }
  if (workspace_host > 0) {
    h_work = std::malloc(workspace_host);
    if (h_work == nullptr) {
      (*probe)[0] = kHostAllocFailed;
      cleanup();
      return absl::OkStatus();
    }
  }
  cuda_status = cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int));
  if (cuda_status != cudaSuccess) {
    (*probe)[0] = kDeviceAllocFailed;
    cleanup();
    return absl::OkStatus();
  }
  JAXMG_RETURN_IF_CUDA_ERROR(
      cudaMemsetAsync(d_info, 0, sizeof(int), cuda_stream));

  std::vector<DataType> host_a;
  std::vector<DataType> host_q;
  const void* host_a_ptr = nullptr;
  const void* host_q_ptr = nullptr;
  if (debug_rank == 0) {
    host_a = MakeSyevdHostA<DataType>(n);
    host_q = MakeSyevdHostQ<DataType>(n);
    host_a_ptr = host_a.data();
    host_q_ptr = host_q.data();
  }

  CusolverMpDebug(debug_rank, "syevd sample scatter A begin");
  status = api.scatter_h2d(handle, n, n, d_a, /*IA=*/1, /*JA=*/1, desc_a,
                           /*root=*/0, host_a_ptr, /*h_ldsrc=*/n);
  CusolverMpDebug(debug_rank, "syevd sample scatter A end status=%d",
                  static_cast<int>(status));
  if (status != CUSOLVER_STATUS_SUCCESS) {
    (*probe)[0] = kScatterFailed;
    (*probe)[11] = static_cast<int32_t>(status);
    cleanup();
    return absl::OkStatus();
  }
  (*probe)[29] = 1;

  CusolverMpDebug(debug_rank, "syevd sample scatter Q begin");
  status = api.scatter_h2d(handle, n, n, d_q, /*IQ=*/1, /*JQ=*/1, desc_q,
                           /*root=*/0, host_q_ptr, /*h_ldsrc=*/n);
  CusolverMpDebug(debug_rank, "syevd sample scatter Q end status=%d",
                  static_cast<int>(status));
  if (status != CUSOLVER_STATUS_SUCCESS) {
    (*probe)[0] = kScatterFailed;
    (*probe)[11] = static_cast<int32_t>(status);
    cleanup();
    return absl::OkStatus();
  }
  (*probe)[30] = 1;

  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));

  CusolverMpDebug(debug_rank, "syevd sample syevd begin");
  status = api.syevd(handle, compz, CUBLAS_FILL_MODE_LOWER, n, d_a, /*IA=*/1,
                     /*JA=*/1, desc_a, d_eigenvalues, d_q, /*IQ=*/1,
                     /*JQ=*/1, desc_q,
                     SolverTraits<DataType>::cuda_data_type, d_work,
                     workspace_device, h_work, workspace_host, d_info);
  CusolverMpDebug(debug_rank, "syevd sample syevd end status=%d",
                  static_cast<int>(status));
  if (status != CUSOLVER_STATUS_SUCCESS) {
    (*probe)[0] = kSyevdFailed;
    (*probe)[11] = static_cast<int32_t>(status);
    cleanup();
    return absl::OkStatus();
  }
  (*probe)[23] = 1;

  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));
  int h_info = -1;
  JAXMG_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
      &h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost, cuda_stream));
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));
  CusolverMpDebug(debug_rank, "syevd sample info=%d", h_info);
  (*probe)[24] = h_info;
  if (h_info != 0) {
    (*probe)[0] = kSyevdInfoNonzero;
  }

  cleanup();
  return absl::OkStatus();
}

// Shared production SYEVD runner for pre-redistributed JAXMg buffers.  The
// input matrix is copied/aliased into work_out as cuSOLVERMp's d_A because
// SYEVD overwrites A.  vectors_out is the separate d_Z buffer where cuSOLVERMp
// writes eigenvectors.
template <typename DataType>
absl::Status RunCusolverMpSyevd(
    const CusolverMpApi& api, CusolverMpOpaqueHandle handle,
    CusolverMpOpaqueGrid grid, cudaStream_t cuda_stream, int64_t n,
    int64_t tile_size, int64_t local_physical_rows,
    int64_t local_physical_cols, int32_t process_row, int32_t process_col,
    ffi::AnyBuffer a, ffi::Result<ffi::AnyBuffer> eigenvalues_out,
    ffi::Result<ffi::AnyBuffer> work_out,
    ffi::Result<ffi::AnyBuffer> vectors_out,
    std::array<int32_t, kSyevdProbeSize>* probe) {
  const int debug_rank = (*probe)[2];
  const int32_t process_rows = (*probe)[4];
  const int32_t process_cols = (*probe)[5];
  const int64_t local_rows = LocalNumroc(n, tile_size, process_row,
                                         static_cast<int32_t>(process_rows));
  const int64_t local_cols = LocalNumroc(n, tile_size, process_col,
                                         static_cast<int32_t>(process_cols));
  (*probe)[17] = static_cast<int32_t>(local_rows);
  (*probe)[18] = static_cast<int32_t>(local_cols);
  if (local_physical_rows < local_rows || local_physical_cols < local_cols) {
    (*probe)[0] = kOutputShapeMismatch;
    return absl::OkStatus();
  }
  CusolverMpDebug(
      debug_rank,
      "syevd enter n=%lld tile=%lld process_coord=(%d,%d) local=%lldx%lld "
      "physical=%lldx%lld dtype=%d",
      static_cast<long long>(n), static_cast<long long>(tile_size),
      process_row, process_col, static_cast<long long>(local_rows),
      static_cast<long long>(local_cols),
      static_cast<long long>(local_physical_rows),
      static_cast<long long>(local_physical_cols),
      static_cast<int>(SolverTraits<DataType>::cuda_data_type));

  CusolverMpOpaqueMatrixDesc desc_a = nullptr;
  CusolverMpOpaqueMatrixDesc desc_q = nullptr;
  CusolverMpDebug(debug_rank, "create desc_a begin");
  cusolverStatus_t status = api.create_matrix_desc(
      &desc_a, grid, SolverTraits<DataType>::cuda_data_type, n, n, tile_size,
      tile_size, /*RSRC_A=*/0, /*CSRC_A=*/0, local_physical_rows);
  CusolverMpDebug(debug_rank, "create desc_a end status=%d desc=%p",
                  static_cast<int>(status), desc_a);
  if (status != CUSOLVER_STATUS_SUCCESS || desc_a == nullptr) {
    (*probe)[0] = kCreateMatrixDescFailed;
    (*probe)[11] = static_cast<int32_t>(status);
    return absl::OkStatus();
  }
  (*probe)[10] = 1;

  CusolverMpDebug(debug_rank, "create desc_q begin");
  status = api.create_matrix_desc(
      &desc_q, grid, SolverTraits<DataType>::cuda_data_type, n, n, tile_size,
      tile_size, /*RSRC_Q=*/0, /*CSRC_Q=*/0, local_physical_rows);
  CusolverMpDebug(debug_rank, "create desc_q end status=%d desc=%p",
                  static_cast<int>(status), desc_q);
  if (status != CUSOLVER_STATUS_SUCCESS || desc_q == nullptr) {
    (*probe)[0] = kCreateMatrixDescFailed;
    (*probe)[11] = static_cast<int32_t>(status);
    api.destroy_matrix_desc(desc_a);
    return absl::OkStatus();
  }
  (*probe)[27] = 1;

  void* d_work = nullptr;
  void* h_work = nullptr;
  int* d_info = nullptr;
  auto cleanup = [&]() {
    if (d_work != nullptr) cudaFree(d_work);
    if (d_info != nullptr) cudaFree(d_info);
    if (h_work != nullptr) std::free(h_work);
    api.destroy_matrix_desc(desc_q);
    api.destroy_matrix_desc(desc_a);
  };

  // cuSOLVERMp SYEVD overwrites d_A and writes eigenvectors to d_Z when
  // requested. Keep those buffers distinct: the donated JAX input aliases the
  // work output used as d_A, while the public eigenvector result is d_Z.
  CusolverMpDebug(debug_rank, "copy input to work begin a=%p work=%p bytes=%lld",
                  a.untyped_data(), work_out->untyped_data(),
                  static_cast<long long>(a.size_bytes()));
  JAXMG_RETURN_IF_ERROR(CopyAnyBufferToOutputIfNeeded(cuda_stream, a, work_out));
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));
  CusolverMpDebug(debug_rank, "copy input to work end");

  size_t workspace_device = 0;
  size_t workspace_host = 0;
  // cuSOLVERMp 0.7.x exposes this selector as `compz` and NVIDIA's
  // distributed SYEVD sample uses 'Z' for the eigenvector-producing path. Keep
  // this aligned with the installed header/runtime rather than the newer
  // `jobz='V'` wording in some versions of the public documentation.
  char compz[] = {'Z', '\0'};
  void* z_data = vectors_out->untyped_data();
  CusolverMpDebug(
      debug_rank,
      "syevd_buffer_size begin compz=%c a=%p evals=%p z=%p desc_a=%p desc_q=%p",
      compz[0], work_out->untyped_data(), eigenvalues_out->untyped_data(),
      z_data, desc_a, desc_q);
  status = api.syevd_buffer_size(
      handle, compz, CUBLAS_FILL_MODE_LOWER, n, work_out->untyped_data(),
      /*IA=*/1, /*JA=*/1, desc_a, eigenvalues_out->untyped_data(),
      z_data, /*IQ=*/1, /*JQ=*/1, desc_q,
      SolverTraits<DataType>::cuda_data_type, &workspace_device,
      &workspace_host);
  CusolverMpDebug(debug_rank,
                  "syevd_buffer_size end status=%d workspace_device=%zu "
                  "workspace_host=%zu",
                  static_cast<int>(status), workspace_device, workspace_host);
  if (status != CUSOLVER_STATUS_SUCCESS) {
    (*probe)[0] = kSyevdWorkspaceFailed;
    (*probe)[11] = static_cast<int32_t>(status);
    cleanup();
    return absl::OkStatus();
  }
  (*probe)[21] = SizeToKiBForProbe(workspace_device);
  (*probe)[22] = SizeToKiBForProbe(workspace_host);

  cudaError_t cuda_status = cudaSuccess;
  if (workspace_device > 0) {
    CusolverMpDebug(debug_rank, "cudaMalloc d_work begin bytes=%zu",
                    workspace_device);
    cuda_status = cudaMalloc(&d_work, workspace_device);
    CusolverMpDebug(debug_rank, "cudaMalloc d_work end status=%d ptr=%p",
                    static_cast<int>(cuda_status), d_work);
    if (cuda_status != cudaSuccess) {
      (*probe)[0] = kDeviceAllocFailed;
      cleanup();
      return absl::OkStatus();
    }
  }
  if (workspace_host > 0) {
    CusolverMpDebug(debug_rank, "malloc h_work begin bytes=%zu", workspace_host);
    h_work = std::malloc(workspace_host);
    CusolverMpDebug(debug_rank, "malloc h_work end ptr=%p", h_work);
    if (h_work == nullptr) {
      (*probe)[0] = kHostAllocFailed;
      cleanup();
      return absl::OkStatus();
    }
  }
  CusolverMpDebug(debug_rank, "cudaMalloc d_info begin");
  cuda_status = cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int));
  CusolverMpDebug(debug_rank, "cudaMalloc d_info end status=%d ptr=%p",
                  static_cast<int>(cuda_status), d_info);
  if (cuda_status != cudaSuccess) {
    (*probe)[0] = kDeviceAllocFailed;
    cleanup();
    return absl::OkStatus();
  }
  JAXMG_RETURN_IF_CUDA_ERROR(
      cudaMemsetAsync(d_info, 0, sizeof(int), cuda_stream));
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));

  CusolverMpDebug(debug_rank, "syevd begin");
  status = api.syevd(handle, compz, CUBLAS_FILL_MODE_LOWER, n,
                     work_out->untyped_data(), /*IA=*/1, /*JA=*/1, desc_a,
                     eigenvalues_out->untyped_data(), z_data, /*IQ=*/1,
                     /*JQ=*/1, desc_q,
                     SolverTraits<DataType>::cuda_data_type, d_work,
                     workspace_device, h_work, workspace_host, d_info);
  CusolverMpDebug(debug_rank, "syevd end status=%d", static_cast<int>(status));
  if (status != CUSOLVER_STATUS_SUCCESS) {
    (*probe)[0] = kSyevdFailed;
    (*probe)[11] = static_cast<int32_t>(status);
    cleanup();
    return absl::OkStatus();
  }
  (*probe)[23] = 1;
  CusolverMpDebug(debug_rank, "stream sync after syevd begin");
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));
  CusolverMpDebug(debug_rank, "stream sync after syevd end");

  int h_info = -1;
  JAXMG_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
      &h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost, cuda_stream));
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));
  CusolverMpDebug(debug_rank, "syevd info=%d", h_info);
  (*probe)[24] = h_info;
  if (h_info != 0) {
    (*probe)[0] = kSyevdInfoNonzero;
  }

  cleanup();
  return absl::OkStatus();
}

}  // namespace

// ---------------------------------------------------------------------------
// FFI prepare hooks
// ---------------------------------------------------------------------------
//
// XLA calls prepare functions while building the compiled executable.  These
// hooks request the all-assigned P2P communicator clique that dispatch will
// later look up through CollectiveCliques.  The same prepare path is used for
// probes and production solvers because all of them need the XLA-owned NCCL
// communicator spanning the cuSOLVERMp process grid.

absl::Status XlaCusolverMpInitProbePrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests) {
  return RequestAllAssignedP2PCommunicator(
      collective_params, clique_requests, "cusolvermp_init_probe");
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

// ---------------------------------------------------------------------------
// Diagnostic FFI dispatchers
// ---------------------------------------------------------------------------
//
// The init/scatter/potrs/syevd probe dispatchers below are intentionally
// verbose.  Each one builds a status vector, checks the XLA communicator,
// dynamically loads cuSOLVERMp, constructs the minimal handle/grid/descriptor
// state for the probe, runs the requested cuSOLVERMp operation, and returns the
// status vector to Python.  They are kept in this file because they share the
// same dynamic ABI boundary and cuSOLVERMp lifecycle helpers as production.

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
      -1,  // cuSOLVERMp grid mapping: 0 column-major, 1 row-major.
  };

  // cuSOLVERMp binds the handle to one CUDA device.  In a JAX multi-device FFI
  // call the host thread's current device is not a reliable proxy for the
  // donated buffer's owner, especially when the JAX mesh uses a column-major
  // process-grid order.  Match the SYEVD path and anchor the handle to A.
  absl::StatusOr<int> buffer_device = DeviceForCudaPointer(a.untyped_data());
  if (!buffer_device.ok()) {
    probe[0] = kCudaDeviceFailed;
    return CopyPotrsProbeToDevice(stream, probe, status_out);
  }
  int cuda_device = *buffer_device;
  cudaError_t cuda_status = cudaSetDevice(cuda_device);
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

// ---------------------------------------------------------------------------
// Shared production solver dispatchers
// ---------------------------------------------------------------------------
//
// The compact production entry-point files call these helpers after
// memory_redist has converted local JAX row-major buffers to column-major,
// compacted edge padding, and redistributed the buffers into cuSOLVERMp's 2D
// block-cyclic layout.  These helpers therefore start at the cuSOLVERMp layer:
// validate the FFI buffer contract, borrow the raw NCCL handle from XLA, build
// the cuSOLVERMp grid/descriptors, run the solver, and return a diagnostic
// status vector alongside the numerical output.

absl::Status CusolverMpDistributedPotrsDispatchImpl(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    int64_t process_rows, int64_t process_cols, int64_t n, int64_t nrhs,
    int64_t tile_size, int64_t grid_mapping,
    absl::Span<const int64_t> rank_map, ffi::AnyBuffer a, ffi::AnyBuffer b,
    ffi::Result<ffi::AnyBuffer> a_out, ffi::Result<ffi::AnyBuffer> b_out,
    ffi::Result<ffi::BufferR1<S32>> status_out,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques, bool validate_solution) {
  // Stage 1: validate the static FFI contract.  By this point Python/JAX has
  // already padded the user arrays and memory_redist has produced local
  // column-major 2D block-cyclic buffers.  The checks here are deliberately
  // about local capacities and dtype consistency, not global mathematical
  // correctness.
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

  // Stage 2: bind CUDA work to the device that owns this rank's local A shard.
  // This avoids relying on ambient host-thread CUDA state, which is not stable
  // across JAX's multi-device FFI callbacks.
  absl::StatusOr<int> buffer_device = DeviceForCudaPointer(a.untyped_data());
  if (!buffer_device.ok()) {
    probe[0] = kCudaDeviceFailed;
    return CopyPotrsProbeToDevice(stream, probe, status_out);
  }
  int cuda_device = *buffer_device;
  cudaError_t cuda_status = cudaSetDevice(cuda_device);
  if (cuda_status != cudaSuccess) {
    probe[0] = kCudaDeviceFailed;
    return CopyPotrsProbeToDevice(stream, probe, status_out);
  }
  probe[1] = cuda_device;

  // Stage 3: retrieve the communicator that XLA already created for the
  // compiled program.  cuSOLVERMp receives the raw NCCL handle from that
  // communicator; JAXMg does not create a separate NCCL communicator here.
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

  // Stage 4: validate that the requested cuSOLVERMp process grid matches the
  // communicator size and that any explicit Python rank map is one of the two
  // dense mappings cuSOLVERMp can describe directly.
  if (process_rows <= 0 || process_cols <= 0 ||
      process_rows * process_cols != nccl_count || n <= 0 || nrhs <= 0 ||
      tile_size <= 0) {
    probe[0] = kGridShapeMismatch;
    return CopyPotrsProbeToDevice(stream, probe, status_out);
  }
  absl::Status grid_mapping_status =
      ValidateCusolverMpGridMapping("cusolvermp_potrs", grid_mapping);
  if (!grid_mapping_status.ok()) {
    return grid_mapping_status;
  }
  probe[39] = static_cast<int32_t>(grid_mapping);
  if (!rank_map.empty()) {
    absl::Status rank_map_status = ValidateStandardRankMapForGridMapping(
        "cusolvermp_potrs", rank_map, process_rows, process_cols,
        grid_mapping);
    if (!rank_map_status.ok()) {
      return rank_map_status;
    }
  }

  // Stage 5: load cuSOLVERMp and create the handle/grid for this rank.  The
  // grid mapping value is passed through to cuSOLVERMp so row-major and
  // column-major JAX mesh orders remain supported without an extra remap layer.
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
      static_cast<int32_t>(process_cols), static_cast<int>(grid_mapping));
  if (cusolver_status != CUSOLVER_STATUS_SUCCESS || grid == nullptr) {
    probe[0] = kCreateGridFailed;
    probe[11] = static_cast<int32_t>(cusolver_status);
    api.destroy(handle);
    dlclose(api.library);
    return CopyPotrsProbeToDevice(stream, probe, status_out);
  }
  probe[9] = 1;

  const auto [process_row, process_col] = ProcessCoordFromRank(
      nccl_rank, process_rows, process_cols, grid_mapping);

  // Stage 6: dispatch on the XLA primitive dtype and run the shared POTRS
  // helper.  Production calls skip the optional host gather/residual check;
  // diagnostic calls can enable it through validate_solution.
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

  // Stage 7: tear down cuSOLVERMp resources in reverse construction order and
  // return the per-rank status vector to Python.
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

absl::Status CusolverMpSyevdDispatchImpl(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    int64_t process_rows, int64_t process_cols, int64_t n,
    int64_t tile_size, int64_t grid_mapping,
    absl::Span<const int64_t> rank_map, ffi::AnyBuffer a,
    ffi::Result<ffi::AnyBuffer> eigenvalues_out,
    ffi::Result<ffi::AnyBuffer> work_out,
    ffi::Result<ffi::AnyBuffer> vectors_out,
    ffi::Result<ffi::BufferR1<S32>> status_out,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  // Stage 1: validate the local FFI buffers.  SYEVD has three matrix buffers:
  // the redistributed input A, work_out used as cuSOLVERMp's overwritten d_A,
  // and vectors_out used as cuSOLVERMp's d_Z eigenvector output.
  if (stream == nullptr || cuda_stream == nullptr) {
    return absl::InvalidArgumentError(
        "cusolvermp_syevd requires XLA and CUDA streams");
  }
  if (a.dimensions().size() != 2 || work_out->dimensions().size() != 2 ||
      vectors_out->dimensions().size() != 2 ||
      eigenvalues_out->dimensions().size() != 1) {
    return absl::InvalidArgumentError(
        "cusolvermp_syevd expects rank-2 matrix buffers and rank-1 "
        "eigenvalue output");
  }
  if (a.element_type() != work_out->element_type() ||
      a.element_type() != vectors_out->element_type()) {
    return absl::InvalidArgumentError(
        "cusolvermp_syevd requires matching matrix/work/vector dtypes");
  }
  const PrimitiveType expected_eigen_type =
      (a.element_type() == F32 || a.element_type() == C64) ? F32 : F64;
  if (eigenvalues_out->element_type() != expected_eigen_type) {
    return absl::InvalidArgumentError(
        "cusolvermp_syevd received an eigenvalue output with the wrong dtype");
  }
  if (a.dimensions()[0] != work_out->dimensions()[0] ||
      a.dimensions()[1] != work_out->dimensions()[1] ||
      a.dimensions()[0] != vectors_out->dimensions()[0] ||
      a.dimensions()[1] != vectors_out->dimensions()[1]) {
    return absl::InvalidArgumentError(
        "cusolvermp_syevd input/output matrix shapes must match");
  }
  if (eigenvalues_out->dimensions()[0] != n) {
    return absl::InvalidArgumentError(
        "cusolvermp_syevd eigenvalue output length must match n");
  }
  if (status_out->dimensions().size() != 1 ||
      status_out->dimensions()[0] != kSyevdProbeSize) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "cusolvermp_syevd expects status shape (%d,)", kSyevdProbeSize));
  }

  std::array<int32_t, kSyevdProbeSize> probe = {
      kProbeOk,
      -1,  // CUDA device selected for this FFI invocation.
      -1,  // NCCL rank reported by the borrowed communicator.
      -1,  // NCCL communicator size.
      static_cast<int32_t>(process_rows),
      static_cast<int32_t>(process_cols),
      -1,  // cuSOLVERMp version, if available.
      0,   // libcusolverMp loaded.
      0,   // cuSOLVERMp handle created.
      0,   // cuSOLVERMp grid created.
      0,   // matrix descriptor for A created.
      0,   // raw cuSOLVER status from the failing call, if any.
      static_cast<int32_t>(a.size_bytes()),
      static_cast<int32_t>(n),
      static_cast<int32_t>(tile_size),
      static_cast<int32_t>(a.dimensions()[0]),
      static_cast<int32_t>(a.dimensions()[1]),
      -1,  // local NUMROC rows.
      -1,  // local NUMROC cols.
      static_cast<int32_t>(eigenvalues_out->size_bytes()),
      1,   // vector-producing SYEVD path.
      -1,  // syevd device workspace, KiB.
      -1,  // syevd host workspace, KiB.
      0,   // cusolverMpSyevd called.
      -1,  // syevd info value copied from device.
      -1,  // dtype code.
      static_cast<int32_t>(grid_mapping),
      0,   // matrix descriptor for Q created.
      0,   // matrix came from native JAX-buffer redistribution.
      0,   // reserved.
      0,   // reserved.
      0,   // reserved.
      0,   // reserved.
      0,   // reserved.
      0,   // reserved.
      0,   // reserved.
  };

  // Stage 2: bind cuSOLVERMp to the CUDA device that owns this rank's local
  // matrix shard.  This mirrors POTRS and avoids stale current-device state in
  // multi-device JAX callbacks.
  absl::StatusOr<int> buffer_device = DeviceForCudaPointer(a.untyped_data());
  if (!buffer_device.ok()) {
    probe[0] = kCudaDeviceFailed;
    return CopySyevdProbeToDevice(stream, probe, status_out);
  }
  int cuda_device = *buffer_device;
  cudaError_t cuda_status = cudaSetDevice(cuda_device);
  if (cuda_status != cudaSuccess) {
    probe[0] = kCudaDeviceFailed;
    return CopySyevdProbeToDevice(stream, probe, status_out);
  }
  probe[1] = cuda_device;
  CusolverMpDebug(-1, "syevd dispatch device=%d n=%lld tile=%lld grid=%lldx%lld "
                      "vectors=1",
                  cuda_device, static_cast<long long>(n),
                  static_cast<long long>(tile_size),
                  static_cast<long long>(process_rows),
                  static_cast<long long>(process_cols));

  // Stage 3: borrow the XLA-created NCCL communicator for this rank.  The
  // communicator rank is then converted to a cuSOLVERMp process-grid
  // coordinate using the requested row-major or column-major mapping.
  if (collective_params == nullptr || collective_cliques == nullptr) {
    probe[0] = kCollectiveContextMissing;
    return CopySyevdProbeToDevice(stream, probe, status_out);
  }

  absl::StatusOr<GpuCliqueKey> clique_key =
      AllAssignedDevicesP2PCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    probe[0] = kCliqueKeyFailed;
    return CopySyevdProbeToDevice(stream, probe, status_out);
  }

  absl::StatusOr<GpuCommunicator*> gpu_comm = collective_cliques->GetComm(
      *clique_key, collective_params->global_device_id);
  if (!gpu_comm.ok() || *gpu_comm == nullptr) {
    probe[0] = kCommunicatorMissing;
    return CopySyevdProbeToDevice(stream, probe, status_out);
  }

  void* platform_handle = (*gpu_comm)->platform_comm().handle;
  if (platform_handle == nullptr) {
    probe[0] = kNcclHandleMissing;
    return CopySyevdProbeToDevice(stream, probe, status_out);
  }
  ncclComm_t nccl_comm = reinterpret_cast<ncclComm_t>(platform_handle);

  int nccl_rank = -1;
  int nccl_count = -1;
  ncclResult_t rank_status = ncclCommUserRank(nccl_comm, &nccl_rank);
  ncclResult_t count_status = ncclCommCount(nccl_comm, &nccl_count);
  if (rank_status != ncclSuccess || count_status != ncclSuccess) {
    probe[0] = kNcclRankMismatch;
    return CopySyevdProbeToDevice(stream, probe, status_out);
  }
  probe[2] = nccl_rank;
  probe[3] = nccl_count;
  CusolverMpDebug(nccl_rank, "nccl communicator rank=%d count=%d",
                  nccl_rank, nccl_count);

  // Stage 4: validate the grid metadata and load the cuSOLVERMp symbols needed
  // by SYEVD.  If the runtime lacks cuSOLVERMp, the status vector reports that
  // cleanly instead of throwing an unresolved-symbol error.
  if (process_rows <= 0 || process_cols <= 0 ||
      process_rows * process_cols != nccl_count || n <= 0 ||
      tile_size <= 0) {
    probe[0] = kGridShapeMismatch;
    return CopySyevdProbeToDevice(stream, probe, status_out);
  }
  JAXMG_RETURN_IF_ERROR(
      ValidateCusolverMpGridMapping("cusolvermp_syevd", grid_mapping));
  if (!rank_map.empty()) {
    JAXMG_RETURN_IF_ERROR(ValidateStandardRankMapForGridMapping(
        "cusolvermp_syevd", rank_map, process_rows, process_cols,
        grid_mapping));
  }

  CusolverMpApi api = LoadCusolverMpApi(&probe);
  if (probe[0] != kProbeOk || !HasRequiredSymbols(api)) {
    if (api.library != nullptr) {
      dlclose(api.library);
    }
    return CopySyevdProbeToDevice(stream, probe, status_out);
  }
  if (!HasSyevdSymbols(api)) {
    probe[0] = kSolverSymbolMissing;
    dlclose(api.library);
    return CopySyevdProbeToDevice(stream, probe, status_out);
  }

  // Stage 5: create the cuSOLVERMp handle and grid against the borrowed NCCL
  // communicator.  The heavy local-buffer redistribution has already happened
  // in memory_redist; this stage only defines cuSOLVERMp's view of the grid.
  CusolverMpOpaqueHandle handle = nullptr;
  CusolverMpDebug(nccl_rank, "cusolverMpCreate begin device=%d stream=%p",
                  cuda_device, reinterpret_cast<void*>(cuda_stream));
  cusolverStatus_t cusolver_status =
      api.create(&handle, cuda_device, cuda_stream);
  CusolverMpDebug(nccl_rank, "cusolverMpCreate end status=%d handle=%p",
                  static_cast<int>(cusolver_status), handle);
  if (cusolver_status != CUSOLVER_STATUS_SUCCESS || handle == nullptr) {
    probe[0] = kCreateHandleFailed;
    probe[11] = static_cast<int32_t>(cusolver_status);
    dlclose(api.library);
    return CopySyevdProbeToDevice(stream, probe, status_out);
  }
  probe[8] = 1;

  int version = -1;
  cusolver_status = api.get_version(handle, &version);
  if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
    probe[0] = kGetVersionFailed;
    probe[11] = static_cast<int32_t>(cusolver_status);
    api.destroy(handle);
    dlclose(api.library);
    return CopySyevdProbeToDevice(stream, probe, status_out);
  }
  probe[6] = version;
  CusolverMpDebug(nccl_rank, "cusolverMp version=%d", version);

  CusolverMpOpaqueGrid grid = nullptr;
  CusolverMpDebug(nccl_rank,
                  "cusolverMpCreateDeviceGrid begin rows=%lld cols=%lld "
                  "mapping=%lld",
                  static_cast<long long>(process_rows),
                  static_cast<long long>(process_cols),
                  static_cast<long long>(grid_mapping));
  cusolver_status = api.create_grid(
      handle, &grid, nccl_comm, static_cast<int32_t>(process_rows),
      static_cast<int32_t>(process_cols), static_cast<int>(grid_mapping));
  CusolverMpDebug(nccl_rank, "cusolverMpCreateDeviceGrid end status=%d grid=%p",
                  static_cast<int>(cusolver_status), grid);
  if (cusolver_status != CUSOLVER_STATUS_SUCCESS || grid == nullptr) {
    probe[0] = kCreateGridFailed;
    probe[11] = static_cast<int32_t>(cusolver_status);
    api.destroy(handle);
    dlclose(api.library);
    return CopySyevdProbeToDevice(stream, probe, status_out);
  }
  probe[9] = 1;

  const auto [process_row, process_col] = ProcessCoordFromRank(
      nccl_rank, process_rows, process_cols, grid_mapping);
  CusolverMpDebug(nccl_rank, "dispatch process_coord=(%d,%d) element_type=%d",
                  process_row, process_col, static_cast<int>(a.element_type()));

  // Stage 6: dispatch on dtype and run the shared SYEVD helper.  The helper
  // keeps d_A and d_Z separate because vector-producing SYEVD overwrites A.
  absl::Status syevd_status;
  switch (a.element_type()) {
    case F32:
      probe[25] = 1;
      syevd_status = RunCusolverMpSyevd<float>(
          api, handle, grid, cuda_stream, n, tile_size, a.dimensions()[0],
          a.dimensions()[1], process_row, process_col, a, eigenvalues_out,
          work_out, vectors_out, &probe);
      break;
    case F64:
      probe[25] = 2;
      syevd_status = RunCusolverMpSyevd<double>(
          api, handle, grid, cuda_stream, n, tile_size, a.dimensions()[0],
          a.dimensions()[1], process_row, process_col, a, eigenvalues_out,
          work_out, vectors_out, &probe);
      break;
    case C64:
      probe[25] = 3;
      syevd_status = RunCusolverMpSyevd<cuFloatComplex>(
          api, handle, grid, cuda_stream, n, tile_size, a.dimensions()[0],
          a.dimensions()[1], process_row, process_col, a, eigenvalues_out,
          work_out, vectors_out, &probe);
      break;
    case C128:
      probe[25] = 4;
      syevd_status = RunCusolverMpSyevd<cuDoubleComplex>(
          api, handle, grid, cuda_stream, n, tile_size, a.dimensions()[0],
          a.dimensions()[1], process_row, process_col, a, eigenvalues_out,
          work_out, vectors_out, &probe);
      break;
    default:
      probe[0] = kUnsupportedDtype;
      break;
  }
  if (!syevd_status.ok()) {
    api.destroy_grid(grid);
    api.destroy(handle);
    dlclose(api.library);
    return syevd_status;
  }

  // Stage 7: release cuSOLVERMp objects and return status to Python.
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
  return CopySyevdProbeToDevice(stream, probe, status_out);
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
      tile_size, kCusolverMpGridMappingRowMajor,
      absl::Span<const int64_t>(), a, b, a_out, b_out, status_out,
      collective_params, collective_cliques, /*validate_solution=*/true);
}

absl::Status XlaCusolverMpSyevdProbePrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests) {
  return XlaCusolverMpDistributedPotrsProbePrepare(collective_params,
                                                  clique_requests);
}

absl::Status XlaCusolverMpSyevdProbeDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    int64_t process_rows, int64_t process_cols, int64_t n, int64_t tile_size,
    int64_t grid_mapping, int64_t use_private_stream,
    ffi::AnyBuffer token, ffi::Result<ffi::BufferR1<S32>> status_out,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  if (stream == nullptr || cuda_stream == nullptr) {
    return absl::InvalidArgumentError(
        "cusolvermp_syevd_probe requires XLA and CUDA streams");
  }
  if (token.dimensions().size() != 2) {
    return absl::InvalidArgumentError(
        "cusolvermp_syevd_probe expects a rank-2 token buffer");
  }
  if (status_out->dimensions().size() != 1 ||
      status_out->dimensions()[0] != kSyevdProbeSize) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "cusolvermp_syevd_probe expects status shape (%d,)",
        kSyevdProbeSize));
  }

  std::array<int32_t, kSyevdProbeSize> probe = {
      kProbeOk,
      -1,  // CUDA device selected for this FFI invocation.
      -1,  // NCCL rank reported by the borrowed communicator.
      -1,  // NCCL communicator size.
      static_cast<int32_t>(process_rows),
      static_cast<int32_t>(process_cols),
      -1,  // cuSOLVERMp version, if available.
      0,   // libcusolverMp loaded.
      0,   // cuSOLVERMp handle created.
      0,   // cuSOLVERMp grid for A created.
      0,   // matrix descriptor for A created.
      0,   // raw cuSOLVER status from the failing call, if any.
      static_cast<int32_t>(token.size_bytes()),
      static_cast<int32_t>(n),
      static_cast<int32_t>(tile_size),
      static_cast<int32_t>(token.dimensions()[0]),
      static_cast<int32_t>(token.dimensions()[1]),
      -1,  // local NUMROC rows.
      -1,  // local NUMROC cols.
      -1,  // eigenvalue allocation, KiB.
      1,   // vector-producing SYEVD path.
      -1,  // syevd device workspace, KiB.
      -1,  // syevd host workspace, KiB.
      0,   // cusolverMpSyevd called.
      -1,  // syevd info value copied from device.
      -1,  // dtype code.
      static_cast<int32_t>(grid_mapping),
      0,   // matrix descriptor for Q created.
      0,   // matrix came from cuSOLVERMp scatter.
      0,   // A scatter called.
      0,   // Q scatter called.
      0,   // cuSOLVERMp grid for Q created.
      static_cast<int32_t>(use_private_stream != 0),
      0,   // reserved.
      0,   // reserved.
      0,   // reserved.
  };

  absl::StatusOr<int> buffer_device = DeviceForCudaPointer(token.untyped_data());
  if (!buffer_device.ok()) {
    probe[0] = kCudaDeviceFailed;
    return CopySyevdProbeToDevice(stream, probe, status_out);
  }
  int cuda_device = *buffer_device;
  cudaError_t cuda_status = cudaSetDevice(cuda_device);
  if (cuda_status != cudaSuccess) {
    probe[0] = kCudaDeviceFailed;
    return CopySyevdProbeToDevice(stream, probe, status_out);
  }
  probe[1] = cuda_device;
  CusolverMpDebug(-1,
                  "syevd probe dispatch device=%d n=%lld tile=%lld "
                  "grid=%lldx%lld mapping=%lld use_private_stream=%lld",
                  cuda_device, static_cast<long long>(n),
                  static_cast<long long>(tile_size),
                  static_cast<long long>(process_rows),
                  static_cast<long long>(process_cols),
                  static_cast<long long>(grid_mapping),
                  static_cast<long long>(use_private_stream));

  if (collective_params == nullptr || collective_cliques == nullptr) {
    probe[0] = kCollectiveContextMissing;
    return CopySyevdProbeToDevice(stream, probe, status_out);
  }

  absl::StatusOr<GpuCliqueKey> clique_key =
      AllAssignedDevicesP2PCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    probe[0] = kCliqueKeyFailed;
    return CopySyevdProbeToDevice(stream, probe, status_out);
  }

  absl::StatusOr<GpuCommunicator*> gpu_comm = collective_cliques->GetComm(
      *clique_key, collective_params->global_device_id);
  if (!gpu_comm.ok() || *gpu_comm == nullptr) {
    probe[0] = kCommunicatorMissing;
    return CopySyevdProbeToDevice(stream, probe, status_out);
  }

  void* platform_handle = (*gpu_comm)->platform_comm().handle;
  if (platform_handle == nullptr) {
    probe[0] = kNcclHandleMissing;
    return CopySyevdProbeToDevice(stream, probe, status_out);
  }
  ncclComm_t nccl_comm = reinterpret_cast<ncclComm_t>(platform_handle);

  int nccl_rank = -1;
  int nccl_count = -1;
  ncclResult_t rank_status = ncclCommUserRank(nccl_comm, &nccl_rank);
  ncclResult_t count_status = ncclCommCount(nccl_comm, &nccl_count);
  if (rank_status != ncclSuccess || count_status != ncclSuccess) {
    probe[0] = kNcclRankMismatch;
    return CopySyevdProbeToDevice(stream, probe, status_out);
  }
  probe[2] = nccl_rank;
  probe[3] = nccl_count;
  CusolverMpDebug(nccl_rank, "syevd probe nccl rank=%d count=%d", nccl_rank,
                  nccl_count);

  if (process_rows <= 0 || process_cols <= 0 ||
      process_rows * process_cols != nccl_count || n <= 0 ||
      tile_size <= 0) {
    probe[0] = kGridShapeMismatch;
    return CopySyevdProbeToDevice(stream, probe, status_out);
  }
  JAXMG_RETURN_IF_ERROR(
      ValidateCusolverMpGridMapping("cusolvermp_syevd_probe", grid_mapping));

  CusolverMpApi api = LoadCusolverMpApi(&probe);
  if (probe[0] != kProbeOk || !HasRequiredSymbols(api)) {
    if (api.library != nullptr) {
      dlclose(api.library);
    }
    return CopySyevdProbeToDevice(stream, probe, status_out);
  }
  if (api.scatter_h2d == nullptr) {
    probe[0] = kScatterSymbolMissing;
    dlclose(api.library);
    return CopySyevdProbeToDevice(stream, probe, status_out);
  }
  if (!HasSyevdSymbols(api)) {
    probe[0] = kSolverSymbolMissing;
    dlclose(api.library);
    return CopySyevdProbeToDevice(stream, probe, status_out);
  }

  cudaStream_t solver_stream = cuda_stream;
  cudaStream_t private_stream = nullptr;
  auto destroy_private_stream = [&]() {
    if (private_stream != nullptr) {
      cudaStreamSynchronize(private_stream);
      cudaStreamDestroy(private_stream);
      private_stream = nullptr;
    }
  };
  if (use_private_stream != 0) {
    cuda_status =
        cudaStreamCreateWithFlags(&private_stream, cudaStreamNonBlocking);
    CusolverMpDebug(nccl_rank,
                    "syevd probe private stream create status=%d stream=%p",
                    static_cast<int>(cuda_status),
                    reinterpret_cast<void*>(private_stream));
    if (cuda_status != cudaSuccess || private_stream == nullptr) {
      probe[0] = kDeviceAllocFailed;
      probe[11] = static_cast<int32_t>(cuda_status);
      dlclose(api.library);
      return CopySyevdProbeToDevice(stream, probe, status_out);
    }
    solver_stream = private_stream;
  }

  CusolverMpOpaqueHandle handle = nullptr;
  cusolverStatus_t cusolver_status =
      api.create(&handle, cuda_device, solver_stream);
  CusolverMpDebug(nccl_rank, "syevd probe cusolverMpCreate status=%d handle=%p",
                  static_cast<int>(cusolver_status), handle);
  if (cusolver_status != CUSOLVER_STATUS_SUCCESS || handle == nullptr) {
    probe[0] = kCreateHandleFailed;
    probe[11] = static_cast<int32_t>(cusolver_status);
    destroy_private_stream();
    dlclose(api.library);
    return CopySyevdProbeToDevice(stream, probe, status_out);
  }
  probe[8] = 1;

  int version = -1;
  cusolver_status = api.get_version(handle, &version);
  if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
    probe[0] = kGetVersionFailed;
    probe[11] = static_cast<int32_t>(cusolver_status);
    api.destroy(handle);
    destroy_private_stream();
    dlclose(api.library);
    return CopySyevdProbeToDevice(stream, probe, status_out);
  }
  probe[6] = version;

  CusolverMpOpaqueGrid grid_a = nullptr;
  CusolverMpOpaqueGrid grid_q = nullptr;
  cusolver_status = api.create_grid(
      handle, &grid_a, nccl_comm, static_cast<int32_t>(process_rows),
      static_cast<int32_t>(process_cols), static_cast<int>(grid_mapping));
  CusolverMpDebug(nccl_rank,
                  "syevd probe create grid_a status=%d grid=%p",
                  static_cast<int>(cusolver_status), grid_a);
  if (cusolver_status != CUSOLVER_STATUS_SUCCESS || grid_a == nullptr) {
    probe[0] = kCreateGridFailed;
    probe[11] = static_cast<int32_t>(cusolver_status);
    api.destroy(handle);
    destroy_private_stream();
    dlclose(api.library);
    return CopySyevdProbeToDevice(stream, probe, status_out);
  }
  probe[9] = 1;

  cusolver_status = api.create_grid(
      handle, &grid_q, nccl_comm, static_cast<int32_t>(process_rows),
      static_cast<int32_t>(process_cols), static_cast<int>(grid_mapping));
  CusolverMpDebug(nccl_rank,
                  "syevd probe create grid_q status=%d grid=%p",
                  static_cast<int>(cusolver_status), grid_q);
  if (cusolver_status != CUSOLVER_STATUS_SUCCESS || grid_q == nullptr) {
    probe[0] = kCreateGridFailed;
    probe[11] = static_cast<int32_t>(cusolver_status);
    api.destroy_grid(grid_a);
    api.destroy(handle);
    destroy_private_stream();
    dlclose(api.library);
    return CopySyevdProbeToDevice(stream, probe, status_out);
  }
  probe[31] = 1;

  const auto [process_row, process_col] = ProcessCoordFromRank(
      nccl_rank, process_rows, process_cols, grid_mapping);

  absl::Status syevd_status;
  switch (token.element_type()) {
    case F32:
      probe[25] = 1;
      syevd_status = RunCusolverMpSyevdSampleProbe<float>(
          api, handle, grid_a, grid_q, solver_stream, n, tile_size, process_row,
          process_col, &probe);
      break;
    case F64:
      probe[25] = 2;
      syevd_status = RunCusolverMpSyevdSampleProbe<double>(
          api, handle, grid_a, grid_q, solver_stream, n, tile_size, process_row,
          process_col, &probe);
      break;
    case C64:
      probe[25] = 3;
      syevd_status = RunCusolverMpSyevdSampleProbe<cuFloatComplex>(
          api, handle, grid_a, grid_q, solver_stream, n, tile_size, process_row,
          process_col, &probe);
      break;
    case C128:
      probe[25] = 4;
      syevd_status = RunCusolverMpSyevdSampleProbe<cuDoubleComplex>(
          api, handle, grid_a, grid_q, solver_stream, n, tile_size, process_row,
          process_col, &probe);
      break;
    default:
      probe[0] = kUnsupportedDtype;
      break;
  }
  if (!syevd_status.ok()) {
    api.destroy_grid(grid_q);
    api.destroy_grid(grid_a);
    api.destroy(handle);
    destroy_private_stream();
    dlclose(api.library);
    return syevd_status;
  }

  if (probe[0] == kProbeOk) {
    cusolver_status = api.destroy_grid(grid_q);
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
      probe[0] = kDestroyGridFailed;
      probe[11] = static_cast<int32_t>(cusolver_status);
    }
  } else {
    api.destroy_grid(grid_q);
  }

  if (probe[0] == kProbeOk) {
    cusolver_status = api.destroy_grid(grid_a);
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
      probe[0] = kDestroyGridFailed;
      probe[11] = static_cast<int32_t>(cusolver_status);
    }
  } else {
    api.destroy_grid(grid_a);
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

  destroy_private_stream();
  dlclose(api.library);
  return CopySyevdProbeToDevice(stream, probe, status_out);
}

}  // namespace xla::gpu
