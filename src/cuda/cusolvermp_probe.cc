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
// Diagnostic cuSOLVERMp initialization probe.
//
// This file is deliberately not a solver implementation. It is the first
// boundary test between the existing XLA/NCCL communicator work and the future
// cuSOLVERMp backend:
//
//   1. request an XLA communicator over every assigned rank;
//   2. borrow the raw NCCL handle from XLA's GpuCommunicator;
//   3. dynamically load cuSOLVERMp if it is present on the host;
//   4. create/destroy a cuSOLVERMp handle;
//   5. create/destroy a cuSOLVERMp process grid from the borrowed NCCL handle;
//   6. create/destroy a tiny matrix descriptor compatible with that grid.
//
// CSD3's current CUDA 12.1 module and the pip nvidia-cusolver package do not
// ship cusolverMp.h/libcusolverMp.so. To keep the rest of the backend
// buildable, this probe uses dlopen/dlsym and returns a device status vector
// when cuSOLVERMp is absent instead of introducing a hard link dependency.
// Once the production cuSOLVERMp dependency is available, this file can be
// tightened to include the real header and link against the library directly.

#include <algorithm>
#include <array>
#include <cstdint>
#include <dlfcn.h>
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
};

constexpr int kProbeSize = 16;
constexpr int kScatterProbeSize = 24;

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

absl::Status CopyScatterProbeToDevice(
    se::Stream* stream, const std::array<int32_t, kScatterProbeSize>& probe,
    ffi::Result<ffi::BufferR1<S32>> out) {
  se::DeviceAddress<int32_t> dst = out->device_memory();
  return stream->MemcpyH2D(absl::MakeConstSpan(probe), &dst);
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

}  // namespace xla::gpu
