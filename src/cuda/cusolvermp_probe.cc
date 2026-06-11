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

struct CusolverMpApi {
  void* library = nullptr;
  CusolverMpCreateFn create = nullptr;
  CusolverMpDestroyFn destroy = nullptr;
  CusolverMpGetVersionFn get_version = nullptr;
  CusolverMpCreateDeviceGridFn create_grid = nullptr;
  CusolverMpDestroyGridFn destroy_grid = nullptr;
  CusolverMpCreateMatrixDescFn create_matrix_desc = nullptr;
  CusolverMpDestroyMatrixDescFn destroy_matrix_desc = nullptr;
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
};

constexpr int kProbeSize = 16;

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

CusolverMpApi LoadCusolverMpApi(std::array<int32_t, kProbeSize>* probe) {
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

}  // namespace xla::gpu
