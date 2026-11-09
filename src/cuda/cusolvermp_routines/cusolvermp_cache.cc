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
// Process-wide cuSOLVERMp handle and grid cache (see cusolvermp_cache.h).

#include "cusolvermp_cache.h"

#include <cstdlib>
#include <mutex>
#include <vector>

namespace xla::gpu {
namespace {

struct CachedHandle {
  int device;
  cusolverMpHandle_t handle;
};

struct CachedGrid {
  cusolverMpHandle_t handle;
  ncclComm_t comm;
  int32_t rows;
  int32_t cols;
  cusolverMpGridMapping_t mapping;
  cusolverMpGrid_t grid;
};

struct Cache {
  std::mutex mutex;
  std::vector<CachedHandle> handles;
  std::vector<CachedGrid> grids;
};

// Intentionally leaked: cached objects must outlive every FFI call, and
// destroying them from a static destructor could run after XLA has destroyed
// the communicators the grids were built on.
Cache& GetCache() {
  static Cache* cache = new Cache();
  return *cache;
}

}  // namespace

bool CusolverMpCacheEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("JAXMG_CUSOLVERMP_CACHE");
    return value == nullptr || value[0] != '0';
  }();
  return enabled;
}

cusolverStatus_t AcquireCachedCusolverMpHandle(const CusolverMpCacheApi& api,
                                               int device, cudaStream_t stream,
                                               cusolverMpHandle_t* handle) {
  Cache& cache = GetCache();
  std::lock_guard<std::mutex> lock(cache.mutex);
  for (const CachedHandle& entry : cache.handles) {
    if (entry.device == device) {
      const cusolverStatus_t status = api.set_stream(entry.handle, stream);
      if (status == CUSOLVER_STATUS_SUCCESS) *handle = entry.handle;
      return status;
    }
  }
  const cusolverStatus_t status = api.create(handle, device, stream);
  if (status == CUSOLVER_STATUS_SUCCESS && *handle != nullptr) {
    cache.handles.push_back(CachedHandle{device, *handle});
  }
  return status;
}

cusolverStatus_t AcquireCachedCusolverMpGrid(const CusolverMpCacheApi& api,
                                             cusolverMpHandle_t handle,
                                             ncclComm_t comm, int32_t rows,
                                             int32_t cols,
                                             cusolverMpGridMapping_t mapping,
                                             cusolverMpGrid_t* grid) {
  Cache& cache = GetCache();
  std::lock_guard<std::mutex> lock(cache.mutex);
  for (const CachedGrid& entry : cache.grids) {
    if (entry.handle == handle && entry.comm == comm && entry.rows == rows &&
        entry.cols == cols && entry.mapping == mapping) {
      *grid = entry.grid;
      return CUSOLVER_STATUS_SUCCESS;
    }
  }
  const cusolverStatus_t status =
      api.create_grid(handle, grid, comm, rows, cols, mapping);
  if (status == CUSOLVER_STATUS_SUCCESS && *grid != nullptr) {
    cache.grids.push_back(
        CachedGrid{handle, comm, rows, cols, mapping, *grid});
  }
  return status;
}

bool IsCachedCusolverMpHandle(cusolverMpHandle_t handle) {
  Cache& cache = GetCache();
  std::lock_guard<std::mutex> lock(cache.mutex);
  for (const CachedHandle& entry : cache.handles) {
    if (entry.handle == handle) return true;
  }
  return false;
}

bool IsCachedCusolverMpGrid(cusolverMpGrid_t grid) {
  Cache& cache = GetCache();
  std::lock_guard<std::mutex> lock(cache.mutex);
  for (const CachedGrid& entry : cache.grids) {
    if (entry.grid == grid) return true;
  }
  return false;
}

}  // namespace xla::gpu
