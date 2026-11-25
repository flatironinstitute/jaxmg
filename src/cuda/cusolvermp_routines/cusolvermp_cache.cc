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
// Process-local cuSOLVERMp handles and grids shared across solver calls.
//
// The public Collectives FFI does not expose communicator retirement. Native
// resources are therefore retained until process exit, without running
// collective destructors in an unknown shutdown order. The execution lock
// covers both setup and CUDA completion; descriptors and workspaces remain
// call-local.

#include "cusolvermp_cache.h"

#include <cstdlib>
#include <map>
#include <tuple>

#include "absl/base/no_destructor.h"

namespace jaxmg {

struct CusolverMpCache::State {
  State(CusolverMpCacheApi api, int device) : api(api), device(device) {}
  CusolverMpCacheApi api;
  int device;
  std::mutex mutex;
  cusolverMpHandle_t handle = nullptr;
  bool poisoned = false;
  std::map<std::tuple<uintptr_t, int32_t, int32_t, int32_t>, cusolverMpGrid_t>
      grids;
};

CusolverMpCache::CusolverMpCache(CusolverMpCacheApi api, int device)
    : state_(std::make_unique<State>(api, device)) {}

CusolverMpCache::~CusolverMpCache() = default;

CusolverMpCache& GetCusolverMpCache(CusolverMpCacheApi api, int device) {
  struct Registry {
    std::mutex mutex;
    std::map<int, std::unique_ptr<CusolverMpCache>> devices;
  };
  static absl::NoDestructor<Registry> registry;
  std::lock_guard<std::mutex> lock(registry->mutex);
  auto& cache = registry->devices[device];
  if (cache == nullptr) cache = std::make_unique<CusolverMpCache>(api, device);
  return *cache;
}

bool CusolverMpCacheEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("JAXMG_CUSOLVERMP_CACHE");
    return value == nullptr || value[0] != '0';
  }();
  return enabled;
}

absl::Status CusolverMpCache::Lock(std::unique_lock<std::mutex>* lock) {
  *lock = std::unique_lock<std::mutex>(state_->mutex);
  if (state_->poisoned) {
    return absl::FailedPreconditionError(
        "JAXMg's cached cuSOLVERMp context is unusable after a setup failure. "
        "Restart the distributed processes.");
  }
  return absl::OkStatus();
}

cusolverStatus_t CusolverMpCache::GetHandle(cudaStream_t stream,
                                            cusolverMpHandle_t* handle) {
  *handle = nullptr;
  if (state_->poisoned) return CUSOLVER_STATUS_INTERNAL_ERROR;
  cusolverStatus_t status;
  if (state_->handle == nullptr) {
    cusolverMpHandle_t created = nullptr;
    status = state_->api.create(&created, state_->device, stream);
    if (status == CUSOLVER_STATUS_SUCCESS) state_->handle = created;
  } else {
    status = state_->api.set_stream(state_->handle, stream);
  }
  if (status != CUSOLVER_STATUS_SUCCESS) state_->poisoned = true;
  if (status == CUSOLVER_STATUS_SUCCESS) *handle = state_->handle;
  return status;
}

cusolverStatus_t CusolverMpCache::GetGrid(ncclComm_t comm, int32_t rows,
                                          int32_t cols,
                                          cusolverMpGridMapping_t mapping,
                                          cusolverMpGrid_t* grid) {
  *grid = nullptr;
  if (state_->poisoned) return CUSOLVER_STATUS_INTERNAL_ERROR;
  if (state_->handle == nullptr) return CUSOLVER_STATUS_NOT_INITIALIZED;
  // shortcut: the FFI has no communicator-generation/retirement token. Revisit
  // this key when one is available; restarting a distributed client requires
  // fresh worker processes, not reuse of this process-lifetime cache.
  const auto key = std::make_tuple(reinterpret_cast<uintptr_t>(comm), rows,
                                   cols, static_cast<int32_t>(mapping));
  auto found = state_->grids.find(key);
  if (found != state_->grids.end()) {
    *grid = found->second;
    return CUSOLVER_STATUS_SUCCESS;
  }
  cusolverMpGrid_t created = nullptr;
  const cusolverStatus_t status = state_->api.create_grid(
      state_->handle, &created, comm, rows, cols, mapping);
  if (status == CUSOLVER_STATUS_SUCCESS && created != nullptr) {
    state_->grids.emplace(key, created);
    *grid = created;
  } else {
    state_->poisoned = true;
  }
  return status;
}

}  // namespace jaxmg
