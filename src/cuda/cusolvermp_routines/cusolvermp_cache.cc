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
// Cached cuSOLVERMp setup with borrowed XLA communicator lifetime tracking.

#include "cusolvermp_cache.h"

#include <cstdlib>
#include <map>
#include <tuple>
#include <utility>

#include "absl/base/no_destructor.h"

namespace xla::gpu {
namespace {

struct CacheLifetime {
  std::mutex mutex;
  bool communicator_retired = false;
};

// XLA destroys tied objects before their clique's communicators. Taking the
// execution lock prevents destruction during a solve. This does not intercept
// NCCL aborts, which are managed by XLA's distributed failure handling.
struct CliqueLifetime {
  explicit CliqueLifetime(std::shared_ptr<CacheLifetime> lifetime)
      : lifetime(std::move(lifetime)) {}
  ~CliqueLifetime() {
    if (!armed) return;
    std::lock_guard<std::mutex> lock(lifetime->mutex);
    lifetime->communicator_retired = true;
  }
  std::shared_ptr<CacheLifetime> lifetime;
  bool armed = false;
};

}  // namespace

struct CusolverMpCache::State {
  State(CusolverMpCacheApi api, int device) : api(api), device(device) {}
  CusolverMpCacheApi api;
  int device;
  std::shared_ptr<CacheLifetime> lifetime = std::make_shared<CacheLifetime>();
  cusolverMpHandle_t handle = nullptr;
  std::map<GpuCliqueKey, tsl::TiedRef<CliqueLifetime>> cliques;
  std::map<std::tuple<GpuCliqueKey, int32_t, int32_t, int32_t>, cusolverMpGrid_t>
      grids;
};

CusolverMpCache::CusolverMpCache(CusolverMpCacheApi api, int device)
    : state_(std::make_unique<State>(api, device)) {}

CusolverMpCache& GetCusolverMpCache(CusolverMpCacheApi api, int device) {
  // StreamExecutor resource IDs are allocated by a library-local counter;
  // JAXMg's static XLA copy must not allocate IDs in jaxlib's resource registry.
  struct Registry {
    std::mutex mutex;
    std::map<int, std::unique_ptr<CusolverMpCache>> devices;
  };
  // Like XLA's communicator cache, retain setup until process exit. Collective
  // destruction in a static destructor could run after NCCL/CUDA shutdown.
  // shortcut: release native setup when XLA offers coordinated pre-abort teardown.
  static absl::NoDestructor<Registry> registry;
  std::lock_guard<std::mutex> lock(registry->mutex);
  auto& cache = registry->devices[device];
  if (cache == nullptr) cache = std::make_unique<CusolverMpCache>(api, device);
  return *cache;
}

CusolverMpCache::~CusolverMpCache() = default;

bool CusolverMpCacheEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("JAXMG_CUSOLVERMP_CACHE");
    return value == nullptr || value[0] != '0';
  }();
  return enabled;
}

absl::Status CusolverMpCache::Lock(std::unique_lock<std::mutex>* lock) {
  *lock = std::unique_lock<std::mutex>(state_->lifetime->mutex);
  if (state_->lifetime->communicator_retired) {
    return absl::FailedPreconditionError(
        "JAXMg's cached cuSOLVERMp context used an XLA communicator that has "
        "been retired. Restart the distributed processes; set "
        "JAXMG_CUSOLVERMP_CACHE=0 before startup if communicator replacement "
        "is required.");
  }
  return absl::OkStatus();
}

cusolverStatus_t CusolverMpCache::GetHandle(cudaStream_t stream,
                                           cusolverMpHandle_t* handle) {
  cusolverStatus_t status;
  if (state_->handle == nullptr) {
    cusolverMpHandle_t created = nullptr;
    status = state_->api.create(&created, state_->device, stream);
    if (status == CUSOLVER_STATUS_SUCCESS) state_->handle = created;
  } else {
    status = state_->api.set_stream(state_->handle, stream);
  }
  if (status == CUSOLVER_STATUS_SUCCESS) *handle = state_->handle;
  return status;
}

absl::Status CusolverMpCache::TrackClique(CollectiveCliques* cliques,
                                        const GpuCliqueKey& key) {
  if (state_->cliques.find(key) != state_->cliques.end()) return absl::OkStatus();
  auto lifetime = std::make_unique<CliqueLifetime>(state_->lifetime);
  auto* notification = lifetime.get();
  auto tied = cliques->Tie(key, std::move(lifetime));
  if (!tied.ok()) return tied.status();
  notification->armed = true;
  state_->cliques.emplace(key, std::move(*tied));
  return absl::OkStatus();
}

cusolverStatus_t CusolverMpCache::GetGrid(
    const GpuCliqueKey& key, ncclComm_t comm, int32_t rows, int32_t cols,
    cusolverMpGridMapping_t mapping, cusolverMpGrid_t* grid) {
  const auto grid_key =
      std::make_tuple(key, rows, cols, static_cast<int32_t>(mapping));
  auto entry = state_->grids.find(grid_key);
  if (entry != state_->grids.end()) {
    *grid = entry->second;
    return CUSOLVER_STATUS_SUCCESS;
  }
  const cusolverStatus_t status = state_->api.create_grid(
      state_->handle, grid, comm, rows, cols, mapping);
  if (status == CUSOLVER_STATUS_SUCCESS && *grid != nullptr) {
    state_->grids.emplace(grid_key, *grid);
  }
  return status;
}

}  // namespace xla::gpu
