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
// Cached cuSOLVERMp setup owned by XLA communicator cliques.
//
// Each device serializes handle use across its live groups. A group's tied
// owner destroys its grids in a common order before XLA releases NCCL, and
// the last group releases the handle. Failed communicators skip collective
// cleanup; their retained resources are reclaimed only when the process exits.

#include "cusolvermp_cache.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <tuple>
#include <utility>

#include "absl/base/no_destructor.h"

namespace xla::gpu {
namespace {

struct CacheContext {
  CacheContext(CusolverMpCacheApi api, int device) : api(api), device(device) {}
  CusolverMpCacheApi api;
  int device;
  std::mutex mutex;
  cusolverMpHandle_t handle = nullptr;
  size_t live_cliques = 0;
  bool poisoned = false;
  bool abandoned = false;
};

[[noreturn]] void AbortCacheCleanup(const char* operation, int status) {
  std::fprintf(stderr,
               "JAXMg fatal: %s failed during cached resource cleanup "
               "(status %d); restart the distributed job.\n", operation, status);
  std::fflush(stderr);
  std::abort();
}

// One owner per communicator, rather than per grid, gives all ranks the same
// grid-destruction order. XLA destroys tied owners before their communicators.
struct CliqueResources {
  CliqueResources(std::shared_ptr<CacheContext> context, GpuCommunicator* comm)
      : context(std::move(context)), comm(comm) {}

  ~CliqueResources() {
    if (!armed) return;
    std::lock_guard<std::mutex> lock(context->mutex);
    int previous_device = -1;
    if (comm->HealthCheck().ok()) {
      cudaError_t cuda_status = cudaGetDevice(&previous_device);
      if (cuda_status != cudaSuccess) {
        AbortCacheCleanup("cudaGetDevice", static_cast<int>(cuda_status));
      }
      cuda_status = cudaSetDevice(context->device);
      if (cuda_status != cudaSuccess) {
        AbortCacheCleanup("cudaSetDevice", static_cast<int>(cuda_status));
      }
      for (const auto& entry : grids) {
        const cusolverStatus_t status = context->api.destroy_grid(entry.second);
        if (status != CUSOLVER_STATUS_SUCCESS) {
          AbortCacheCleanup("cusolverMpDestroyGrid", static_cast<int>(status));
        }
      }
    } else {
      // A collective destructor cannot safely run on an aborted communicator.
      context->poisoned = true;
      context->abandoned = true;
      std::fprintf(stderr,
                   "JAXMg: skipping cached grid cleanup on an unhealthy XLA "
                   "communicator; restart the distributed job.\n");
    }
    if (--context->live_cliques == 0 && context->handle != nullptr &&
        !context->abandoned) {
      const cusolverStatus_t status =
          context->api.destroy(std::exchange(context->handle, nullptr));
      if (status != CUSOLVER_STATUS_SUCCESS) {
        AbortCacheCleanup("cusolverMpDestroy", static_cast<int>(status));
      }
    }
    if (previous_device >= 0 && previous_device != context->device) {
      const cudaError_t status = cudaSetDevice(previous_device);
      if (status != cudaSuccess) {
        AbortCacheCleanup("cudaSetDevice", static_cast<int>(status));
      }
    }
  }

  std::shared_ptr<CacheContext> context;
  GpuCommunicator* comm;
  std::map<std::tuple<int32_t, int32_t, int32_t>, cusolverMpGrid_t> grids;
  bool armed = false;
};

}  // namespace

struct CusolverMpCache::State {
  State(CusolverMpCacheApi api, int device)
      : context(std::make_shared<CacheContext>(api, device)) {}
  std::shared_ptr<CacheContext> context;
  std::map<std::pair<GpuCliqueKey, uintptr_t>, tsl::TiedRef<CliqueResources>> cliques;
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
  // Retain only the registry at shutdown. Native cleanup follows XLA's clique
  // lifetime, not the unordered destruction of extension-module statics.
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
  *lock = std::unique_lock<std::mutex>(state_->context->mutex);
  if (state_->context->poisoned) {
    return absl::FailedPreconditionError(
        "JAXMg's cached cuSOLVERMp context is unusable after a handle or "
        "communicator failure. Restart the distributed processes.");
  }
  return absl::OkStatus();
}

cusolverStatus_t CusolverMpCache::GetHandle(cudaStream_t stream,
                                           cusolverMpHandle_t* handle) {
  auto& context = *state_->context;
  *handle = nullptr;
  if (context.poisoned) return CUSOLVER_STATUS_INTERNAL_ERROR;
  cusolverStatus_t status;
  if (context.handle == nullptr) {
    cusolverMpHandle_t created = nullptr;
    status = context.api.create(&created, context.device, stream);
    if (status == CUSOLVER_STATUS_SUCCESS) context.handle = created;
  } else {
    status = context.api.set_stream(context.handle, stream);
  }
  if (status == CUSOLVER_STATUS_INTERNAL_ERROR) context.poisoned = true;
  if (status == CUSOLVER_STATUS_SUCCESS) *handle = context.handle;
  return status;
}

absl::Status CusolverMpCache::TrackClique(CollectiveCliques* cliques,
                                        const GpuCliqueKey& key,
                                        GpuCommunicator* comm) {
  // Expiry distinguishes replacement cliques even if NCCL reuses an address.
  for (auto it = state_->cliques.begin(); it != state_->cliques.end();) {
    if (it->second.Expired()) {
      it = state_->cliques.erase(it);
    } else {
      ++it;
    }
  }
  // Separate live XLA clients can assign the same clique key to different comms.
  const auto identity =
      std::make_pair(key, reinterpret_cast<uintptr_t>(comm->platform_comm().handle));
  if (state_->cliques.find(identity) != state_->cliques.end()) return absl::OkStatus();
  auto resources = std::make_unique<CliqueResources>(state_->context, comm);
  auto* owner = resources.get();
  auto tied = cliques->Tie(key, std::move(resources));
  if (!tied.ok()) return tied.status();
  owner->armed = true;
  ++state_->context->live_cliques;
  state_->cliques.emplace(identity, std::move(*tied));
  return absl::OkStatus();
}

cusolverStatus_t CusolverMpCache::GetGrid(
    const GpuCliqueKey& key, ncclComm_t comm, int32_t rows, int32_t cols,
    cusolverMpGridMapping_t mapping, cusolverMpGrid_t* grid) {
  *grid = nullptr;
  if (state_->context->poisoned) return CUSOLVER_STATUS_INTERNAL_ERROR;
  const auto clique = state_->cliques.find(
      std::make_pair(key, reinterpret_cast<uintptr_t>(comm)));
  if (clique == state_->cliques.end()) return CUSOLVER_STATUS_NOT_INITIALIZED;
  auto resources = clique->second.Lock();
  if (resources == nullptr) return CUSOLVER_STATUS_NOT_INITIALIZED;
  const auto grid_key = std::make_tuple(rows, cols, static_cast<int32_t>(mapping));
  auto entry = resources->grids.find(grid_key);
  if (entry != resources->grids.end()) {
    *grid = entry->second;
    return CUSOLVER_STATUS_SUCCESS;
  }
  const cusolverStatus_t status = state_->context->api.create_grid(
      state_->context->handle, grid, comm, rows, cols, mapping);
  if (status == CUSOLVER_STATUS_SUCCESS && *grid != nullptr) {
    resources->grids.emplace(grid_key, *grid);
  }
  return status;
}

}  // namespace xla::gpu
