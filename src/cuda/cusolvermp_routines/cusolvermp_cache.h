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
// Clique-owned cuSOLVERMp grids and shared handles, reused across FFI calls.
//
// Reusing library setup avoids repeated initialization without retaining matrix
// descriptors or explicit solver workspace. Each execution holds the cache lock
// until its CUDA work has completed. XLA owns each group's grids through Tie;
// healthy retirement destroys them before their borrowed communicator. The
// last group releases the shared handle. Aborted groups skip collective cleanup
// and leave the device cache unusable until the distributed job restarts.
// JAXMG_CUSOLVERMP_CACHE=0 retains per-call handle and grid ownership.

#ifndef JAXMG_CUSOLVERMP_CACHE_H_
#define JAXMG_CUSOLVERMP_CACHE_H_

#include <cstdint>
#include <memory>
#include <mutex>

#include <cusolverMp.h>

#include "absl/status/status.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"

namespace xla::gpu {

// Library entry points used for cached setup, stream selection, and cleanup.
struct CusolverMpCacheApi {
  cusolverStatus_t (*create)(cusolverMpHandle_t*, int, cudaStream_t);
  cusolverStatus_t (*set_stream)(cusolverMpHandle_t, cudaStream_t);
  cusolverStatus_t (*create_grid)(cusolverMpHandle_t, cusolverMpGrid_t*,
                                  ncclComm_t, int32_t, int32_t,
                                  cusolverMpGridMapping_t);
  cusolverStatus_t (*destroy_grid)(cusolverMpGrid_t);
  cusolverStatus_t (*destroy)(cusolverMpHandle_t);
};

// False when JAXMG_CUSOLVERMP_CACHE is set to 0.
bool CusolverMpCacheEnabled();

// One handle per device, shared by all solver routines and live clique groups.
// Grids match both the XLA clique and NCCL handle; expired references are never
// reused, including when a new communicator occupies an old address.
class CusolverMpCache {
 public:
  CusolverMpCache(CusolverMpCacheApi api, int device);
  ~CusolverMpCache();

  // Acquire exclusive access before setting the stream or using any resource.
  // The caller must finish its CUDA work before releasing the returned lock.
  absl::Status Lock(std::unique_lock<std::mutex>* lock);

  // These methods require the execution to hold the lock obtained above.
  // Track the clique before creation so every resource has a live owner.
  cusolverStatus_t GetHandle(cudaStream_t stream, cusolverMpHandle_t* handle);
  absl::Status TrackClique(CollectiveCliques* cliques, const GpuCliqueKey& key,
                          GpuCommunicator* comm);
  cusolverStatus_t GetGrid(const GpuCliqueKey& key, ncclComm_t comm,
                          int32_t rows, int32_t cols,
                          cusolverMpGridMapping_t mapping,
                          cusolverMpGrid_t* grid);

 private:
  struct State;
  std::unique_ptr<State> state_;
};

// Returns stable per-device metadata. Native resources belong to live cliques;
// retaining this registry avoids unordered collective cleanup at process exit.
CusolverMpCache& GetCusolverMpCache(CusolverMpCacheApi api, int device);

}  // namespace xla::gpu

#endif  // JAXMG_CUSOLVERMP_CACHE_H_
