// Copyright 2026 JAXMg contributors.
// SPDX-License-Identifier: Apache-2.0
//
// Exercise XLA's actual clique ownership with fake CUDA/cuSOLVERMp entry points.
// No numerical kernels or GPU allocations are needed for these lifecycle tests.

#include "cusolvermp_routines/cusolvermp_cache.h"

#include <cassert>
#include <csignal>
#include <cstdio>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include "absl/container/btree_map.h"
#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/backends/gpu/collectives/gpu_clique.h"
#include "xla/backends/gpu/collectives/gpu_cliques.h"
#include "xla/backends/gpu/collectives/loopback_communicator.h"

namespace {
struct FakeState {
  int device = 7;
  int handles_created = 0;
  int grids_created = 0;
  int stream_calls = 0;
  cusolverStatus_t stream_status = CUSOLVER_STATUS_SUCCESS;
  cusolverStatus_t destroy_grid_status = CUSOLVER_STATUS_SUCCESS;
  std::vector<std::string> events;
} state;
}  // namespace

extern "C" cudaError_t CUDARTAPI cudaGetDevice(int* device) {
  *device = state.device;
  return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaSetDevice(int device) {
  state.device = device;
  return cudaSuccess;
}

namespace xla::gpu {
namespace {

const CusolverMpCacheApi kApi{
    [](cusolverMpHandle_t* handle, int device, cudaStream_t) {
      assert(device == 3);
      *handle = reinterpret_cast<cusolverMpHandle_t>(++state.handles_created);
      return CUSOLVER_STATUS_SUCCESS;
    },
    [](cusolverMpHandle_t, cudaStream_t) {
      ++state.stream_calls;
      return state.stream_status;
    },
    [](cusolverMpHandle_t handle, cusolverMpGrid_t* grid, ncclComm_t,
       int32_t, int32_t, cusolverMpGridMapping_t) {
      assert(handle != nullptr);
      *grid = reinterpret_cast<cusolverMpGrid_t>(++state.grids_created);
      return CUSOLVER_STATUS_SUCCESS;
    },
    [](cusolverMpGrid_t grid) {
      assert(state.device == 3);
      state.events.push_back(
          "grid:" + std::to_string(reinterpret_cast<uintptr_t>(grid)));
      return state.destroy_grid_status;
    },
    [](cusolverMpHandle_t) {
      assert(state.device == 3);
      state.events.push_back("handle");
      return CUSOLVER_STATUS_SUCCESS;
    }};

class TestCommunicator : public LoopbackCommunicator {
 public:
  explicit TestCommunicator(int id)
      : LoopbackCommunicator(nullptr, 1, 0), id_(id) {}
  ~TestCommunicator() override {
    state.events.push_back("comm:" + std::to_string(id_));
  }
  PlatformCommunicatorHandle platform_comm() const override {
    return {reinterpret_cast<void*>(static_cast<uintptr_t>(id_))};
  }
  absl::Status HealthCheck() const override {
    return healthy ? absl::OkStatus() : absl::InternalError("aborted communicator");
  }
  bool healthy = true;

 private:
  int id_;
};

struct Group {
  explicit Group(int id) : key({GlobalDeviceId(0)}, 1, CommunicationId(id)) {
    auto communicator = std::make_unique<TestCommunicator>(id);
    comm = communicator.get();
    absl::btree_map<RankId, std::unique_ptr<Communicator>> comms;
    comms.emplace(RankId(0), std::move(communicator));
    owner = std::make_unique<LockableGpuClique>(
        key, std::nullopt, std::move(comms), false,
        std::make_shared<CancellationToken>());
  }

  CollectiveCliques Acquire() {
    AcquiredCliquesMap acquired;
    acquired.emplace(
        key, std::make_shared<LockableGpuClique::Lock>(owner->Acquire()));
    return CollectiveCliques(std::move(acquired));
  }

  GpuCliqueKey key;
  TestCommunicator* comm;
  std::unique_ptr<LockableGpuClique> owner;
};

cusolverMpGrid_t Prepare(CusolverMpCache& cache, Group& group,
                        int rows = 1, int cols = 1) {
  auto cliques = group.Acquire();
  std::unique_lock<std::mutex> lock;
  assert(cache.Lock(&lock).ok());
  assert(cache.TrackClique(&cliques, group.key, group.comm).ok());
  cusolverMpHandle_t handle = nullptr;
  assert(cache.GetHandle(nullptr, &handle) == CUSOLVER_STATUS_SUCCESS);
  cusolverMpGrid_t grid = nullptr;
  const auto comm =
      reinterpret_cast<ncclComm_t>(group.comm->platform_comm().handle);
  assert(cache.GetGrid(group.key, comm, rows, cols,
                       CUSOLVERMP_GRID_MAPPING_ROW_MAJOR, &grid) ==
         CUSOLVER_STATUS_SUCCESS);
  assert(handle != nullptr && grid != nullptr);
  return grid;
}

void TestReuseAndRetirement() {
  state = {};
  CusolverMpCache cache(kApi, 3);
  Group first(1), second(2);
  const auto grid = Prepare(cache, first, 2, 1);
  assert(Prepare(cache, first, 2, 1) == grid);
  Prepare(cache, first, 1, 2);
  Prepare(cache, second);
  assert(state.handles_created == 1 && state.grids_created == 3);
  assert(state.events.empty());

  // Shape order, not creation order; other groups keep the shared handle alive.
  first.owner.reset();
  assert((state.events == std::vector<std::string>{"grid:2", "grid:1", "comm:1"}));
  assert(state.device == 7);
  second.owner.reset();
  assert((state.events == std::vector<std::string>{
      "grid:2", "grid:1", "comm:1", "grid:3", "handle", "comm:2"}));
  assert(state.device == 7);

  // A replacement with the same key and NCCL address must not reuse old setup.
  Group replacement(1);
  assert(Prepare(cache, replacement, 2, 1) != grid);
  assert(state.handles_created == 2 && state.grids_created == 4);
  replacement.owner.reset();
}

void TestAbortedCommunicator() {
  state = {};
  CusolverMpCache cache(kApi, 3);
  Group failed(1), healthy(2);
  Prepare(cache, failed);
  Prepare(cache, healthy);
  failed.comm->healthy = false;
  failed.owner.reset();
  assert((state.events == std::vector<std::string>{"comm:1"}));
  {
    std::unique_lock<std::mutex> lock;
    assert(!cache.Lock(&lock).ok());
  }
  healthy.owner.reset();
  // The healthy group's grids can be freed, but the abandoned handle is retained.
  assert((state.events == std::vector<std::string>{"comm:1", "grid:2", "comm:2"}));
  assert(state.device == 7);
}

void TestStreamFailure() {
  state = {};
  CusolverMpCache cache(kApi, 3);
  Group group(1);
  Prepare(cache, group);
  {
    auto cliques = group.Acquire();
    std::unique_lock<std::mutex> lock;
    assert(cache.Lock(&lock).ok());
    cusolverMpHandle_t handle = nullptr;
    state.stream_status = CUSOLVER_STATUS_INVALID_VALUE;
    assert(cache.GetHandle(nullptr, &handle) == CUSOLVER_STATUS_INVALID_VALUE);
    assert(handle == nullptr);
    state.stream_status = CUSOLVER_STATUS_SUCCESS;
    assert(cache.GetHandle(nullptr, &handle) == CUSOLVER_STATUS_SUCCESS);
    state.stream_status = CUSOLVER_STATUS_INTERNAL_ERROR;
    assert(cache.GetHandle(nullptr, &handle) == CUSOLVER_STATUS_INTERNAL_ERROR);
    assert(handle == nullptr);
    const int calls = state.stream_calls;
    assert(cache.GetHandle(nullptr, &handle) == CUSOLVER_STATUS_INTERNAL_ERROR);
    assert(state.stream_calls == calls);
  }
  {
    std::unique_lock<std::mutex> lock;
    assert(!cache.Lock(&lock).ok());
  }
  group.owner.reset();
  assert((state.events == std::vector<std::string>{"grid:1", "handle", "comm:1"}));
}

void TestFailedTie() {
  state = {};
  CusolverMpCache cache(kApi, 3);
  Group group(1);
  {
    CollectiveCliques empty{AcquiredCliquesMap{}};
    std::unique_lock<std::mutex> lock;
    assert(cache.Lock(&lock).ok());
    assert(!cache.TrackClique(&empty, group.key, group.comm).ok());
  }
  Prepare(cache, group);
  group.owner.reset();
  assert((state.events == std::vector<std::string>{"grid:1", "handle", "comm:1"}));
}

void TestFatalGridCleanup() {
  const pid_t child = fork();
  assert(child >= 0);
  if (child == 0) {
    const rlimit no_core{0, 0};
    setrlimit(RLIMIT_CORE, &no_core);
    state = {};
    CusolverMpCache cache(kApi, 3);
    Group group(1);
    Prepare(cache, group);
    state.destroy_grid_status = CUSOLVER_STATUS_INTERNAL_ERROR;
    group.owner.reset();
    _exit(1);
  }
  int status = 0;
  assert(waitpid(child, &status, 0) == child);
  assert(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
}

}  // namespace
}  // namespace xla::gpu

int main() {
  xla::gpu::TestReuseAndRetirement();
  xla::gpu::TestAbortedCommunicator();
  xla::gpu::TestStreamFailure();
  xla::gpu::TestFailedTie();
  xla::gpu::TestFatalGridCleanup();
  std::puts("Cache reuse, clique retirement, and failure handling passed.");
}
