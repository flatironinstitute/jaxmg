// Copyright 2026 JAXMg contributors.
// Licensed under the Apache License, Version 2.0.

// Exercise setup reuse and failure handling without a GPU or an XLA runtime.
#include <cassert>
#include <chrono>
#include <future>

#include "cusolvermp_routines/cusolvermp_cache.h"

namespace {
int handles = 0, grids = 0, stream_changes = 0;
bool fail_stream = false, fail_grid = false;

cusolverStatus_t Create(cusolverMpHandle_t* handle, int, cudaStream_t) {
  *handle = reinterpret_cast<cusolverMpHandle_t>(++handles);
  return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t SetStream(cusolverMpHandle_t, cudaStream_t) {
  ++stream_changes;
  return fail_stream ? CUSOLVER_STATUS_INTERNAL_ERROR : CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t CreateGrid(cusolverMpHandle_t, cusolverMpGrid_t* grid,
                            ncclComm_t, int32_t, int32_t,
                            cusolverMpGridMapping_t) {
  if (fail_grid) return CUSOLVER_STATUS_INTERNAL_ERROR;
  *grid = reinterpret_cast<cusolverMpGrid_t>(++grids);
  return CUSOLVER_STATUS_SUCCESS;
}
}  // namespace

int main() {
  const jaxmg::CusolverMpCacheApi api{Create, SetStream, CreateGrid};
  auto& cache = jaxmg::GetCusolverMpCache(api, 0);
  assert(&cache == &jaxmg::GetCusolverMpCache(api, 0));
  assert(&cache != &jaxmg::GetCusolverMpCache(api, 1));
  std::unique_lock<std::mutex> lock;
  assert(cache.Lock(&lock).ok());
  cusolverMpHandle_t first, second;
  assert(cache.GetHandle(nullptr, &first) == CUSOLVER_STATUS_SUCCESS);
  assert(cache.GetHandle(reinterpret_cast<cudaStream_t>(1), &second) ==
         CUSOLVER_STATUS_SUCCESS);
  assert(first == second && handles == 1 && stream_changes == 1);
  const auto comm = reinterpret_cast<ncclComm_t>(1);
  const auto mapping = static_cast<cusolverMpGridMapping_t>(0);
  cusolverMpGrid_t a, b, c;
  assert(cache.GetGrid(comm, 2, 2, mapping, &a) == CUSOLVER_STATUS_SUCCESS);
  assert(cache.GetGrid(comm, 2, 2, mapping, &b) == CUSOLVER_STATUS_SUCCESS);
  assert(a == b && grids == 1);
  assert(cache.GetGrid(comm, 4, 1, mapping, &b) == CUSOLVER_STATUS_SUCCESS);
  assert(a != b);
  assert(cache.GetGrid(reinterpret_cast<ncclComm_t>(2), 2, 2, mapping, &c) ==
         CUSOLVER_STATUS_SUCCESS);
  assert(a != c && grids == 3);
  assert(cache.GetGrid(comm, 2, 2, static_cast<cusolverMpGridMapping_t>(1),
                       &c) == CUSOLVER_STATUS_SUCCESS);
  assert(a != c && grids == 4);

  // A second invocation cannot change the handle's stream before completion.
  auto waiter = std::async(std::launch::async, [&] {
    std::unique_lock<std::mutex> other;
    return cache.Lock(&other).ok();
  });
  assert(waiter.wait_for(std::chrono::milliseconds(20)) ==
         std::future_status::timeout);
  lock.unlock();
  assert(waiter.get());

  assert(cache.Lock(&lock).ok());
  fail_stream = true;
  assert(cache.GetHandle(nullptr, &second) == CUSOLVER_STATUS_INTERNAL_ERROR);
  assert(second == nullptr);
  lock.unlock();
  assert(!cache.Lock(&lock).ok());
  lock.unlock();
  fail_stream = false;

  jaxmg::CusolverMpCache failed_grid(api, 2);
  assert(failed_grid.Lock(&lock).ok());
  assert(failed_grid.GetHandle(nullptr, &second) == CUSOLVER_STATUS_SUCCESS);
  fail_grid = true;
  assert(failed_grid.GetGrid(comm, 2, 2, mapping, &a) ==
         CUSOLVER_STATUS_INTERNAL_ERROR);
  assert(a == nullptr);
  fail_grid = false;
  assert(failed_grid.GetGrid(comm, 2, 2, mapping, &a) ==
         CUSOLVER_STATUS_INTERNAL_ERROR);
  lock.unlock();
  // Cache destruction releases host metadata only; there are no collective
  // destructors that could run after XLA has already destroyed a communicator.
}
