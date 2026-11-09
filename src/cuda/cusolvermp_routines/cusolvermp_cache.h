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
// Process-wide cuSOLVERMp handles and device grids, reused across FFI calls.
//
// Setting up and tearing down a cuSOLVERMp handle and device grid in every call
// costs about 1 s per call on 8 GPUs (0.2 s on 2), whatever the matrix size.
// The cache creates them on first use and keeps them for the life of the
// process:
//
//   - one handle per CUDA device, pointed at the caller's stream on each call;
//   - one grid per (handle, NCCL communicator, process grid, grid mapping).
//
// JAXMG_CUSOLVERMP_CACHE=0 turns the cache off: every call then creates and
// destroys its own handle and grid, as before.
//
// The cache reaches cuSOLVERMp only through the function pointers it is given,
// so this file depends on cusolverMp.h alone.

#ifndef JAXMG_CUSOLVERMP_CACHE_H_
#define JAXMG_CUSOLVERMP_CACHE_H_

#include <cstdint>

#include <cusolverMp.h>

namespace xla::gpu {

// The cuSOLVERMp entry points the cache creates objects with.
struct CusolverMpCacheApi {
  cusolverStatus_t (*create)(cusolverMpHandle_t*, int, cudaStream_t);
  cusolverStatus_t (*set_stream)(cusolverMpHandle_t, cudaStream_t);
  cusolverStatus_t (*create_grid)(cusolverMpHandle_t, cusolverMpGrid_t*,
                                  ncclComm_t, int32_t, int32_t,
                                  cusolverMpGridMapping_t);
};

// False when JAXMG_CUSOLVERMP_CACHE is set to 0.
bool CusolverMpCacheEnabled();

// The handle of `device`, created on first use and set to `stream`.
cusolverStatus_t AcquireCachedCusolverMpHandle(const CusolverMpCacheApi& api,
                                               int device, cudaStream_t stream,
                                               cusolverMpHandle_t* handle);

// The grid of `handle` over `comm` with the given process grid and mapping,
// created on first use.
cusolverStatus_t AcquireCachedCusolverMpGrid(const CusolverMpCacheApi& api,
                                             cusolverMpHandle_t handle,
                                             ncclComm_t comm, int32_t rows,
                                             int32_t cols,
                                             cusolverMpGridMapping_t mapping,
                                             cusolverMpGrid_t* grid);

// Whether the cache owns `handle` / `grid`: owned objects are never destroyed.
bool IsCachedCusolverMpHandle(cusolverMpHandle_t handle);
bool IsCachedCusolverMpGrid(cusolverMpGrid_t grid);

}  // namespace xla::gpu

#endif  // JAXMG_CUSOLVERMP_CACHE_H_
