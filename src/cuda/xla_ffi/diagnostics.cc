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

#include "diagnostics.h"

#include <cstdio>
#include <cstdlib>

namespace xla::gpu {
namespace {

bool TruthyEnv(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

}  // namespace

bool JaxmgCudaDiagnosticsEnabled() {
  return TruthyEnv("JAXMG_CUDA_DIAGNOSTICS");
}

bool JaxmgCudaMemoryDebugEnabled() {
  return JaxmgCudaDiagnosticsEnabled() || TruthyEnv("JAXMG_CUDA_MEM_DEBUG");
}

bool JaxmgCudaTimingDebugEnabled() {
  return JaxmgCudaDiagnosticsEnabled() || TruthyEnv("JAXMG_CUDA_TIMING_DEBUG");
}

void JaxmgCudaMemoryCheckpoint(int rank, const char* label) {
  if (!JaxmgCudaMemoryDebugEnabled()) {
    return;
  }

  int device = -1;
  cudaError_t device_status = cudaGetDevice(&device);
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  cudaError_t mem_status = cudaMemGetInfo(&free_bytes, &total_bytes);
  const double gib = 1024.0 * 1024.0 * 1024.0;
  if (device_status != cudaSuccess || mem_status != cudaSuccess) {
    std::fprintf(stderr,
                 "JAXMG_CUDA_MEM_DEBUG rank=%d label=%s "
                 "cudaGetDevice=%d cudaMemGetInfo=%d\n",
                 rank, label, static_cast<int>(device_status),
                 static_cast<int>(mem_status));
    std::fflush(stderr);
    return;
  }

  const size_t used_bytes = total_bytes - free_bytes;
  std::fprintf(stderr,
               "JAXMG_CUDA_MEM_DEBUG rank=%d device=%d label=%s "
               "free_bytes=%zu used_bytes=%zu total_bytes=%zu "
               "free_gib=%.3f used_gib=%.3f total_gib=%.3f\n",
               rank, device, label, free_bytes, used_bytes, total_bytes,
               free_bytes / gib, used_bytes / gib, total_bytes / gib);
  std::fflush(stderr);
}

void JaxmgCudaSizeRecord(int rank, const char* label, size_t bytes) {
  if (!JaxmgCudaMemoryDebugEnabled()) {
    return;
  }

  const double gib = 1024.0 * 1024.0 * 1024.0;
  std::fprintf(stderr,
               "JAXMG_CUDA_MEM_DEBUG rank=%d label=%s bytes=%zu gib=%.3f\n",
               rank, label, bytes, bytes / gib);
  std::fflush(stderr);
}

JaxmgCudaStageTimer::JaxmgCudaStageTimer(int rank, const char* label,
                                         cudaStream_t stream)
    : rank_(rank),
      label_(label),
      stream_(stream),
      start_(nullptr),
      stop_(nullptr),
      enabled_(JaxmgCudaTimingDebugEnabled() && stream != nullptr),
      stopped_(false) {
  if (!enabled_) {
    return;
  }

  cudaError_t status = cudaEventCreate(&start_);
  if (status != cudaSuccess) {
    enabled_ = false;
    return;
  }
  status = cudaEventCreate(&stop_);
  if (status != cudaSuccess) {
    cudaEventDestroy(start_);
    start_ = nullptr;
    enabled_ = false;
    return;
  }
  status = cudaEventRecord(start_, stream_);
  if (status != cudaSuccess) {
    cudaEventDestroy(stop_);
    cudaEventDestroy(start_);
    stop_ = nullptr;
    start_ = nullptr;
    enabled_ = false;
  }
}

JaxmgCudaStageTimer::~JaxmgCudaStageTimer() { Stop(); }

void JaxmgCudaStageTimer::Stop() {
  if (!enabled_ || stopped_) {
    return;
  }
  stopped_ = true;

  cudaError_t status = cudaEventRecord(stop_, stream_);
  if (status == cudaSuccess) {
    status = cudaEventSynchronize(stop_);
  }

  float elapsed_ms = -1.0f;
  if (status == cudaSuccess) {
    status = cudaEventElapsedTime(&elapsed_ms, start_, stop_);
  }

  if (status == cudaSuccess) {
    std::fprintf(stderr,
                 "JAXMG_CUDA_TIMING_DEBUG rank=%d label=%s "
                 "elapsed_ms=%.6f\n",
                 rank_, label_, elapsed_ms);
  } else {
    std::fprintf(stderr,
                 "JAXMG_CUDA_TIMING_DEBUG rank=%d label=%s cuda_status=%d\n",
                 rank_, label_, static_cast<int>(status));
  }
  std::fflush(stderr);

  cudaEventDestroy(stop_);
  cudaEventDestroy(start_);
  stop_ = nullptr;
  start_ = nullptr;
}

}  // namespace xla::gpu
