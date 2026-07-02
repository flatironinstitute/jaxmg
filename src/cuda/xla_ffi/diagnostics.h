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
// Optional native CUDA diagnostics for benchmark runs.
//
// Production solver calls should not pay for memory polling, event timing, or
// verbose logging.  This file provides environment-gated helpers that are
// cheap when disabled and shared by both memory_redist and cuSOLVERMp routines.
// Benchmark jobs can enable them to collect per-rank GPU memory checkpoints,
// requested allocation sizes, and CUDA-event stage timings without changing the
// public Python solver API.

#ifndef JAXMG_XLA_FFI_DIAGNOSTICS_H_
#define JAXMG_XLA_FFI_DIAGNOSTICS_H_

#include <cstddef>

#include <cuda_runtime_api.h>

namespace xla::gpu {

// Enables memory and timing diagnostics together.
bool JaxmgCudaDiagnosticsEnabled();

// Enables cudaMemGetInfo() checkpoints.  This is true when either
// JAXMG_CUDA_MEM_DEBUG or JAXMG_CUDA_DIAGNOSTICS is set to a truthy value.
bool JaxmgCudaMemoryDebugEnabled();

// Enables CUDA-event stage timing.  This is true when either
// JAXMG_CUDA_TIMING_DEBUG or JAXMG_CUDA_DIAGNOSTICS is set to a truthy value.
bool JaxmgCudaTimingDebugEnabled();

// Emits a cudaMemGetInfo() checkpoint for the active CUDA device.
void JaxmgCudaMemoryCheckpoint(int rank, const char* label);

// Emits a requested allocation/workspace size.
void JaxmgCudaSizeRecord(int rank, const char* label, size_t bytes);

// RAII CUDA-event timer used only in diagnostic benchmark mode.
//
// Construct this at the start of a stage.  Destruction records a stop event on
// the same stream, synchronizes that event, and emits one elapsed-time line.
// The synchronization perturbs timings slightly, which is why this helper is
// gated behind explicit benchmark/debug environment variables.
class JaxmgCudaStageTimer {
 public:
  JaxmgCudaStageTimer(int rank, const char* label, cudaStream_t stream);
  ~JaxmgCudaStageTimer();

  JaxmgCudaStageTimer(const JaxmgCudaStageTimer&) = delete;
  JaxmgCudaStageTimer& operator=(const JaxmgCudaStageTimer&) = delete;

  // Stops the timer before the end of the C++ scope.  Calling Stop more than
  // once is safe.
  void Stop();

 private:
  int rank_;
  const char* label_;
  cudaStream_t stream_;
  cudaEvent_t start_;
  cudaEvent_t stop_;
  bool enabled_;
  bool stopped_;
};

}  // namespace xla::gpu

#endif  // JAXMG_XLA_FFI_DIAGNOSTICS_H_
