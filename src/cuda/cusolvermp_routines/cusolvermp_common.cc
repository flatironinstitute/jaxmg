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
// Shared cuSOLVERMp grid, status, and CUDA utility implementation.
//
// This file implements the common mechanics required by POTRS, LU solve, and
// SYEVD. These include process-grid validation, local NUMROC capacity
// calculations, device status copies, and CUDA device selection for donated JAX
// buffers.

#include "cusolvermp_common.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace xla::gpu {

// Normalizes the integer attribute used by Python into the cuSOLVERMp enum
// consumed by cusolverMpCreateDeviceGrid.
cusolverMpGridMapping_t ToCusolverMpGridMapping(int64_t grid_mapping) {
  return static_cast<cusolverMpGridMapping_t>(grid_mapping);
}

// Rejects unsupported process-grid mapping values before cuSOLVERMp sees them.
absl::Status ValidateCusolverMpGridMapping(const char* caller,
                                           int64_t grid_mapping) {
  if (grid_mapping != kCusolverMpGridMappingColMajor &&
      grid_mapping != kCusolverMpGridMappingRowMajor) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s grid_mapping must be 0 (column-major) or 1 (row-major), got %d",
        caller, grid_mapping));
  }
  return absl::OkStatus();
}

// Ensures the JAX mesh rank map is a dense row-major or column-major map that
// cuSOLVERMp can represent directly with its grid mapping enum.
absl::Status ValidateStandardRankMapForGridMapping(
    const char* caller, absl::Span<const int64_t> rank_map,
    int64_t process_rows, int64_t process_cols, int64_t grid_mapping) {
  JAXMG_RETURN_IF_ERROR(ValidateCusolverMpGridMapping(caller, grid_mapping));
  const int64_t num_ranks = process_rows * process_cols;
  if (rank_map.size() != static_cast<size_t>(num_ranks)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s expected rank_map length %d, got %d", caller, num_ranks,
        rank_map.size()));
  }

  std::vector<bool> seen(num_ranks, false);
  for (int64_t process_rank = 0; process_rank < num_ranks; ++process_rank) {
    const int64_t communicator_rank = rank_map[process_rank];
    if (communicator_rank < 0 || communicator_rank >= num_ranks) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "%s rank_map[%d]=%d is outside [0, %d)", caller, process_rank,
          communicator_rank, num_ranks));
    }
    if (seen[communicator_rank]) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "%s rank_map contains duplicate communicator rank %d", caller,
          communicator_rank));
    }
    seen[communicator_rank] = true;

    const int64_t process_row = process_rank / process_cols;
    const int64_t process_col = process_rank % process_cols;
    const int64_t expected =
        grid_mapping == kCusolverMpGridMappingRowMajor
            ? process_rank
            : process_col * process_rows + process_row;
    if (communicator_rank != expected) {
      return absl::UnimplementedError(absl::StrFormat(
          "%s supports only rank maps matching the requested cuSOLVERMp "
          "grid_mapping; rank_map[%d]=%d, expected %d",
          caller, process_rank, communicator_rank, expected));
    }
  }
  return absl::OkStatus();
}

// Converts the communicator rank reported by NCCL into a cuSOLVERMp process
// coordinate under the selected grid mapping.
std::pair<int32_t, int32_t> ProcessCoordFromRank(int nccl_rank,
                                                 int64_t process_rows,
                                                 int64_t process_cols,
                                                 int64_t grid_mapping) {
  if (grid_mapping == kCusolverMpGridMappingRowMajor) {
    return {
        nccl_rank / static_cast<int32_t>(process_cols),
        nccl_rank % static_cast<int32_t>(process_cols),
    };
  }
  return {
      nccl_rank % static_cast<int32_t>(process_rows),
      nccl_rank / static_cast<int32_t>(process_rows),
  };
}

// Implements the standard block-cyclic local-count calculation for one process
// coordinate along a distributed matrix axis.
int64_t LocalNumroc(int64_t n, int64_t block, int32_t process,
                    int32_t process_count) {
  int64_t owned = 0;
  const int64_t tile_count = (n + block - 1) / block;
  for (int64_t tile = 0; tile < tile_count; ++tile) {
    if (tile % process_count != process) {
      continue;
    }
    const int64_t start = tile * block;
    owned += std::min(block, n - start);
  }
  return owned;
}

// Writes the POTRS per-rank status words into the rank-1 device output buffer.
absl::Status CopyPotrsStatusToDevice(
    se::Stream* stream, const std::array<int32_t, kPotrsStatusSize>& status,
    ffi::Result<ffi::BufferR1<S32>> out) {
  se::DeviceAddress<int32_t> dst = out->device_memory();
  return stream->MemcpyH2D(absl::MakeConstSpan(status), &dst);
}

// Writes the LU-solve per-rank status words into the rank-1 device output
// buffer.
absl::Status CopyLuSolveStatusToDevice(
    se::Stream* stream, const std::array<int32_t, kLuSolveStatusSize>& status,
    ffi::Result<ffi::BufferR1<S32>> out) {
  se::DeviceAddress<int32_t> dst = out->device_memory();
  return stream->MemcpyH2D(absl::MakeConstSpan(status), &dst);
}

// Writes the SYEVD per-rank status words into the rank-1 device output buffer.
absl::Status CopySyevdStatusToDevice(
    se::Stream* stream, const std::array<int32_t, kSyevdStatusSize>& status,
    ffi::Result<ffi::BufferR1<S32>> out) {
  se::DeviceAddress<int32_t> dst = out->device_memory();
  return stream->MemcpyH2D(absl::MakeConstSpan(status), &dst);
}

// Writes the GESVD per-rank status words into the rank-1 device output buffer.
absl::Status CopyGesvdStatusToDevice(
    se::Stream* stream, const std::array<int32_t, kGesvdStatusSize>& status,
    ffi::Result<ffi::BufferR1<S32>> out) {
  se::DeviceAddress<int32_t> dst = out->device_memory();
  return stream->MemcpyH2D(absl::MakeConstSpan(status), &dst);
}

// Converts byte counts to rounded-up KiB values that fit in the compact status
// schema returned to Python.
int32_t SizeToKiBForStatus(size_t bytes) {
  const size_t kib = (bytes + 1023) / 1024;
  if (kib > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return std::numeric_limits<int32_t>::max();
  }
  return static_cast<int32_t>(kib);
}

// Copies an input/work buffer only when XLA did not alias it with the output
// buffer requested by the fused solver handler.
absl::Status CopyAnyBufferToOutputIfNeeded(cudaStream_t cuda_stream,
                                           ffi::AnyBuffer input,
                                           ffi::Result<ffi::AnyBuffer> output) {
  if (input.untyped_data() == output->untyped_data()) {
    return absl::OkStatus();
  }
  if (input.size_bytes() != output->size_bytes()) {
    return absl::InvalidArgumentError(
        "cuSOLVERMp received mismatched input/output buffer sizes");
  }
  JAXMG_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
      output->untyped_data(), input.untyped_data(), input.size_bytes(),
      cudaMemcpyDeviceToDevice, cuda_stream));
  return absl::OkStatus();
}

// Reads the native debug switch once per call site without forcing logging
// overhead in normal runs.
bool CusolverMpDebugEnabled() {
  const char* value = std::getenv("JAXMG_CUSOLVERMP_DEBUG");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

// Prints a rank-scoped debug line for cuSOLVERMp failures that are otherwise
// difficult to inspect across multiple Slurm tasks.
void CusolverMpDebug(int rank, const char* format, ...) {
  if (!CusolverMpDebugEnabled()) {
    return;
  }
  std::fprintf(stderr, "JAXMG_CUSOLVERMP_DEBUG rank=%d ", rank);
  va_list args;
  va_start(args, format);
  std::vfprintf(stderr, format, args);
  va_end(args);
  std::fprintf(stderr, "\n");
  std::fflush(stderr);
}

// Uses CUDA pointer attributes to bind the current host thread to the GPU that
// owns the donated JAX buffer before creating cuSOLVERMp handles/descriptors.
absl::StatusOr<int> DeviceForCudaPointer(const void* ptr) {
  cudaPointerAttributes attrs;
  cudaError_t status = cudaPointerGetAttributes(&attrs, ptr);
  if (status != cudaSuccess) {
    return absl::InternalError(absl::StrFormat(
        "cudaPointerGetAttributes failed while resolving the cuSOLVERMp "
        "buffer device: %s",
        cudaGetErrorString(status)));
  }
  return attrs.device;
}

}  // namespace xla::gpu
