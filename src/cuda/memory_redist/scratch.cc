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
// Native scratch sizing for the cuSOLVERMp redistribution path.
//
// The fused solver handlers use one CUDA-owned scratch allocation for every
// temporary movement inside a call:
//
//   1. local row-major <-> column-major layout conversion;
//   2. top-left edge-padding compaction and its inverse;
//   3. 2D block-cyclic tile-slab redistribution and its inverse.
//
// The 2D cyclic stage defines the fixed bound:
//
//   3 * max(tile_cols * local_rows, tile_rows * local_cols)
//
// The factor of three provides saved, send, and receive slots for a closed
// permutation cycle. Edge-padding is an open-chain move and uses the allocation
// as one larger payload. Layout conversion requires only a one-row or one-column
// minimum, which is covered when tile sizes and local dimensions are positive.

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "memory_redist.h"

namespace xla::gpu {

// Computes the scratch requirement for a single matrix moving through local
// layout conversion, edge-padding compaction, and 2D block-cyclic redistribution.
absl::StatusOr<int64_t> RequiredPadded2DRedistScratchElements(
    const Padded2DRedistScratchRequest& request) {
  // Reuse execution-plan validation so invalid redistribution geometry fails
  // before scratch allocation.
  absl::StatusOr<int64_t> elements =
      RequiredPadded2DNativePlanScratchElements(
          request.process_rows, request.process_cols, request.tile_rows,
          request.tile_cols, request.logical_rows, request.logical_cols,
          request.local_rows, request.local_cols, request.rank_map);
  if (!elements.ok()) {
    return elements.status();
  }

  // RequiredPadded2DNativePlanScratchElements already validates the padding
  // geometry and returns the 3-slot cyclic redistribution bound. This defensive
  // check documents the invariant that lets layout conversion reuse the same
  // scratch without requesting an extra max(local_rows, local_cols) term.
  const int64_t layout_minimum =
      std::max(request.local_rows, request.local_cols);
  if (*elements < layout_minimum) {
    return absl::InternalError(absl::StrFormat(
        "padded 2D redistribution scratch invariant failed: computed %d "
        "elements, but layout conversion requires at least %d",
        *elements, layout_minimum));
  }
  return *elements;
}

// Allocates one CUDA scratch buffer sized to the maximum requirement across all
// matrices participating in a fused solver call.
absl::StatusOr<Padded2DRedistScratch> AllocatePadded2DRedistScratch(
    cudaStream_t cuda_stream, size_t element_bytes,
    absl::Span<const Padded2DRedistScratchRequest> requests,
    const char* caller) {
  if (element_bytes == 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s requires a positive element size for redistribution scratch",
        caller));
  }
  if (requests.empty()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s requires at least one redistribution scratch request", caller));
  }

  int64_t max_elements = 0;
  for (const Padded2DRedistScratchRequest& request : requests) {
    // POTRS passes both A and B requests because their logical column counts
    // can differ.  SYEVD passes only A.  One allocation is sized to the maximum
    // across all requested fused-call buffers.
    absl::StatusOr<int64_t> elements =
        RequiredPadded2DRedistScratchElements(request);
    if (!elements.ok()) {
      return elements.status();
    }
    max_elements = std::max(max_elements, *elements);
  }

  const size_t bytes =
      std::max<size_t>(1, static_cast<size_t>(max_elements)) * element_bytes;
  void* scratch_pointer = nullptr;
  const cudaError_t cuda_status =
      cudaMallocAsync(&scratch_pointer, bytes, cuda_stream);
  if (cuda_status != cudaSuccess) {
    return absl::ResourceExhaustedError(absl::StrFormat(
        "Unable to allocate CUDA scratch memory for %s: requested %zu "
        "bytes, cuda status %d (%s)",
        caller, bytes, static_cast<int>(cuda_status),
        cudaGetErrorString(cuda_status)));
  }

  return Padded2DRedistScratch{
      /*base=*/se::DeviceAddressBase(scratch_pointer, bytes),
      /*pointer=*/scratch_pointer,
      /*elements=*/max_elements,
      /*bytes=*/bytes,
  };
}

// Releases CUDA scratch memory allocated for one fused redistribution call.
absl::Status FreePadded2DRedistScratch(cudaStream_t cuda_stream,
                                       Padded2DRedistScratch scratch,
                                       const char* caller) {
  void* pointer = scratch.pointer;
  if (pointer == nullptr || scratch.bytes == 0) {
    return absl::OkStatus();
  }
  const cudaError_t cuda_status = cudaFreeAsync(pointer, cuda_stream);
  if (cuda_status != cudaSuccess) {
    return absl::InternalError(absl::StrFormat(
        "Unable to free CUDA scratch memory for %s: cuda status %d (%s)",
        caller, static_cast<int>(cuda_status),
        cudaGetErrorString(cuda_status)));
  }
  return absl::OkStatus();
}

}  // namespace xla::gpu
