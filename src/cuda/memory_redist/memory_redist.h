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
// Public internal interface for native 2D memory redistribution.
//
// This header owns the small value types and entry points used to convert padded
// JAX local shards into cuSOLVERMp's local 2D block-cyclic storage and back. The
// implementation is split across edge-padding planning, block-cyclic planning,
// scratch sizing, and rectangle transport.

#ifndef JAXMG_MEMORY_REDIST_H_
#define JAXMG_MEMORY_REDIST_H_

#include "../xla_ffi/runtime.h"

namespace xla::gpu {

// A rectangular view into one local GPU shard. All coordinates are logical
// local row/column indices, independent of the physical byte address ordering.
// The rectangle executors interpret these coordinates in column-major local
// storage after the fused solver handlers have performed layout conversion.
struct NativeLocalRect {
  int64_t row_start;
  int64_t col_start;
  int64_t row_count;
  int64_t col_count;
};

// Scheduler operation types used by the low-memory cyclic redistribution.
// kMove copies one source rectangle into one target rectangle.  kSaveScratch
// and kRestoreScratch are the two endpoints of a closed cycle: save one live
// rectangle, rotate the rest of the cycle, then restore the saved rectangle.
enum class Native2DStepKind : int64_t {
  kMove = 0,
  kSaveScratch = 1,
  kRestoreScratch = 2,
};

// One scheduled rectangle movement.  `phase` distinguishes horizontal/vertical
// redistribution phases, and `sequence` is the dependency wave within that
// phase. Steps with the same phase/sequence/kind may be batched only if their
// source and target ranks do not conflict.
struct Native2DStep {
  int64_t phase;
  int64_t sequence;
  Native2DStepKind kind;
  int64_t source_rank;
  int64_t target_rank;
  NativeLocalRect source;
  NativeLocalRect target;
};

// Conflict-free execution batch.  Batches are the unit passed to the rectangle
// transport layer: every rank can inspect a batch and decide whether it is a
// sender, receiver, both for a local move, or idle.
struct Native2DStepBatch {
  int64_t phase;
  Native2DStepKind kind;
  std::vector<Native2DStep> steps;
};

// Lightweight copy helpers used by production wrappers when XLA could not
// alias an input buffer with the requested output/work buffer.
absl::Status CopyMatrixIfNeeded(cudaStream_t cuda_stream,
                                ffi::AnyBuffer matrix,
                                ffi::Result<ffi::AnyBuffer> matrix_out);
absl::Status CopyScratchIfNeeded(cudaStream_t cuda_stream,
                                 ffi::AnyBuffer scratch,
                                 ffi::Result<ffi::AnyBuffer> scratch_out);
absl::Status ConvertRowMajorToColumnMajorInPlace(
    cudaStream_t cuda_stream, const char* caller, ffi::AnyBuffer matrix,
    se::DeviceAddressBase scratch_base, int64_t scratch_elements);
absl::Status ConvertColumnMajorToRowMajorInPlace(
    cudaStream_t cuda_stream, const char* caller, ffi::AnyBuffer matrix,
    se::DeviceAddressBase scratch_base, int64_t scratch_elements);

// Low-level rectangle transport helpers. These remain exposed only inside the
// native backend so tests can use the exact production pack/unpack/NCCL path
// without duplicating transport logic.
absl::Status ValidateRect(const char* caller, int64_t row_start,
                          int64_t col_start, int64_t row_count,
                          int64_t col_count, int64_t local_rows,
                          int64_t local_cols);
absl::Status PackRect(cudaStream_t cuda_stream, int64_t local_rows,
                      int64_t row_start, int64_t col_start,
                      int64_t row_count, int64_t col_count,
                      size_t element_bytes, se::DeviceAddressBase matrix_base,
                      se::DeviceAddressBase packed_base);
absl::Status UnpackRect(cudaStream_t cuda_stream, int64_t local_rows,
                        int64_t row_start, int64_t col_start,
                        int64_t row_count, int64_t col_count,
                        size_t element_bytes,
                        se::DeviceAddressBase packed_base,
                        se::DeviceAddressBase matrix_base);
absl::Status RunRawNcclSendRecv(
    const char* caller, se::Stream* stream, se::Stream* comm_stream,
    cudaStream_t cuda_stream, GpuCommunicator* comm, int64_t rank_value,
    int64_t num_ranks, se::DeviceAddressBase send_buffer,
    se::DeviceAddressBase recv_buffer, uint64_t byte_count,
    std::optional<RankId> source_rank, absl::Span<const RankId> target_ranks);

// Planning and execution entry points for the tile-aligned slab reshuffler.
// Build* functions produce an abstract schedule; Batch* groups independent
// steps; Execute* performs the pack/NCCL/unpack work against real buffers.
int64_t MaxStepElementCount(const std::vector<Native2DStep>& steps);
std::vector<Native2DStepBatch> BatchNative2DSteps(
    const std::vector<Native2DStep>& steps);
absl::Status ExecuteNative2DStepBatches(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    const std::vector<Native2DStepBatch>& batches, int64_t local_rows,
    int64_t local_cols, int64_t rank_value, int64_t num_ranks,
    size_t element_bytes, int64_t slot_elements, ffi::AnyBuffer matrix,
    se::DeviceAddressBase matrix_out_base,
    se::DeviceAddressBase scratch_out_base, GpuCommunicator* comm);
absl::Status ExecuteEdgePaddingBatches(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    const std::vector<Native2DStepBatch>& batches, int64_t local_rows,
    int64_t local_cols, int64_t rank_value, int64_t num_ranks,
    size_t element_bytes, int64_t scratch_elements, ffi::AnyBuffer matrix,
    se::DeviceAddressBase matrix_out_base,
    se::DeviceAddressBase scratch_base, GpuCommunicator* comm);

// Edge-padding compaction planners.  The forward planner moves real data to
// the global top-left edge-padded layout.  ReverseEdgePaddingSteps builds the
// inverse schedule for solver outputs.
absl::StatusOr<std::vector<Native2DStep>> BuildEdgePaddingNative2DSteps(
    int64_t process_rows, int64_t process_cols, int64_t tile_rows,
    int64_t tile_cols, int64_t logical_rows, int64_t logical_cols,
    int64_t local_rows, int64_t local_cols,
    int64_t padding_slot_elements, absl::Span<const int64_t> rank_map);
std::vector<Native2DStep> ReverseEdgePaddingSteps(
    const std::vector<Native2DStep>& forward_steps);

// Tile-slab 2D block-cyclic planners.  The schedule is separable: a column
// owner phase followed by a row owner phase for forward redistribution, with
// the order inverted for reverse redistribution.
absl::StatusOr<std::vector<Native2DStep>> BuildSlabNative2DSteps(
    int64_t process_rows, int64_t process_cols, int64_t tile_rows,
    int64_t tile_cols, int64_t local_rows, int64_t local_cols,
    absl::Span<const int64_t> rank_map, bool reverse = false);
absl::StatusOr<int64_t> RequiredPadded2DNativePlanScratchElements(
    int64_t process_rows, int64_t process_cols, int64_t tile_rows,
    int64_t tile_cols, int64_t logical_rows, int64_t logical_cols,
    int64_t local_rows, int64_t local_cols,
    absl::Span<const int64_t> rank_map);

// Scratch sizing for the padded 2D redistribution pipeline.
//
// The memory_redist layer owns this calculation because it is an implementation
// detail of the native layout conversion, edge-padding compaction, and 2D
// block-cyclic scheduler. Solver files should describe the redistribution
// requests they need, then receive one XLA scratch allocation that is large
// enough for all phases in the fused FFI call.
struct Padded2DRedistScratchRequest {
  int64_t process_rows;
  int64_t process_cols;
  int64_t tile_rows;
  int64_t tile_cols;
  int64_t logical_rows;
  int64_t logical_cols;
  int64_t local_rows;
  int64_t local_cols;
  absl::Span<const int64_t> rank_map;
};

struct Padded2DRedistScratch {
  se::DeviceAddressBase base;
  int64_t elements;
  size_t bytes;
};

absl::StatusOr<int64_t> RequiredPadded2DRedistScratchElements(
    const Padded2DRedistScratchRequest& request);
absl::StatusOr<Padded2DRedistScratch> AllocatePadded2DRedistScratch(
    se::ScratchAllocator& scratch, size_t element_bytes,
    absl::Span<const Padded2DRedistScratchRequest> requests,
    const char* caller);

// Production raw executor used inside fused solver handlers.  It assumes the
// caller already copied/aliased into the work buffer and supplied one scratch
// allocation sized by AllocatePadded2DRedistScratch.
absl::Status ExecutePadded2DNativePlanRaw(
    const char* caller, se::Stream* stream, se::Stream* comm_stream,
    cudaStream_t cuda_stream, int64_t process_rows, int64_t process_cols,
    int64_t tile_rows, int64_t tile_cols, int64_t logical_rows,
    int64_t logical_cols, int64_t reverse,
    absl::Span<const int64_t> rank_map, ffi::AnyBuffer matrix,
    se::DeviceAddressBase matrix_out_base,
    se::DeviceAddressBase scratch_base, int64_t scratch_elements,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques);

}  // namespace xla::gpu

#endif  // JAXMG_MEMORY_REDIST_H_
