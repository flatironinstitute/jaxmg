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
// Declares the value types and entry points used to convert padded JAX shards
// into cuSOLVERMp's local 2D block-cyclic storage and back. Implementations are
// split across edge-padding planning, block-cyclic planning, scratch sizing, and
// rectangle transport.

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

// Copies a matrix into its work/output buffer only when XLA did not alias both
// values to the same device allocation.
absl::Status CopyMatrixIfNeeded(cudaStream_t cuda_stream,
                                ffi::AnyBuffer matrix,
                                ffi::Result<ffi::AnyBuffer> matrix_out);

// Copies scratch only when the requested output is a distinct allocation.
absl::Status CopyScratchIfNeeded(cudaStream_t cuda_stream,
                                 ffi::AnyBuffer scratch,
                                 ffi::Result<ffi::AnyBuffer> scratch_out);

// Converts one local JAX row-major shard to cuSOLVERMp column-major local
// storage in place using the caller-provided bounded scratch allocation.
absl::Status ConvertRowMajorToColumnMajorInPlace(
    cudaStream_t cuda_stream, const char* caller, ffi::AnyBuffer matrix,
    se::DeviceAddressBase scratch_base, int64_t scratch_elements);

// Applies the inverse local layout conversion so solver outputs become
// row-major JAX-facing shards again.
absl::Status ConvertColumnMajorToRowMajorInPlace(
    cudaStream_t cuda_stream, const char* caller, ffi::AnyBuffer matrix,
    se::DeviceAddressBase scratch_base, int64_t scratch_elements);

// Low-level rectangle validation, packing, and transport operations.
// Checks that a logical local rectangle lies inside a rank-local matrix.
absl::Status ValidateRect(const char* caller, int64_t row_start,
                          int64_t col_start, int64_t row_count,
                          int64_t col_count, int64_t local_rows,
                          int64_t local_cols);

// Packs a strided column-major rectangle into contiguous scratch memory.
absl::Status PackRect(cudaStream_t cuda_stream, int64_t local_rows,
                      int64_t row_start, int64_t col_start,
                      int64_t row_count, int64_t col_count,
                      size_t element_bytes, se::DeviceAddressBase matrix_base,
                      se::DeviceAddressBase packed_base);

// Unpacks one contiguous scratch payload into a strided column-major rectangle.
absl::Status UnpackRect(cudaStream_t cuda_stream, int64_t local_rows,
                        int64_t row_start, int64_t col_start,
                        int64_t row_count, int64_t col_count,
                        size_t element_bytes,
                        se::DeviceAddressBase packed_base,
                        se::DeviceAddressBase matrix_base);

// Performs one raw NCCL point-to-point send/receive round using the communicator
// borrowed from XLA's GpuCommunicator.
absl::Status RunRawNcclSendRecv(
    const char* caller, se::Stream* stream, se::Stream* comm_stream,
    cudaStream_t cuda_stream, GpuCommunicator* comm, int64_t rank_value,
    int64_t num_ranks, se::DeviceAddressBase send_buffer,
    se::DeviceAddressBase recv_buffer, uint64_t byte_count,
    std::optional<RankId> source_rank, absl::Span<const RankId> target_ranks);

// Planning and execution entry points for tile-aligned slab redistribution.
// Build* functions produce schedules, Batch* groups independent steps, and
// Execute* performs pack/NCCL/unpack operations against device buffers.
// Returns the largest rectangle payload in a planned 2D movement schedule.
int64_t MaxStepElementCount(const std::vector<Native2DStep>& steps);

// Groups same-sequence movement steps into conflict-free batches.
std::vector<Native2DStepBatch> BatchNative2DSteps(
    const std::vector<Native2DStep>& steps);

// Executes conflict-free batches for the cyclic 2D block redistribution phase
// using saved/send/receive scratch slots.
absl::Status ExecuteNative2DStepBatches(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    const std::vector<Native2DStepBatch>& batches, int64_t local_rows,
    int64_t local_cols, int64_t rank_value, int64_t num_ranks,
    size_t element_bytes, int64_t slot_elements, ffi::AnyBuffer matrix,
    se::DeviceAddressBase matrix_out_base,
    se::DeviceAddressBase scratch_out_base, GpuCommunicator* comm);

// Executes open-chain edge-padding batches using the whole scratch allocation
// as a single temporary payload.
absl::Status ExecuteEdgePaddingBatches(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    const std::vector<Native2DStepBatch>& batches, int64_t local_rows,
    int64_t local_cols, int64_t rank_value, int64_t num_ranks,
    size_t element_bytes, int64_t scratch_elements, ffi::AnyBuffer matrix,
    se::DeviceAddressBase matrix_out_base,
    se::DeviceAddressBase scratch_base, GpuCommunicator* comm);

// Builds the forward top-left edge-padding compaction schedule for one padded
// JAX-sharded matrix.
absl::StatusOr<std::vector<Native2DStep>> BuildEdgePaddingNative2DSteps(
    int64_t process_rows, int64_t process_cols, int64_t tile_rows,
    int64_t tile_cols, int64_t logical_rows, int64_t logical_cols,
    int64_t local_rows, int64_t local_cols,
    int64_t padding_slot_elements, absl::Span<const int64_t> rank_map);

// Builds the inverse edge-padding schedule used after a solver returns data in
// the compacted global layout.
std::vector<Native2DStep> ReverseEdgePaddingSteps(
    const std::vector<Native2DStep>& forward_steps);

// Builds the tile-slab 2D block-cyclic ownership permutation, or its inverse
// when reverse=true. The forward schedule applies a column-owner phase followed
// by a row-owner phase; the inverse applies them in reverse order.
absl::StatusOr<std::vector<Native2DStep>> BuildSlabNative2DSteps(
    int64_t process_rows, int64_t process_cols, int64_t tile_rows,
    int64_t tile_cols, int64_t local_rows, int64_t local_cols,
    absl::Span<const int64_t> rank_map, bool reverse = false);

// Computes the fixed 3-slot scratch bound required by the native padded 2D
// plan after validating edge-padding and slab schedules.
absl::StatusOr<int64_t> RequiredPadded2DNativePlanScratchElements(
    int64_t process_rows, int64_t process_cols, int64_t tile_rows,
    int64_t tile_cols, int64_t logical_rows, int64_t logical_cols,
    int64_t local_rows, int64_t local_cols,
    absl::Span<const int64_t> rank_map);

// Describes one matrix's redistribution geometry for shared scratch sizing.
// The largest request in a fused solver call determines the single allocation
// reused by layout conversion, edge-padding compaction, and cyclic movement.
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
  void* pointer;
  int64_t elements;
  size_t bytes;
};

// Computes the scratch requirement for one matrix participating in a fused
// padded 2D redistribution call.
absl::StatusOr<int64_t> RequiredPadded2DRedistScratchElements(
    const Padded2DRedistScratchRequest& request);

// Allocates one CUDA scratch buffer large enough for every matrix described by
// the fused solver's redistribution requests.
absl::StatusOr<Padded2DRedistScratch> AllocatePadded2DRedistScratch(
    cudaStream_t cuda_stream, size_t element_bytes,
    absl::Span<const Padded2DRedistScratchRequest> requests,
    const char* caller);

// Releases a scratch buffer allocated by AllocatePadded2DRedistScratch.
absl::Status FreePadded2DRedistScratch(cudaStream_t cuda_stream,
                                       Padded2DRedistScratch scratch,
                                       const char* caller);

// Runs forward or reverse edge-padding plus 2D block-cyclic redistribution on
// a matrix buffer using the XLA-owned communicator. The caller supplies a work
// buffer and scratch allocation sized by AllocatePadded2DRedistScratch.
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
