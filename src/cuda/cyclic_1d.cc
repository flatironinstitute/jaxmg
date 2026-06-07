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
// 1D block-cyclic redistribution over the XLA-owned GPU communicator.
//
// This is the XLA communicator replacement for the previous CUDA peer/shared
// memory reshuffler. The planning logic uses the same slot and permutation
// cycle model: construct the global source->target column map, decompose it
// into open chains or closed cycles, and execute each move with one
// column-sized scratch slot when a closed cycle needs staging.

#include "include/xla_comm_backend.h"

namespace xla::gpu {

absl::Status XlaCommMatrixColumnStepPrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests) {
  if (collective_params == nullptr || clique_requests == nullptr) {
    return absl::InvalidArgumentError(
        "xla_comm_matrix_column_step requires XLA collective prepare contexts");
  }

  absl::StatusOr<GpuCliqueKey> clique_key =
      LocalDevicesP2PCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    return clique_key.status();
  }

  std::vector<GlobalDeviceId> device_group =
      LocalGlobalDeviceGroup(*collective_params);
  return clique_requests->RequestClique(*clique_key, {device_group});
}

absl::Status XlaCommMatrixColumnStepDispatch(
    se::Stream* stream, se::Stream* comm_stream, int64_t kind,
    int64_t source_rank_attr, int64_t target_rank_attr, int64_t source_col,
    int64_t target_col, ffi::AnyBuffer matrix, ffi::AnyBuffer scratch,
    ffi::Result<ffi::AnyBuffer> matrix_out,
    ffi::Result<ffi::AnyBuffer> scratch_out,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  // Diagnostic single-step API:
  //   kind == 0: move matrix[source_rank, source_col] ->
  //              matrix[target_rank, target_col]
  //   kind == 1: save matrix[source_rank, source_col] -> scratch[source_rank]
  //   kind == 2: restore scratch[source_rank] -> matrix[target_rank, target_col]
  // The fused solvers do not call this one-step interface directly; it is kept
  // for isolated tests and for comparing individual planned moves.
  // The dispatch flow is:
  //   1. Validate all FFI contexts and buffer contracts.
  //   2. Resolve this invocation's XLA rank and communicator.
  //   3. Branch on kind to select save/move/restore addressing.
  //   4. Use a local D2D copy for same-rank moves, otherwise CollectivePermute.
  if (stream == nullptr || comm_stream == nullptr) {
    return absl::InvalidArgumentError(
        "xla_comm_matrix_column_step requires XLA stream contexts");
  }
  if (collective_params == nullptr || collective_cliques == nullptr) {
    return absl::InvalidArgumentError(
        "xla_comm_matrix_column_step requires XLA collective contexts");
  }
  if (matrix.dimensions().size() != 2 ||
      matrix_out->dimensions().size() != 2 ||
      matrix.dimensions()[0] != matrix_out->dimensions()[0] ||
      matrix.dimensions()[1] != matrix_out->dimensions()[1]) {
    return absl::InvalidArgumentError(
        "xla_comm_matrix_column_step expects matching rank-2 matrix input/output");
  }
  if (scratch.dimensions().size() != 1 ||
      scratch_out->dimensions().size() != 1 ||
      scratch.dimensions()[0] != scratch_out->dimensions()[0]) {
    return absl::InvalidArgumentError(
        "xla_comm_matrix_column_step expects matching rank-1 scratch input/output");
  }
  if (matrix.element_type() != matrix_out->element_type() ||
      matrix.element_type() != scratch.element_type() ||
      scratch.element_type() != scratch_out->element_type()) {
    return absl::InvalidArgumentError(
        "xla_comm_matrix_column_step requires matrix and scratch dtypes to match");
  }

  const int64_t local_slots = matrix.dimensions()[0];
  const int64_t column_elements = matrix.dimensions()[1];
  if (local_slots <= 0 || column_elements <= 0) {
    return absl::InvalidArgumentError(
        "xla_comm_matrix_column_step requires non-empty matrix dimensions");
  }
  if (scratch.dimensions()[0] != column_elements) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "xla_comm_matrix_column_step scratch length %d must equal matrix "
        "column length %d",
        scratch.dimensions()[0], column_elements));
  }

  absl::StatusOr<GpuCliqueKey> clique_key =
      LocalDevicesP2PCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    return clique_key.status();
  }

  const int64_t num_ranks = static_cast<int64_t>(clique_key->num_devices());
  if (num_ranks <= 0) {
    return absl::InvalidArgumentError(
        "xla_comm_matrix_column_step requires a non-empty clique");
  }
  if (source_rank_attr < -1 || source_rank_attr >= num_ranks ||
      target_rank_attr < -1 || target_rank_attr >= num_ranks) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "xla_comm_matrix_column_step source/target ranks must be in [-1, %d), "
        "got %d/%d",
        num_ranks, source_rank_attr, target_rank_attr));
  }

  std::optional<RankId> rank =
      clique_key->rank(collective_params->global_device_id);
  if (!rank.has_value()) {
    return absl::InvalidArgumentError(
        "xla_comm_matrix_column_step could not resolve this device rank");
  }

  absl::StatusOr<Communicator*> comm = collective_cliques->GetComm(
      *clique_key, collective_params->global_device_id);
  if (!comm.ok()) {
    return comm.status();
  }

  const size_t element_bytes =
      matrix.size_bytes() / static_cast<size_t>(matrix.element_count());
  const uint64_t column_bytes =
      static_cast<uint64_t>(column_elements) * element_bytes;
  se::DeviceAddressBase matrix_base = matrix.device_memory();
  se::DeviceAddressBase matrix_out_base = matrix_out->device_memory();
  se::DeviceAddressBase scratch_base = scratch.device_memory();
  se::DeviceAddressBase scratch_out_base = scratch_out->device_memory();

  const int64_t rank_value = static_cast<int64_t>(rank->value());
  auto matrix_column = [&](se::DeviceAddressBase base,
                           int64_t col) -> se::DeviceAddressBase {
    return base.GetByteSlice(static_cast<uint64_t>(col) * column_bytes,
                             column_bytes);
  };

  if (kind == 1) {
    // Save-to-scratch is the first step of a closed permutation cycle. Only the
    // source rank has useful work to do; all other ranks take the no-op exit so
    // the SPMD FFI call remains structurally identical on every device.
    if (source_rank_attr < 0 || source_col < 0 || source_col >= local_slots) {
      return absl::InvalidArgumentError(
          "xla_comm_matrix_column_step save_scratch requires a valid source");
    }
    if (rank_value != source_rank_attr) {
      return absl::OkStatus();
    }
    se::DeviceAddressBase src_addr = matrix_column(matrix_base, source_col);
    se::DeviceAddressBase dst_addr =
        scratch_out_base.GetByteSlice(0, column_bytes);
    return stream->MemcpyD2D(&dst_addr, src_addr, column_bytes);
  }

  // From here on, the operation is either a normal matrix-column move or a
  // scratch restore. Both need a valid target rank/column. Only normal moves
  // need a matrix source column; restores source their payload from scratch.
  if (kind != 0 && kind != 2) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "xla_comm_matrix_column_step kind must be 0(move), 1(save_scratch), "
        "or 2(restore_scratch), got %d",
        kind));
  }
  if (source_rank_attr < 0 || target_rank_attr < 0) {
    return absl::InvalidArgumentError(
        "xla_comm_matrix_column_step move/restore requires source and target ranks");
  }
  if (target_col < 0 || target_col >= local_slots) {
    return absl::InvalidArgumentError(
        "xla_comm_matrix_column_step requires a valid target column");
  }
  if (kind == 0 && (source_col < 0 || source_col >= local_slots)) {
    return absl::InvalidArgumentError(
        "xla_comm_matrix_column_step move requires a valid source column");
  }

  se::DeviceAddressBase send_addr;
  if (kind == 0) {
    send_addr = matrix_column(matrix_base, source_col);
  } else {
    send_addr = scratch_base.GetByteSlice(0, column_bytes);
  }
  se::DeviceAddressBase recv_addr = matrix_column(matrix_out_base, target_col);

  if (source_rank_attr == target_rank_attr) {
    // Intra-rank movement never needs the collective communicator. Keeping
    // these copies local avoids unnecessary NCCL/XLA scheduling overhead.
    if (rank_value != source_rank_attr) {
      return absl::OkStatus();
    }
    return stream->MemcpyD2D(&recv_addr, send_addr, column_bytes);
  }

  // Cross-rank moves are represented in XLA as "maybe receive from one source"
  // plus "maybe send to one target". Non-participating ranks pass empty source
  // and target sets to the no-op path below.
  std::optional<RankId> source_rank;
  if (rank_value == target_rank_attr) {
    source_rank = RankId(source_rank_attr);
  }

  std::vector<RankId> target_ranks;
  if (rank_value == source_rank_attr) {
    target_ranks.push_back(RankId(target_rank_attr));
  }

  if (!source_rank.has_value() && target_ranks.empty()) {
    // Every rank executes the same FFI call, but only the sender and receiver
    // participate in each point-to-point CollectivePermute step.
    return absl::OkStatus();
  }

  absl::Status status = comm_stream->WaitFor(stream);
  if (!status.ok()) {
    return status;
  }

  Future<> future = (*comm)->CollectivePermute(
      send_addr, recv_addr, matrix.element_type(),
      static_cast<size_t>(column_elements), source_rank, target_ranks,
      GpuCollectives::On(*comm_stream));
  status = future.Await();
  if (!status.ok()) {
    return status;
  }

  return stream->WaitFor(comm_stream);
}

absl::Status XlaCommMatrixColumnBatchPrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests) {
  if (collective_params == nullptr || clique_requests == nullptr) {
    return absl::InvalidArgumentError(
        "xla_comm_matrix_column_batch requires XLA collective prepare contexts");
  }

  absl::StatusOr<GpuCliqueKey> clique_key =
      LocalDevicesP2PCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    return clique_key.status();
  }

  std::vector<GlobalDeviceId> device_group =
      LocalGlobalDeviceGroup(*collective_params);
  return clique_requests->RequestClique(*clique_key, {device_group});
}

absl::Status ExecuteMatrixColumnTransferStep(
    const char* caller, int64_t step, se::Stream* stream,
    se::Stream* comm_stream, Communicator* comm, ffi::AnyBuffer matrix,
    se::DeviceAddressBase matrix_base, se::DeviceAddressBase matrix_out_base,
    se::DeviceAddressBase scratch_base, se::DeviceAddressBase scratch_out_base,
    int64_t local_slots, int64_t column_elements, uint64_t column_bytes,
    int64_t local_scratch_slots, int64_t num_ranks, int64_t rank_value,
    int64_t kind, int64_t source_rank_attr, int64_t target_rank_attr,
    int64_t source_col, int64_t target_col, int64_t scratch_slot) {
  // Execute one planned column move. Same-rank moves remain ordinary
  // device-to-device copies on the XLA stream; cross-rank moves use XLA's
  // communicator through CollectivePermute on the communication stream.
  // This helper is the execution backend for both open chains and closed
  // cycles. It deliberately mirrors the diagnostic one-step handler's branch
  // structure so test failures can be mapped back to the lower-level API.
  if (source_rank_attr < -1 || source_rank_attr >= num_ranks ||
      target_rank_attr < -1 || target_rank_attr >= num_ranks) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s step %d source/target ranks must be in [-1, %d), got %d/%d",
        caller, step, num_ranks, source_rank_attr, target_rank_attr));
  }

  auto matrix_column = [&](se::DeviceAddressBase base,
                           int64_t col) -> se::DeviceAddressBase {
    return base.GetByteSlice(static_cast<uint64_t>(col) * column_bytes,
                             column_bytes);
  };
  auto scratch_column = [&](se::DeviceAddressBase base,
                            int64_t slot) -> se::DeviceAddressBase {
    return base.GetByteSlice(static_cast<uint64_t>(slot) * column_bytes,
                             column_bytes);
  };

  if (kind == 1) {
    // Closed-cycle staging. One rank copies one matrix column into its scratch
    // slot, creating the free destination needed to rotate the rest of the
    // cycle without losing data.
    if (source_rank_attr < 0 || source_col < 0 || source_col >= local_slots ||
        scratch_slot < 0 || scratch_slot >= local_scratch_slots) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "%s save_scratch step %d has invalid source/scratch coordinates",
          caller, step));
    }
    if (rank_value == source_rank_attr) {
      se::DeviceAddressBase src_addr = matrix_column(matrix_base, source_col);
      se::DeviceAddressBase dst_addr =
          scratch_column(scratch_out_base, scratch_slot);
      return stream->MemcpyD2D(&dst_addr, src_addr, column_bytes);
    }
    return absl::OkStatus();
  }

  // Normal moves and scratch restores share the same destination checks. The
  // source checks differ because restore_scratch reads from the scratch arena
  // rather than from a matrix column.
  if (kind != 0 && kind != 2) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s step %d kind must be 0(move), 1(save_scratch), or "
        "2(restore_scratch), got %d",
        caller, step, kind));
  }
  if (source_rank_attr < 0 || target_rank_attr < 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s step %d move/restore requires source and target ranks", caller,
        step));
  }
  if (target_col < 0 || target_col >= local_slots) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s step %d requires a valid target column", caller, step));
  }
  if (kind == 0 && (source_col < 0 || source_col >= local_slots)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s move step %d requires a valid source column", caller, step));
  }
  if (kind == 2 && (scratch_slot < 0 || scratch_slot >= local_scratch_slots)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s restore_scratch step %d requires a valid scratch slot", caller,
        step));
  }

  se::DeviceAddressBase send_addr =
      kind == 0 ? matrix_column(matrix_base, source_col)
                : scratch_column(scratch_base, scratch_slot);
  se::DeviceAddressBase recv_addr = matrix_column(matrix_out_base, target_col);

  if (source_rank_attr == target_rank_attr) {
    // Local copy path. This is important for performance and correctness:
    // CollectivePermute is only needed when the source and destination slots
    // live on different ranks.
    if (rank_value != source_rank_attr) {
      return absl::OkStatus();
    }
    return stream->MemcpyD2D(&recv_addr, send_addr, column_bytes);
  }

  // CollectivePermute is asymmetric from each rank's point of view. The target
  // rank declares the source it expects to receive from; the source rank
  // declares the target it sends to; all other ranks are idle for this step.
  std::optional<RankId> source_rank;
  if (rank_value == target_rank_attr) {
    source_rank = RankId(source_rank_attr);
  }

  std::vector<RankId> target_rank_values;
  if (rank_value == source_rank_attr) {
    target_rank_values.push_back(RankId(target_rank_attr));
  }

  if (!source_rank.has_value() && target_rank_values.empty()) {
    return absl::OkStatus();
  }

  absl::Status status = comm_stream->WaitFor(stream);
  if (!status.ok()) {
    return status;
  }

  Future<> future = comm->CollectivePermute(
      send_addr, recv_addr, matrix.element_type(),
      static_cast<size_t>(column_elements), source_rank, target_rank_values,
      GpuCollectives::On(*comm_stream));
  status = future.Await();
  if (!status.ok()) {
    return status;
  }

  return stream->WaitFor(comm_stream);
}

absl::Status XlaCommMatrixColumnBatchDispatch(
    se::Stream* stream, se::Stream* comm_stream,
    absl::Span<const int64_t> kinds,
    absl::Span<const int64_t> source_ranks,
    absl::Span<const int64_t> target_ranks,
    absl::Span<const int64_t> source_cols,
    absl::Span<const int64_t> target_cols,
    absl::Span<const int64_t> scratch_slots, ffi::AnyBuffer matrix,
    ffi::AnyBuffer scratch, ffi::Result<ffi::AnyBuffer> matrix_out,
    ffi::Result<ffi::AnyBuffer> scratch_out,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  // Batched diagnostic API. Python supplies the exact move list as static FFI
  // attributes. The production path avoids this Python scheduling overhead by
  // constructing the same schedule inside ExecuteMatrixColumnNativePlanRaw.
  if (stream == nullptr || comm_stream == nullptr) {
    return absl::InvalidArgumentError(
        "xla_comm_matrix_column_batch requires XLA stream contexts");
  }
  if (collective_params == nullptr || collective_cliques == nullptr) {
    return absl::InvalidArgumentError(
        "xla_comm_matrix_column_batch requires XLA collective contexts");
  }
  if (matrix.dimensions().size() != 2 ||
      matrix_out->dimensions().size() != 2 ||
      matrix.dimensions()[0] != matrix_out->dimensions()[0] ||
      matrix.dimensions()[1] != matrix_out->dimensions()[1]) {
    return absl::InvalidArgumentError(
        "xla_comm_matrix_column_batch expects matching rank-2 matrix input/output");
  }
  if (scratch.dimensions().size() != 1 ||
      scratch_out->dimensions().size() != 1 ||
      scratch.dimensions()[0] != scratch_out->dimensions()[0]) {
    return absl::InvalidArgumentError(
        "xla_comm_matrix_column_batch expects matching rank-1 scratch input/output");
  }
  if (matrix.element_type() != matrix_out->element_type() ||
      matrix.element_type() != scratch.element_type() ||
      scratch.element_type() != scratch_out->element_type()) {
    return absl::InvalidArgumentError(
        "xla_comm_matrix_column_batch requires matrix and scratch dtypes to match");
  }

  const size_t num_steps = kinds.size();
  // All schedule arrays are parallel arrays. A step is valid only if every
  // field has an entry at the same index.
  if (source_ranks.size() != num_steps || target_ranks.size() != num_steps ||
      source_cols.size() != num_steps || target_cols.size() != num_steps ||
      scratch_slots.size() != num_steps) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "xla_comm_matrix_column_batch attr lengths must match; got "
        "%d/%d/%d/%d/%d/%d",
        kinds.size(), source_ranks.size(), target_ranks.size(),
        source_cols.size(), target_cols.size(), scratch_slots.size()));
  }
  if (num_steps == 0) {
    // Empty schedules are allowed so tests and callers can exercise the FFI
    // path without forcing a data movement.
    return absl::OkStatus();
  }

  const int64_t local_slots = matrix.dimensions()[0];
  const int64_t column_elements = matrix.dimensions()[1];
  if (local_slots <= 0 || column_elements <= 0) {
    return absl::InvalidArgumentError(
        "xla_comm_matrix_column_batch requires non-empty matrix dimensions");
  }
  if (scratch.dimensions()[0] % column_elements != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "xla_comm_matrix_column_batch scratch length %d must be a multiple of "
        "matrix column length %d",
        scratch.dimensions()[0], column_elements));
  }
  const int64_t local_scratch_slots = scratch.dimensions()[0] / column_elements;
  if (local_scratch_slots <= 0) {
    return absl::InvalidArgumentError(
        "xla_comm_matrix_column_batch requires at least one scratch slot");
  }

  absl::StatusOr<GpuCliqueKey> clique_key =
      LocalDevicesP2PCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    return clique_key.status();
  }

  const int64_t num_ranks = static_cast<int64_t>(clique_key->num_devices());
  if (num_ranks <= 0) {
    return absl::InvalidArgumentError(
        "xla_comm_matrix_column_batch requires a non-empty clique");
  }

  std::optional<RankId> rank =
      clique_key->rank(collective_params->global_device_id);
  if (!rank.has_value()) {
    return absl::InvalidArgumentError(
        "xla_comm_matrix_column_batch could not resolve this device rank");
  }

  absl::StatusOr<Communicator*> comm = collective_cliques->GetComm(
      *clique_key, collective_params->global_device_id);
  if (!comm.ok()) {
    return comm.status();
  }

  const size_t element_bytes =
      matrix.size_bytes() / static_cast<size_t>(matrix.element_count());
  const uint64_t column_bytes =
      static_cast<uint64_t>(column_elements) * element_bytes;
  se::DeviceAddressBase matrix_base = matrix.device_memory();
  se::DeviceAddressBase matrix_out_base = matrix_out->device_memory();
  se::DeviceAddressBase scratch_base = scratch.device_memory();
  se::DeviceAddressBase scratch_out_base = scratch_out->device_memory();

  const int64_t rank_value = static_cast<int64_t>(rank->value());

  for (size_t step = 0; step < num_steps; ++step) {
    // Execute in the caller-provided order. The schedule is already dependency
    // ordered by the cycle planner, so this loop intentionally does not try to
    // reorder or parallelize steps.
    absl::Status status = ExecuteMatrixColumnTransferStep(
        "xla_comm_matrix_column_batch", static_cast<int64_t>(step), stream,
        comm_stream, *comm, matrix, matrix_base, matrix_out_base, scratch_base,
        scratch_out_base, local_slots, column_elements, column_bytes,
        local_scratch_slots, num_ranks, rank_value, kinds[step],
        source_ranks[step], target_ranks[step], source_cols[step],
        target_cols[step], scratch_slots[step]);
    if (!status.ok()) {
      return status;
    }
  }

  return absl::OkStatus();
}

static std::pair<int64_t, int64_t> SlotToRankLocal(int64_t slot,
                                                   int64_t local_slots) {
  // A global slot is a logical column location in rank-major order. Converting
  // through slots lets the permutation planner ignore physical device pointers
  // until the final transfer step.
  return {slot / local_slots, slot % local_slots};
}

absl::Status XlaCommMatrixColumnNativePlanDispatch(
    se::Stream* stream, se::Stream* comm_stream, int64_t tile_size,
    ffi::AnyBuffer matrix, ffi::AnyBuffer scratch,
    ffi::Result<ffi::AnyBuffer> matrix_out,
    ffi::Result<ffi::AnyBuffer> scratch_out,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  if (stream == nullptr || comm_stream == nullptr) {
    return absl::InvalidArgumentError(
        "xla_comm_matrix_column_native_plan requires XLA stream contexts");
  }
  if (collective_params == nullptr || collective_cliques == nullptr) {
    return absl::InvalidArgumentError(
        "xla_comm_matrix_column_native_plan requires XLA collective contexts");
  }
  if (tile_size <= 0) {
    return absl::InvalidArgumentError(
        "xla_comm_matrix_column_native_plan requires positive tile_size");
  }
  if (matrix.dimensions().size() != 2 ||
      matrix_out->dimensions().size() != 2 ||
      matrix.dimensions()[0] != matrix_out->dimensions()[0] ||
      matrix.dimensions()[1] != matrix_out->dimensions()[1]) {
    return absl::InvalidArgumentError(
        "xla_comm_matrix_column_native_plan expects matching rank-2 matrix "
        "input/output");
  }
  if (scratch.dimensions().size() != 1 ||
      scratch_out->dimensions().size() != 1 ||
      scratch.dimensions()[0] != scratch_out->dimensions()[0]) {
    return absl::InvalidArgumentError(
        "xla_comm_matrix_column_native_plan expects matching rank-1 scratch "
        "input/output");
  }
  if (matrix.element_type() != matrix_out->element_type() ||
      matrix.element_type() != scratch.element_type() ||
      scratch.element_type() != scratch_out->element_type()) {
    return absl::InvalidArgumentError(
        "xla_comm_matrix_column_native_plan requires matrix and scratch dtypes "
        "to match");
  }

  const int64_t local_slots = matrix.dimensions()[0];
  const int64_t column_elements = matrix.dimensions()[1];
  if (local_slots <= 0 || column_elements <= 0) {
    return absl::InvalidArgumentError(
        "xla_comm_matrix_column_native_plan requires non-empty matrix "
        "dimensions");
  }
  if (scratch.dimensions()[0] % column_elements != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "xla_comm_matrix_column_native_plan scratch length %d must be a "
        "multiple of matrix column length %d",
        scratch.dimensions()[0], column_elements));
  }
  const int64_t local_scratch_slots = scratch.dimensions()[0] / column_elements;
  if (local_scratch_slots <= 0) {
    return absl::InvalidArgumentError(
        "xla_comm_matrix_column_native_plan requires at least one scratch slot");
  }

  absl::StatusOr<GpuCliqueKey> clique_key =
      LocalDevicesP2PCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    return clique_key.status();
  }
  const int64_t num_ranks = static_cast<int64_t>(clique_key->num_devices());
  if (num_ranks <= 0 || column_elements % num_ranks != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "xla_comm_matrix_column_native_plan requires matrix width %d to be "
        "divisible by rank count %d",
        column_elements, num_ranks));
  }
  const int64_t shard_size = column_elements / num_ranks;
  if (local_slots < shard_size) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "xla_comm_matrix_column_native_plan local slots %d must be at least "
        "the unpadded shard size %d",
        local_slots, shard_size));
  }

  std::optional<RankId> rank =
      clique_key->rank(collective_params->global_device_id);
  if (!rank.has_value()) {
    return absl::InvalidArgumentError(
        "xla_comm_matrix_column_native_plan could not resolve this device rank");
  }

  absl::StatusOr<Communicator*> comm = collective_cliques->GetComm(
      *clique_key, collective_params->global_device_id);
  if (!comm.ok()) {
    return comm.status();
  }

  const size_t element_bytes =
      matrix.size_bytes() / static_cast<size_t>(matrix.element_count());
  const uint64_t column_bytes =
      static_cast<uint64_t>(column_elements) * element_bytes;
  se::DeviceAddressBase matrix_base = matrix.device_memory();
  se::DeviceAddressBase matrix_out_base = matrix_out->device_memory();
  se::DeviceAddressBase scratch_base = scratch.device_memory();
  se::DeviceAddressBase scratch_out_base = scratch_out->device_memory();

  const int64_t rank_value = static_cast<int64_t>(rank->value());
  const int64_t effective_tile_size = std::min(tile_size, shard_size);
  const int64_t total_slots = local_slots * num_ranks;
  const int64_t padding_offset = local_slots - shard_size;
  const bool out_of_place = matrix_base.opaque() != matrix_out_base.opaque();
  // local_slots may include row padding. Logical source slots skip over that
  // padding by adding padding_offset at shard boundaries, while target slots
  // fill each rank's cyclic storage from the beginning.

  auto matrix_column = [&](se::DeviceAddressBase base,
                           int64_t col) -> se::DeviceAddressBase {
    return base.GetByteSlice(static_cast<uint64_t>(col) * column_bytes,
                             column_bytes);
  };

  std::vector<int64_t> dst_cols(num_ranks, 0);
  std::vector<int64_t> target_for_source(total_slots, -1);
  int64_t dst_rank = -1;
  // Build the same global column map as the previous CUDA peer shuffler:
  // target_for_source[src_slot] = dst_slot. Padding slots remain -1, which
  // terminates open chains.
  for (int64_t column = 0; column < column_elements; ++column) {
    if (column % effective_tile_size == 0) {
      dst_rank = (dst_rank + 1) % num_ranks;
    }
    const int64_t source_slot =
        column + padding_offset * (column / shard_size);
    const int64_t target_slot = dst_cols[dst_rank] + dst_rank * local_slots;
    if (source_slot < 0 || source_slot >= total_slots || target_slot < 0 ||
        target_slot >= total_slots) {
      return absl::InternalError(absl::StrFormat(
          "xla_comm_matrix_column_native_plan generated invalid slot %d -> %d",
          source_slot, target_slot));
    }
    target_for_source[source_slot] = target_slot;
    dst_cols[dst_rank] += 1;
  }

  std::vector<uint8_t> visited(total_slots, 0);
  std::map<int64_t, std::vector<int64_t>> cycles;
  // Follow the column map to construct disjoint chains/cycles. There are three
  // cases, matching the old CUDA helper:
  // 1. The chain reaches its start: closed cycle, use scratch for one slot.
  // 2. The chain reaches a slot already visited by an earlier chain: merge.
  // 3. The chain reaches padding (-1): open chain, no scratch restore.
  for (int64_t key = 0; key < total_slots; ++key) {
    int64_t target = target_for_source[key];
    if (target < 0 || visited[key]) {
      // Padding slots and slots already consumed by an earlier walk cannot
      // start a new chain.
      continue;
    }
    if (target == key) {
      // Fixed point. In-place reshuffles can ignore it. Out-of-place reshuffles
      // still need to copy the column from input to output.
      if (out_of_place) {
        auto [source_rank, source_col] = SlotToRankLocal(key, local_slots);
        if (rank_value == source_rank) {
          se::DeviceAddressBase src_addr = matrix_column(matrix_base, source_col);
          se::DeviceAddressBase dst_addr =
              matrix_column(matrix_out_base, source_col);
          absl::Status status =
              stream->MemcpyD2D(&dst_addr, src_addr, column_bytes);
          if (!status.ok()) {
            return status;
          }
        }
      }
      visited[key] = 1;
      continue;
    }

    std::vector<int64_t> cycle = {key};
    visited[key] = 1;
    while (true) {
      if (target < 0 || target >= total_slots) {
        // The planner should only generate valid slots or padding sentinels.
        // Anything else indicates a bug in the slot arithmetic above.
        return absl::InternalError(absl::StrFormat(
            "xla_comm_matrix_column_native_plan reached invalid target slot %d",
            target));
      }
      const int64_t next_target = target_for_source[target];
      if (next_target < 0) {
        // Open chain: the current target is real data's final destination and
        // the next hop is padding. Executing this tail-to-head needs no scratch
        // restore because padding is the free slot.
        cycle.push_back(target);
        break;
      }

      const bool dst_visited = visited[target] != 0;
      if (next_target == key) {
        // Closed cycle: following the map returns to the starting slot. Append
        // the start slot again so the execution phase can distinguish closed
        // cycles with slots.front() == slots.back().
        cycle.push_back(target);
        visited[target] = 1;
        cycle.push_back(next_target);
        break;
      }
      if (dst_visited) {
        // This walk reached a path discovered from another starting key. Merge
        // with that prior path when available, otherwise record the visited
        // target as the terminal point.
        auto prior = cycles.find(target);
        if (prior != cycles.end()) {
          cycle.insert(cycle.end(), prior->second.begin(), prior->second.end());
          cycles.erase(prior);
        } else {
          cycle.push_back(target);
        }
        break;
      }

      cycle.push_back(target);
      visited[target] = 1;
      target = next_target;
    }

    if (cycle.size() > 1) {
      cycles.emplace(key, std::move(cycle));
    }
  }

  int64_t step = 0;
  for (const auto& cycle_entry : cycles) {
    const std::vector<int64_t>& slots = cycle_entry.second;
    const bool is_closed = slots.size() > 1 && slots.front() == slots.back();
    if (is_closed) {
      // Closed-cycle execution has three phases:
      //   1. Save the final live slot into scratch.
      //   2. Move the remaining slots tail-to-head.
      //   3. Restore scratch into the first slot.
      // Closed cycles need one staged column. This is the same one-buffer
      // scheme used by the old 1D shuffler, with XLA CollectivePermute
      // replacing cudaMemcpyPeerAsync for cross-rank moves.
      const int64_t saved_slot = slots[slots.size() - 2];
      auto [source_rank, source_col] = SlotToRankLocal(saved_slot, local_slots);
      absl::Status status = ExecuteMatrixColumnTransferStep(
          "xla_comm_matrix_column_native_plan", step++, stream, comm_stream,
          *comm, matrix, matrix_base, matrix_out_base, scratch_base,
          scratch_out_base, local_slots, column_elements, column_bytes,
          local_scratch_slots, num_ranks, rank_value, 1, source_rank, -1,
          source_col, -1, 0);
      if (!status.ok()) {
        return status;
      }

      // Execute each chain backwards so every destination is written after its
      // previous contents have either moved onward or, for closed cycles, been
      // saved into the one-column scratch slot.
      for (int64_t index = static_cast<int64_t>(slots.size()) - 3; index >= 0;
           --index) {
        auto [move_source_rank, move_source_col] =
            SlotToRankLocal(slots[index], local_slots);
        auto [target_rank, target_col] =
            SlotToRankLocal(slots[index + 1], local_slots);
        status = ExecuteMatrixColumnTransferStep(
            "xla_comm_matrix_column_native_plan", step++, stream, comm_stream,
            *comm, matrix, matrix_base, matrix_out_base, scratch_base,
            scratch_out_base, local_slots, column_elements, column_bytes,
            local_scratch_slots, num_ranks, rank_value, 0, move_source_rank,
            target_rank, move_source_col, target_col, 0);
        if (!status.ok()) {
          return status;
        }
      }

      auto [target_rank, target_col] = SlotToRankLocal(slots[0], local_slots);
      status = ExecuteMatrixColumnTransferStep(
          "xla_comm_matrix_column_native_plan", step++, stream, comm_stream,
          *comm, matrix, matrix_base, matrix_out_base, scratch_base,
          scratch_out_base, local_slots, column_elements, column_bytes,
          local_scratch_slots, num_ranks, rank_value, 2, source_rank,
          target_rank, -1, target_col, 0);
      if (!status.ok()) {
        return status;
      }
      continue;
    }

    // Open chains end in padding, so tail-to-head movement is enough; nothing
    // has to be restored from scratch.
    // Example shape: A -> B -> C -> padding. We move C first, then B, then A,
    // so every destination has already been vacated when it is overwritten.
    for (int64_t index = static_cast<int64_t>(slots.size()) - 2; index >= 0;
         --index) {
      auto [source_rank, source_col] = SlotToRankLocal(slots[index], local_slots);
      auto [target_rank, target_col] =
          SlotToRankLocal(slots[index + 1], local_slots);
      absl::Status status = ExecuteMatrixColumnTransferStep(
          "xla_comm_matrix_column_native_plan", step++, stream, comm_stream,
          *comm, matrix, matrix_base, matrix_out_base, scratch_base,
          scratch_out_base, local_slots, column_elements, column_bytes,
          local_scratch_slots, num_ranks, rank_value, 0, source_rank,
          target_rank, source_col, target_col, 0);
      if (!status.ok()) {
        return status;
      }
    }
  }

  return absl::OkStatus();
}

absl::Status ExecuteMatrixColumnNativePlanRaw(
    const char* caller, se::Stream* stream, se::Stream* comm_stream,
    Communicator* comm, ffi::AnyBuffer matrix,
    se::DeviceAddressBase matrix_base, se::DeviceAddressBase matrix_out_base,
    se::DeviceAddressBase scratch_base, se::DeviceAddressBase scratch_out_base,
    int64_t local_slots, int64_t column_elements, uint64_t column_bytes,
    int64_t local_scratch_slots, int64_t num_ranks, int64_t rank_value,
    int64_t tile_size, bool reverse) {
  const int64_t shard_size = column_elements / num_ranks;
  const int64_t effective_tile_size = std::min(tile_size, shard_size);
  const int64_t total_slots = local_slots * num_ranks;
  const int64_t padding_offset = local_slots - shard_size;
  const bool out_of_place = matrix_base.opaque() != matrix_out_base.opaque();
  // This raw helper is the fair production replacement for the old C++ CUDA
  // peer planner: it computes the schedule and executes it in one native call.
  // The caller chooses direction with reverse=false for row-sharded -> cyclic
  // and reverse=true for cyclic -> row-sharded.

  auto matrix_column = [&](se::DeviceAddressBase base,
                           int64_t col) -> se::DeviceAddressBase {
    return base.GetByteSlice(static_cast<uint64_t>(col) * column_bytes,
                             column_bytes);
  };

  std::vector<int64_t> dst_cols(num_ranks, 0);
  std::vector<int64_t> target_for_source(total_slots, -1);
  int64_t dst_rank = -1;
  // Build the original row-sharded <-> cyclic slot map. For the reverse pass,
  // invert the same source->target map so a cuSolverMg cyclic buffer is
  // returned to the original row-sharded JAX layout.
  for (int64_t column = 0; column < column_elements; ++column) {
    if (column % effective_tile_size == 0) {
      dst_rank = (dst_rank + 1) % num_ranks;
    }
    const int64_t source_slot =
        column + padding_offset * (column / shard_size);
    const int64_t target_slot = dst_cols[dst_rank] + dst_rank * local_slots;
    if (source_slot < 0 || source_slot >= total_slots || target_slot < 0 ||
        target_slot >= total_slots) {
      return absl::InternalError(absl::StrFormat(
          "%s generated invalid slot %d -> %d", caller, source_slot,
          target_slot));
    }
    if (reverse) {
      target_for_source[target_slot] = source_slot;
    } else {
      target_for_source[source_slot] = target_slot;
    }
    dst_cols[dst_rank] += 1;
  }

  std::vector<uint8_t> visited(total_slots, 0);
  std::map<int64_t, std::vector<int64_t>> cycles;
  // Decompose the map into the same closed-cycle/open-chain representation as
  // the CUDA peer backend. Padding slots are valid chain ends and therefore do
  // not require scratch restoration.
  for (int64_t key = 0; key < total_slots; ++key) {
    int64_t target = target_for_source[key];
    if (target < 0 || visited[key]) {
      // Padding slots and slots already consumed by an earlier walk cannot
      // start a new chain.
      continue;
    }
    if (target == key) {
      // Fixed point. In-place reshuffles can ignore it. Out-of-place reshuffles
      // still need to copy the column from input to output.
      if (out_of_place) {
        auto [source_rank, source_col] = SlotToRankLocal(key, local_slots);
        if (rank_value == source_rank) {
          se::DeviceAddressBase src_addr = matrix_column(matrix_base, source_col);
          se::DeviceAddressBase dst_addr =
              matrix_column(matrix_out_base, source_col);
          absl::Status status =
              stream->MemcpyD2D(&dst_addr, src_addr, column_bytes);
          if (!status.ok()) {
            return status;
          }
        }
      }
      visited[key] = 1;
      continue;
    }

    std::vector<int64_t> cycle = {key};
    visited[key] = 1;
    while (true) {
      if (target < 0 || target >= total_slots) {
        // The planner should only generate valid slots or padding sentinels.
        // Anything else indicates a bug in the slot arithmetic above.
        return absl::InternalError(
            absl::StrFormat("%s reached invalid target slot %d", caller,
                            target));
      }
      const int64_t next_target = target_for_source[target];
      if (next_target < 0) {
        // Open chain: the current target is real data's final destination and
        // the next hop is padding. Executing this tail-to-head needs no scratch
        // restore because padding is the free slot.
        cycle.push_back(target);
        break;
      }

      const bool dst_visited = visited[target] != 0;
      if (next_target == key) {
        // Closed cycle: following the map returns to the starting slot. Append
        // the start slot again so the execution phase can distinguish closed
        // cycles with slots.front() == slots.back().
        cycle.push_back(target);
        visited[target] = 1;
        cycle.push_back(next_target);
        break;
      }
      if (dst_visited) {
        // This walk reached a path discovered from another starting key. Merge
        // with that prior path when available, otherwise record the visited
        // target as the terminal point.
        auto prior = cycles.find(target);
        if (prior != cycles.end()) {
          cycle.insert(cycle.end(), prior->second.begin(), prior->second.end());
          cycles.erase(prior);
        } else {
          cycle.push_back(target);
        }
        break;
      }

      cycle.push_back(target);
      visited[target] = 1;
      target = next_target;
    }

    if (cycle.size() > 1) {
      cycles.emplace(key, std::move(cycle));
    }
  }

  int64_t step = 0;
  for (const auto& cycle_entry : cycles) {
    const std::vector<int64_t>& slots = cycle_entry.second;
    const bool is_closed = slots.size() > 1 && slots.front() == slots.back();
    if (is_closed) {
      // Closed-cycle execution has three phases:
      //   1. Save the final live slot into scratch.
      //   2. Move the remaining slots tail-to-head.
      //   3. Restore scratch into the first slot.
      // Save one column before rotating a closed cycle. Open chains end in
      // padding and can be executed tail-to-head without this staging step.
      const int64_t saved_slot = slots[slots.size() - 2];
      auto [source_rank, source_col] = SlotToRankLocal(saved_slot, local_slots);
      absl::Status status = ExecuteMatrixColumnTransferStep(
          caller, step++, stream, comm_stream, comm, matrix, matrix_base,
          matrix_out_base, scratch_base, scratch_out_base, local_slots,
          column_elements, column_bytes, local_scratch_slots, num_ranks,
          rank_value, 1, source_rank, -1, source_col, -1, 0);
      if (!status.ok()) {
        return status;
      }

      // Move tail-to-head to avoid overwriting data that still needs to move.
      for (int64_t index = static_cast<int64_t>(slots.size()) - 3; index >= 0;
           --index) {
        auto [move_source_rank, move_source_col] =
            SlotToRankLocal(slots[index], local_slots);
        auto [target_rank, target_col] =
            SlotToRankLocal(slots[index + 1], local_slots);
        status = ExecuteMatrixColumnTransferStep(
            caller, step++, stream, comm_stream, comm, matrix, matrix_base,
            matrix_out_base, scratch_base, scratch_out_base, local_slots,
            column_elements, column_bytes, local_scratch_slots, num_ranks,
            rank_value, 0, move_source_rank, target_rank, move_source_col,
            target_col, 0);
        if (!status.ok()) {
          return status;
        }
      }

      auto [target_rank, target_col] = SlotToRankLocal(slots[0], local_slots);
      status = ExecuteMatrixColumnTransferStep(
          caller, step++, stream, comm_stream, comm, matrix, matrix_base,
          matrix_out_base, scratch_base, scratch_out_base, local_slots,
          column_elements, column_bytes, local_scratch_slots, num_ranks,
          rank_value, 2, source_rank, target_rank, -1, target_col, 0);
      if (!status.ok()) {
        return status;
      }
      continue;
    }

    // Open chains end in padding, so tail-to-head movement is enough; nothing
    // has to be restored from scratch. Example shape: A -> B -> C -> padding.
    // We move C first, then B, then A, so every destination has already been
    // vacated when it is overwritten.
    for (int64_t index = static_cast<int64_t>(slots.size()) - 2; index >= 0;
         --index) {
      auto [source_rank, source_col] = SlotToRankLocal(slots[index], local_slots);
      auto [target_rank, target_col] =
          SlotToRankLocal(slots[index + 1], local_slots);
      absl::Status status = ExecuteMatrixColumnTransferStep(
          caller, step++, stream, comm_stream, comm, matrix, matrix_base,
          matrix_out_base, scratch_base, scratch_out_base, local_slots,
          column_elements, column_bytes, local_scratch_slots, num_ranks,
          rank_value, 0, source_rank, target_rank, source_col, target_col, 0);
      if (!status.ok()) {
        return status;
      }
    }
  }

  return absl::OkStatus();
}

}  // namespace xla::gpu
