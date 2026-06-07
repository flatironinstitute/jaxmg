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
// Diagnostic XLA communicator FFI handlers for the cuSolverMg backend.
//
// These handlers are not on the production solver path. They verify that an
// FFI invocation can request an XLA collective clique, look up the XLA-owned
// GPU communicator, and execute small collective operations through that
// communicator.

#include "include/xla_comm_backend.h"

namespace xla::gpu {

// Every collective FFI handler has a prepare stage and an execute stage. The
// prepare stage requests the communicator clique that XLA must make available
// to the execute stage; failing to request the clique here means GetComm will
// not have a communicator to return later.
absl::Status XlaCommCollectiveProbePrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests) {
  if (collective_params == nullptr || clique_requests == nullptr) {
    return absl::InvalidArgumentError(
        "xla_comm_collective_probe requires XLA collective prepare contexts");
  }

  absl::StatusOr<GpuCliqueKey> clique_key =
      LocalDevicesCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    return clique_key.status();
  }

  std::vector<GlobalDeviceId> device_group =
      LocalGlobalDeviceGroup(*collective_params);
  return clique_requests->RequestClique(*clique_key, {device_group});
}

absl::Status XlaCommCollectiveProbeDispatch(
    se::Stream* stream, ffi::AnyBuffer token,
    ffi::Result<ffi::BufferR1<S32>> out,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  if (stream == nullptr) {
    return absl::InvalidArgumentError(
        "xla_comm_collective_probe requires an XLA stream context");
  }
  if (out->dimensions().size() != 1 || out->dimensions()[0] != 8) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "xla_comm_collective_probe expects output shape (8,), got rank %d",
        out->dimensions().size()));
  }

  std::array<int32_t, 8> probe = {
      0,   // status
      -1,  // reserved for parity with the CMake probe's cuda_device field
      -1,  // local_device_id
      -1,  // global_device_id
      -1,  // local_device_count
      0,   // has_platform_handle
      static_cast<int32_t>(token.size_bytes()),
      -1,  // reserved for parity with the CMake probe's cuda_device_count field
  };

  // Keep failures encoded in the device output instead of returning early.
  // This probe is often used while debugging registration/prepare failures, so
  // a small status vector is easier to inspect from Python than a failed FFI.
  if (collective_params == nullptr || collective_cliques == nullptr) {
    probe[0] = 1;
  } else {
    probe[2] = static_cast<int32_t>(collective_params->local_device_id.value());
    probe[3] =
        static_cast<int32_t>(collective_params->global_device_id.value());
    probe[4] = static_cast<int32_t>(collective_params->local_device_count);

    absl::StatusOr<GpuCliqueKey> clique_key =
        LocalDevicesCliqueKey(*collective_params);
    if (!clique_key.ok()) {
      probe[0] = 2;
    } else {
      absl::StatusOr<GpuCommunicator*> comm = collective_cliques->GetComm(
          *clique_key, collective_params->global_device_id);
      if (!comm.ok()) {
        probe[0] = 3;
      } else {
        probe[5] = (*comm)->platform_comm().handle != nullptr ? 1 : 0;
      }
    }
  }

  se::DeviceAddress<int32_t> dst = out->device_memory();
  return stream->MemcpyH2D(absl::MakeConstSpan(probe), &dst);
}

absl::Status XlaCommAllReduceProbePrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests) {
  if (collective_params == nullptr || clique_requests == nullptr) {
    return absl::InvalidArgumentError(
        "xla_comm_allreduce_probe requires XLA collective prepare contexts");
  }

  absl::StatusOr<GpuCliqueKey> clique_key =
      LocalDevicesCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    return clique_key.status();
  }

  std::vector<GlobalDeviceId> device_group =
      LocalGlobalDeviceGroup(*collective_params);
  return clique_requests->RequestClique(*clique_key, {device_group});
}

absl::Status XlaCommAllReduceProbeDispatch(
    se::Stream* stream, se::Stream* comm_stream, ffi::BufferR1<U32> src,
    ffi::Result<ffi::BufferR1<U32>> dst,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  if (stream == nullptr || comm_stream == nullptr) {
    return absl::InvalidArgumentError(
        "xla_comm_allreduce_probe requires XLA stream contexts");
  }
  if (collective_params == nullptr || collective_cliques == nullptr) {
    return absl::InvalidArgumentError(
        "xla_comm_allreduce_probe requires XLA collective contexts");
  }
  if (src.dimensions().size() != 1 || dst->dimensions().size() != 1 ||
      src.dimensions()[0] != dst->dimensions()[0]) {
    return absl::InvalidArgumentError(
        "xla_comm_allreduce_probe expects matching rank-1 input/output");
  }

  absl::StatusOr<GpuCliqueKey> clique_key =
      LocalDevicesCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    return clique_key.status();
  }

  absl::StatusOr<Communicator*> comm = collective_cliques->GetComm(
      *clique_key, collective_params->global_device_id);
  if (!comm.ok()) {
    return comm.status();
  }

  // XLA passes two streams: the main execution stream and a communication
  // stream. The explicit WaitFor calls preserve ordering without forcing a host
  // device-wide synchronization.
  absl::Status status = comm_stream->WaitFor(stream);
  if (!status.ok()) {
    return status;
  }

  Future<> future = (*comm)->AllReduce(
      src.device_memory(), dst->device_memory(), src.element_type(),
      src.element_count(), ReductionKind::SUM, GpuCollectives::On(*comm_stream));
  status = future.Await();
  if (!status.ok()) {
    return status;
  }

  return stream->WaitFor(comm_stream);
}

absl::Status XlaCommRingPermuteProbePrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests) {
  if (collective_params == nullptr || clique_requests == nullptr) {
    return absl::InvalidArgumentError(
        "xla_comm_ring_permute_probe requires XLA collective prepare contexts");
  }

  // CollectivePermute uses the point-to-point clique key. This mirrors the
  // production reshuffler, which uses CollectivePermute for column movement.
  absl::StatusOr<GpuCliqueKey> clique_key =
      LocalDevicesP2PCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    return clique_key.status();
  }

  std::vector<GlobalDeviceId> device_group =
      LocalGlobalDeviceGroup(*collective_params);
  return clique_requests->RequestClique(*clique_key, {device_group});
}

absl::Status XlaCommRingPermuteProbeDispatch(
    se::Stream* stream, se::Stream* comm_stream, ffi::BufferR1<U32> src,
    ffi::Result<ffi::BufferR1<U32>> dst,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  if (stream == nullptr || comm_stream == nullptr) {
    return absl::InvalidArgumentError(
        "xla_comm_ring_permute_probe requires XLA stream contexts");
  }
  if (collective_params == nullptr || collective_cliques == nullptr) {
    return absl::InvalidArgumentError(
        "xla_comm_ring_permute_probe requires XLA collective contexts");
  }
  if (src.dimensions().size() != 1 || dst->dimensions().size() != 1 ||
      src.dimensions()[0] != dst->dimensions()[0]) {
    return absl::InvalidArgumentError(
        "xla_comm_ring_permute_probe expects matching rank-1 input/output");
  }

  absl::StatusOr<GpuCliqueKey> clique_key =
      LocalDevicesP2PCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    return clique_key.status();
  }

  const int64_t num_ranks = static_cast<int64_t>(clique_key->num_devices());
  if (num_ranks <= 0) {
    return absl::InvalidArgumentError(
        "xla_comm_ring_permute_probe requires a non-empty clique");
  }

  if (num_ranks == 1) {
    // Degenerate one-rank collectives still preserve input/output aliasing and
    // layout behavior, but no communicator traffic is required.
    se::DeviceAddressBase dst_addr = dst->device_memory();
    return stream->MemcpyD2D(&dst_addr, src.device_memory(), src.size_bytes());
  }

  std::optional<RankId> rank =
      clique_key->rank(collective_params->global_device_id);
  if (!rank.has_value()) {
    return absl::InvalidArgumentError(
        "xla_comm_ring_permute_probe could not resolve this device rank");
  }

  absl::StatusOr<Communicator*> comm = collective_cliques->GetComm(
      *clique_key, collective_params->global_device_id);
  if (!comm.ok()) {
    return comm.status();
  }

  const int64_t rank_value = static_cast<int64_t>(rank->value());
  std::optional<RankId> source_rank =
      RankId((rank_value + num_ranks - 1) % num_ranks);
  std::vector<RankId> target_ranks = {
      RankId((rank_value + 1) % num_ranks),
  };

  absl::Status status = comm_stream->WaitFor(stream);
  if (!status.ok()) {
    return status;
  }

  Future<> future = (*comm)->CollectivePermute(
      src.device_memory(), dst->device_memory(), src.element_type(),
      src.element_count(), source_rank, target_ranks,
      GpuCollectives::On(*comm_stream));
  status = future.Await();
  if (!status.ok()) {
    return status;
  }

  return stream->WaitFor(comm_stream);
}

absl::Status XlaCommShiftPermuteProbePrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests) {
  if (collective_params == nullptr || clique_requests == nullptr) {
    return absl::InvalidArgumentError(
        "xla_comm_shift_permute_probe requires XLA collective prepare contexts");
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

absl::Status XlaCommShiftPermuteProbeDispatch(
    se::Stream* stream, se::Stream* comm_stream, int64_t shift,
    ffi::BufferR1<U32> src, ffi::Result<ffi::BufferR1<U32>> dst,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  if (stream == nullptr || comm_stream == nullptr) {
    return absl::InvalidArgumentError(
        "xla_comm_shift_permute_probe requires XLA stream contexts");
  }
  if (collective_params == nullptr || collective_cliques == nullptr) {
    return absl::InvalidArgumentError(
        "xla_comm_shift_permute_probe requires XLA collective contexts");
  }
  if (src.dimensions().size() != 1 || dst->dimensions().size() != 1 ||
      src.dimensions()[0] != dst->dimensions()[0]) {
    return absl::InvalidArgumentError(
        "xla_comm_shift_permute_probe expects matching rank-1 input/output");
  }

  absl::StatusOr<GpuCliqueKey> clique_key =
      LocalDevicesP2PCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    return clique_key.status();
  }

  const int64_t num_ranks = static_cast<int64_t>(clique_key->num_devices());
  if (num_ranks <= 0) {
    return absl::InvalidArgumentError(
        "xla_comm_shift_permute_probe requires a non-empty clique");
  }

  const int64_t normalized_shift = ((shift % num_ranks) + num_ranks) % num_ranks;
  if (normalized_shift == 0) {
    se::DeviceAddressBase dst_addr = dst->device_memory();
    return stream->MemcpyD2D(&dst_addr, src.device_memory(), src.size_bytes());
  }

  std::optional<RankId> rank =
      clique_key->rank(collective_params->global_device_id);
  if (!rank.has_value()) {
    return absl::InvalidArgumentError(
        "xla_comm_shift_permute_probe could not resolve this device rank");
  }

  absl::StatusOr<Communicator*> comm = collective_cliques->GetComm(
      *clique_key, collective_params->global_device_id);
  if (!comm.ok()) {
    return comm.status();
  }

  const int64_t rank_value = static_cast<int64_t>(rank->value());
  std::optional<RankId> source_rank =
      RankId((rank_value + num_ranks - normalized_shift) % num_ranks);
  std::vector<RankId> target_ranks = {
      RankId((rank_value + normalized_shift) % num_ranks),
  };

  absl::Status status = comm_stream->WaitFor(stream);
  if (!status.ok()) {
    return status;
  }

  Future<> future = (*comm)->CollectivePermute(
      src.device_memory(), dst->device_memory(), src.element_type(),
      src.element_count(), source_rank, target_ranks,
      GpuCollectives::On(*comm_stream));
  status = future.Await();
  if (!status.ok()) {
    return status;
  }

  return stream->WaitFor(comm_stream);
}

absl::Status XlaCommPermuteProbePrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests) {
  if (collective_params == nullptr || clique_requests == nullptr) {
    return absl::InvalidArgumentError(
        "xla_comm_permute_probe requires XLA collective prepare contexts");
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

absl::Status XlaCommPermuteProbeDispatch(
    se::Stream* stream, se::Stream* comm_stream,
    absl::Span<const int64_t> targets, ffi::BufferR1<U32> src,
    ffi::Result<ffi::BufferR1<U32>> dst,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  if (stream == nullptr || comm_stream == nullptr) {
    return absl::InvalidArgumentError(
        "xla_comm_permute_probe requires XLA stream contexts");
  }
  if (collective_params == nullptr || collective_cliques == nullptr) {
    return absl::InvalidArgumentError(
        "xla_comm_permute_probe requires XLA collective contexts");
  }
  if (src.dimensions().size() != 1 || dst->dimensions().size() != 1 ||
      src.dimensions()[0] != dst->dimensions()[0]) {
    return absl::InvalidArgumentError(
        "xla_comm_permute_probe expects matching rank-1 input/output");
  }

  absl::StatusOr<GpuCliqueKey> clique_key =
      LocalDevicesP2PCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    return clique_key.status();
  }

  const int64_t num_ranks = static_cast<int64_t>(clique_key->num_devices());
  if (num_ranks <= 0) {
    return absl::InvalidArgumentError(
        "xla_comm_permute_probe requires a non-empty clique");
  }
  if (targets.size() != static_cast<size_t>(num_ranks)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "xla_comm_permute_probe expected %d targets, got %d", num_ranks,
        targets.size()));
  }

  std::optional<RankId> rank =
      clique_key->rank(collective_params->global_device_id);
  if (!rank.has_value()) {
    return absl::InvalidArgumentError(
        "xla_comm_permute_probe could not resolve this device rank");
  }

  std::vector<bool> seen(num_ranks, false);
  for (int64_t source = 0; source < num_ranks; ++source) {
    const int64_t target = targets[source];
    if (target < 0 || target >= num_ranks) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "xla_comm_permute_probe target[%d]=%d is outside [0, %d)", source,
          target, num_ranks));
    }
    if (seen[target]) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "xla_comm_permute_probe target rank %d appears more than once",
          target));
    }
    seen[target] = true;
  }

  absl::StatusOr<Communicator*> comm = collective_cliques->GetComm(
      *clique_key, collective_params->global_device_id);
  if (!comm.ok()) {
    return comm.status();
  }

  const int64_t rank_value = static_cast<int64_t>(rank->value());
  std::optional<RankId> source_rank;
  for (int64_t source = 0; source < num_ranks; ++source) {
    if (targets[source] == rank_value) {
      source_rank = RankId(source);
      break;
    }
  }
  if (!source_rank.has_value()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "xla_comm_permute_probe could not find a source for rank %d",
        rank_value));
  }

  const int64_t target_rank_value = targets[rank_value];
  if (source_rank->value() == rank_value && target_rank_value == rank_value) {
    se::DeviceAddressBase dst_addr = dst->device_memory();
    return stream->MemcpyD2D(&dst_addr, src.device_memory(), src.size_bytes());
  }

  std::vector<RankId> target_ranks = {RankId(target_rank_value)};

  absl::Status status = comm_stream->WaitFor(stream);
  if (!status.ok()) {
    return status;
  }

  Future<> future = (*comm)->CollectivePermute(
      src.device_memory(), dst->device_memory(), src.element_type(),
      src.element_count(), source_rank, target_ranks,
      GpuCollectives::On(*comm_stream));
  status = future.Await();
  if (!status.ok()) {
    return status;
  }

  return stream->WaitFor(comm_stream);
}

absl::Status XlaCommChunkPermuteProbePrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests) {
  if (collective_params == nullptr || clique_requests == nullptr) {
    return absl::InvalidArgumentError(
        "xla_comm_chunk_permute_probe requires XLA collective prepare contexts");
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

absl::Status XlaCommChunkPermuteProbeDispatch(
    se::Stream* stream, se::Stream* comm_stream,
    absl::Span<const int64_t> targets,
    absl::Span<const int64_t> src_offsets,
    absl::Span<const int64_t> dst_offsets, int64_t count,
    ffi::BufferR1<U32> src, ffi::Result<ffi::BufferR1<U32>> dst,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  if (stream == nullptr || comm_stream == nullptr) {
    return absl::InvalidArgumentError(
        "xla_comm_chunk_permute_probe requires XLA stream contexts");
  }
  if (collective_params == nullptr || collective_cliques == nullptr) {
    return absl::InvalidArgumentError(
        "xla_comm_chunk_permute_probe requires XLA collective contexts");
  }
  if (src.dimensions().size() != 1 || dst->dimensions().size() != 1 ||
      src.dimensions()[0] != dst->dimensions()[0]) {
    return absl::InvalidArgumentError(
        "xla_comm_chunk_permute_probe expects matching rank-1 input/output");
  }
  if (count < 0) {
    return absl::InvalidArgumentError(
        "xla_comm_chunk_permute_probe requires a non-negative count");
  }

  absl::StatusOr<GpuCliqueKey> clique_key =
      LocalDevicesP2PCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    return clique_key.status();
  }

  const int64_t num_ranks = static_cast<int64_t>(clique_key->num_devices());
  if (num_ranks <= 0) {
    return absl::InvalidArgumentError(
        "xla_comm_chunk_permute_probe requires a non-empty clique");
  }
  const size_t expected_size = static_cast<size_t>(num_ranks);
  if (targets.size() != expected_size || src_offsets.size() != expected_size ||
      dst_offsets.size() != expected_size) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "xla_comm_chunk_permute_probe expected %d targets/src_offsets/"
        "dst_offsets, got %d/%d/%d",
        num_ranks, targets.size(), src_offsets.size(), dst_offsets.size()));
  }

  std::optional<RankId> rank =
      clique_key->rank(collective_params->global_device_id);
  if (!rank.has_value()) {
    return absl::InvalidArgumentError(
        "xla_comm_chunk_permute_probe could not resolve this device rank");
  }

  const int64_t local_elements = static_cast<int64_t>(src.element_count());
  if (count > local_elements) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "xla_comm_chunk_permute_probe count %d exceeds local element count %d",
        count, local_elements));
  }

  std::vector<bool> seen(num_ranks, false);
  for (int64_t source = 0; source < num_ranks; ++source) {
    const int64_t target = targets[source];
    if (target < -1 || target >= num_ranks) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "xla_comm_chunk_permute_probe target[%d]=%d is outside [-1, %d)",
          source, target, num_ranks));
    }
    if (target >= 0) {
      if (seen[target]) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "xla_comm_chunk_permute_probe target rank %d appears more than once",
            target));
      }
      seen[target] = true;
    }
    if (src_offsets[source] < 0 ||
        src_offsets[source] + count > local_elements) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "xla_comm_chunk_permute_probe src_offsets[%d]=%d with count %d is "
          "outside local element count %d",
          source, src_offsets[source], count, local_elements));
    }
    if (dst_offsets[source] < 0 ||
        dst_offsets[source] + count > local_elements) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "xla_comm_chunk_permute_probe dst_offsets[%d]=%d with count %d is "
          "outside local element count %d",
          source, dst_offsets[source], count, local_elements));
    }
  }

  absl::StatusOr<Communicator*> comm = collective_cliques->GetComm(
      *clique_key, collective_params->global_device_id);
  if (!comm.ok()) {
    return comm.status();
  }

  se::DeviceAddressBase dst_base = dst->device_memory();
  absl::Status status =
      stream->MemcpyD2D(&dst_base, src.device_memory(), src.size_bytes());
  if (!status.ok()) {
    return status;
  }

  if (count == 0) {
    return absl::OkStatus();
  }

  const int64_t rank_value = static_cast<int64_t>(rank->value());
  const int64_t target_rank_value = targets[rank_value];
  std::optional<RankId> source_rank;
  int64_t source_for_this_rank = -1;
  for (int64_t source = 0; source < num_ranks; ++source) {
    if (targets[source] == rank_value) {
      source_rank = RankId(source);
      source_for_this_rank = source;
      break;
    }
  }

  const uint64_t element_bytes = sizeof(uint32_t);
  const uint64_t count_bytes =
      static_cast<uint64_t>(count) * element_bytes;
  se::DeviceAddressBase src_base = src.device_memory();
  se::DeviceAddressBase send_addr = src_base.GetByteSlice(0, count_bytes);
  if (target_rank_value >= 0) {
    send_addr = src_base.GetByteSlice(
        static_cast<uint64_t>(src_offsets[rank_value]) * element_bytes,
        count_bytes);
  }

  se::DeviceAddressBase recv_addr = dst_base.GetByteSlice(0, count_bytes);
  if (source_rank.has_value()) {
    recv_addr = dst_base.GetByteSlice(
        static_cast<uint64_t>(dst_offsets[source_for_this_rank]) *
            element_bytes,
        count_bytes);
  }

  if (!source_rank.has_value() && target_rank_value < 0) {
    return absl::OkStatus();
  }

  if (source_rank.has_value() && source_rank->value() == rank_value &&
      target_rank_value == rank_value) {
    return stream->MemcpyD2D(&recv_addr, send_addr, count_bytes);
  }

  std::vector<RankId> target_ranks;
  if (target_rank_value >= 0) {
    target_ranks.push_back(RankId(target_rank_value));
  }

  status = comm_stream->WaitFor(stream);
  if (!status.ok()) {
    return status;
  }

  Future<> future = (*comm)->CollectivePermute(
      send_addr, recv_addr, src.element_type(), static_cast<size_t>(count),
      source_rank, target_ranks, GpuCollectives::On(*comm_stream));
  status = future.Await();
  if (!status.ok()) {
    return status;
  }

  return stream->WaitFor(comm_stream);
}

}  // namespace xla::gpu
