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
// Shared helper definitions for the XLA communicator cuSolverMg backend.
//
// This file owns process-local synchronization state, common CUDA/cuSolver
// status conversion, scratch allocation, local clique construction, and small
// communicator utilities used by the solver-specific translation units.

#include "../include/xla_comm_backend.h"

namespace xla::gpu {

absl::Status CudaToStatus(cudaError_t err, const char* file, int line) {
  if (err == cudaSuccess) {
    return absl::OkStatus();
  }
  return absl::InternalError(absl::StrFormat(
      "CUDA error %d (%s) at %s:%d", static_cast<int>(err),
      cudaGetErrorString(err), file, line));
}

absl::Status CusolverToStatus(cusolverStatus_t err, const char* file,
                              int line) {
  if (err == CUSOLVER_STATUS_SUCCESS) {
    return absl::OkStatus();
  }
  return absl::InternalError(absl::StrFormat("cuSolver error %d at %s:%d",
                                             static_cast<int>(err), file,
                                             line));
}

ReusableHostBarrier fused_potrs_barrier;
FusedPotrsState fused_potrs_state;
ReusableHostBarrier fused_potri_barrier;
FusedMatrixState fused_potri_state;
ReusableHostBarrier fused_syevd_barrier;
FusedSyevdState fused_syevd_state;
ReusableHostBarrier fused_syevd_no_v_barrier;
FusedSyevdState fused_syevd_no_v_state;

absl::StatusOr<void*> AllocateFfiScratch(se::ScratchAllocator& scratch,
                                         size_t bytes, const char* name) {
  // XLA owns the lifetime of scratch allocations for the FFI call. Returning
  // the raw pointer is safe only within the current invocation; solver handlers
  // publish it through process-local state after all ranks have allocated.
  absl::StatusOr<se::DeviceAddress<uint8_t>> allocation =
      scratch.AllocateBytes(static_cast<int64_t>(bytes));
  if (!allocation.ok()) {
    return absl::ResourceExhaustedError(absl::StrFormat(
        "Unable to allocate scratch memory for %s: %s", name,
        allocation.status().message()));
  }
  return allocation->opaque();
}

ReplicaGroup LocalDevicesReplicaGroup(const CollectiveParams& params) {
  ReplicaGroup group;
  const int device_count = std::max(params.local_device_count, 1);
  // The production backend is single-node for now. Replica ids therefore match
  // local flattened device ids.
  for (int i = 0; i < device_count; ++i) {
    group.add_replica_ids(i);
  }
  return group;
}

std::vector<GlobalDeviceId> LocalGlobalDeviceGroup(
    const CollectiveParams& params) {
  std::vector<GlobalDeviceId> device_group;
  if (params.global_device_id_map != nullptr &&
      !params.global_device_id_map->empty()) {
    // Prefer XLA's explicit global device map when present so the diagnostic
    // handlers and production path agree with XLA's device ordering.
    device_group.reserve(params.global_device_id_map->size());
    for (const auto& entry : *params.global_device_id_map) {
      device_group.push_back(entry.second);
    }
    std::sort(device_group.begin(), device_group.end(),
              [](const GlobalDeviceId& a, const GlobalDeviceId& b) {
                return a.value() < b.value();
              });
    return device_group;
  }

  const int device_count = std::max(params.local_device_count, 1);
  device_group.reserve(device_count);
  for (int i = 0; i < device_count; ++i) {
    device_group.push_back(GlobalDeviceId(i));
  }
  return device_group;
}

absl::StatusOr<GpuCliqueKey> LocalDevicesCliqueKey(
    const CollectiveParams& params) {
  // Flattened-id mode matches the 1D mesh used by the Python wrappers.
  std::vector<ReplicaGroup> replica_groups = {LocalDevicesReplicaGroup(params)};
  return GetGpuCliqueKey(
      params, replica_groups,
      CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID, false);
}

absl::StatusOr<GpuCliqueKey> LocalDevicesP2PCliqueKey(
    const CollectiveParams& params) {
  // Point-to-point CollectivePermute uses a communication id. Keep this helper
  // separate from LocalDevicesCliqueKey so all move-style handlers request the
  // same clique shape.
  std::vector<ReplicaGroup> replica_groups = {LocalDevicesReplicaGroup(params)};
  return GetGpuCliqueKey(
      params, replica_groups,
      CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID,
      CommunicationId(1));
}

absl::Status BroadcastBufferFromRank0(
    const char* caller, se::Stream* stream, se::Stream* comm_stream,
    Communicator* comm, se::DeviceAddressBase buffer, PrimitiveType element_type,
    size_t element_count, int64_t num_ranks, int64_t rank_value) {
  // cuSolverMg returns some outputs only on the rank-0 host path. Keep the
  // broadcast on the XLA communicator rather than falling back to CUDA peer
  // copies so the backend has one communication mechanism.
  if (num_ranks <= 1 || element_count == 0) {
    return absl::OkStatus();
  }
  if (stream == nullptr || comm_stream == nullptr || comm == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s requires stream and communicator contexts", caller));
  }
  if (rank_value < 0 || rank_value >= num_ranks) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s rank %d is outside [0, %d)", caller, rank_value, num_ranks));
  }

  for (int64_t target = 1; target < num_ranks; ++target) {
    // Use a sequence of rank-0 -> target point-to-point sends. This avoids a
    // separate CUDA peer broadcast path and keeps ordering explicit on the XLA
    // communication stream.
    std::optional<RankId> source_rank;
    if (rank_value == target) {
      source_rank = RankId(0);
    }

    std::vector<RankId> target_ranks;
    if (rank_value == 0) {
      target_ranks.push_back(RankId(target));
    }

    if (!source_rank.has_value() && target_ranks.empty()) {
      continue;
    }

    absl::Status status = comm_stream->WaitFor(stream);
    if (!status.ok()) {
      return status;
    }
    Future<> future = comm->CollectivePermute(
        buffer, buffer, element_type, element_count, source_rank, target_ranks,
        GpuCollectives::On(*comm_stream));
    status = future.Await();
    if (!status.ok()) {
      return status;
    }
    status = stream->WaitFor(comm_stream);
    if (!status.ok()) {
      return status;
    }
  }

  return absl::OkStatus();
}

}  // namespace xla::gpu
