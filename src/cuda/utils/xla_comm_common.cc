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
//
// File workflow:
//   1. Translate CUDA/cuSolverMg return codes into absl::Status for FFI.
//   2. Own the SPMD process-local barriers and pointer arrays declared in the
//      shared header.
//   3. Build XLA collective clique keys from CollectiveParams for both SPMD
//      and node-scoped MPMD execution.
//   4. Provide small communicator helpers, such as rank-0 broadcast, used by
//      potrs and syevd after cuSolverMg writes replicated outputs.

#include "../include/xla_comm_backend.h"

namespace xla::gpu {
namespace {

int EnvDevicesPerNodeOrDefault(int fallback) {
  // In MPMD a process commonly sees one local device, while cuSolverMg still
  // needs the number of GPUs participating in this node-local solve. The Python
  // setup code supplies JAXMG_NUMBER_OF_DEVICES for that case.
  const char* env = std::getenv("JAXMG_NUMBER_OF_DEVICES");
  if (env == nullptr || env[0] == '\0') {
    return std::max(fallback, 1);
  }
  char* end = nullptr;
  long parsed = std::strtol(env, &end, 10);
  if (end == env || parsed <= 0 || parsed > 16) {
    return std::max(fallback, 1);
  }
  return static_cast<int>(parsed);
}

struct AssignedDeviceEntry {
  int replica_id;
  GlobalDeviceId global_device_id;
};

std::vector<AssignedDeviceEntry> AssignedDevices(
    const CollectiveParams& params) {
  // Prefer XLA's explicit device assignment when present. Some FFI contexts
  // instead provide a global_device_id_map; the fallback handles simple local
  // tests where neither map is populated.
  std::vector<AssignedDeviceEntry> devices;
  if (params.device_assn != nullptr) {
    const int replica_count = params.device_assn->replica_count();
    const int computation_count = params.device_assn->computation_count();
    devices.reserve(static_cast<size_t>(replica_count) *
                    static_cast<size_t>(computation_count));
    for (int replica = 0; replica < replica_count; ++replica) {
      for (int computation = 0; computation < computation_count;
           ++computation) {
        const int flattened_id = replica * computation_count + computation;
        devices.push_back(AssignedDeviceEntry{
            flattened_id,
            GlobalDeviceId((*params.device_assn)(replica, computation))});
      }
    }
    return devices;
  }

  if (params.global_device_id_map != nullptr &&
      !params.global_device_id_map->empty()) {
    devices.reserve(params.global_device_id_map->size());
    for (const auto& entry : *params.global_device_id_map) {
      devices.push_back(AssignedDeviceEntry{entry.first.value(),
                                            entry.second});
    }
    return devices;
  }

  const int device_count = std::max(params.local_device_count, 1);
  devices.reserve(device_count);
  for (int i = 0; i < device_count; ++i) {
    devices.push_back(AssignedDeviceEntry{i, GlobalDeviceId(i)});
  }
  return devices;
}

absl::StatusOr<std::pair<int, int>> NodeScopedIndexRange(
    const CollectiveParams& params, int device_count) {
  // cuSolverMg remains single-node. For MPMD/multi-process launches, split the
  // global device assignment into consecutive node-sized groups and choose the
  // group containing this rank's global device id.
  std::vector<AssignedDeviceEntry> devices = AssignedDevices(params);
  std::sort(devices.begin(), devices.end(),
            [](const AssignedDeviceEntry& a, const AssignedDeviceEntry& b) {
              return a.global_device_id.value() < b.global_device_id.value();
            });
  auto self = std::find_if(
      devices.begin(), devices.end(), [&](const AssignedDeviceEntry& entry) {
        return entry.global_device_id == params.global_device_id;
      });
  if (self == devices.end()) {
    return absl::InvalidArgumentError(
        "Could not find this device in XLA device assignment");
  }

  const int self_index = static_cast<int>(self - devices.begin());
  const int node_start = (self_index / device_count) * device_count;
  const int node_end =
      std::min(node_start + device_count, static_cast<int>(devices.size()));
  if (node_end <= node_start) {
    return absl::InvalidArgumentError(
        "Computed an empty node-scoped XLA communicator group");
  }
  return std::pair<int, int>(node_start, node_end);
}

}  // namespace

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

ReplicaGroup AllAssignedDevicesReplicaGroup(const CollectiveParams& params) {
  // cuSOLVERMp is the distributed-memory migration target. Unlike cuSolverMg,
  // which is intentionally scoped to one node, this helper keeps every assigned
  // replica in one group so the borrowed NCCL communicator can represent the
  // full cuSOLVERMp process grid when JAX is launched across nodes.
  std::vector<AssignedDeviceEntry> devices = AssignedDevices(params);
  std::sort(devices.begin(), devices.end(),
            [](const AssignedDeviceEntry& a, const AssignedDeviceEntry& b) {
              return a.global_device_id.value() < b.global_device_id.value();
            });

  ReplicaGroup group;
  for (const AssignedDeviceEntry& device : devices) {
    group.add_replica_ids(device.replica_id);
  }
  return group;
}

std::vector<GlobalDeviceId> AllAssignedGlobalDeviceGroup(
    const CollectiveParams& params) {
  std::vector<AssignedDeviceEntry> devices = AssignedDevices(params);
  std::sort(devices.begin(), devices.end(),
            [](const AssignedDeviceEntry& a, const AssignedDeviceEntry& b) {
              return a.global_device_id.value() < b.global_device_id.value();
            });

  std::vector<GlobalDeviceId> device_group;
  device_group.reserve(devices.size());
  for (const AssignedDeviceEntry& device : devices) {
    device_group.push_back(device.global_device_id);
  }
  return device_group;
}

absl::StatusOr<std::vector<GlobalDeviceId>> NodeScopedGlobalDeviceGroup(
    const CollectiveParams& params) {
  std::vector<AssignedDeviceEntry> devices = AssignedDevices(params);
  std::sort(devices.begin(), devices.end(),
            [](const AssignedDeviceEntry& a, const AssignedDeviceEntry& b) {
              return a.global_device_id.value() < b.global_device_id.value();
            });
  const int devices_per_node =
      EnvDevicesPerNodeOrDefault(static_cast<int>(devices.size()));
  absl::StatusOr<std::pair<int, int>> range =
      NodeScopedIndexRange(params, devices_per_node);
  if (!range.ok()) {
    return range.status();
  }

  std::vector<GlobalDeviceId> device_group;
  device_group.reserve(range->second - range->first);
  for (int i = range->first; i < range->second; ++i) {
    device_group.push_back(devices[i].global_device_id);
  }
  return device_group;
}

absl::StatusOr<ReplicaGroup> NodeScopedReplicaGroup(
    const CollectiveParams& params) {
  // MPMD helper: construct the replica group for only the current node's local
  // ranks. This prevents single-node cuSolverMg paths from accidentally
  // requesting a multi-node communicator.
  std::vector<AssignedDeviceEntry> devices = AssignedDevices(params);
  std::sort(devices.begin(), devices.end(),
            [](const AssignedDeviceEntry& a, const AssignedDeviceEntry& b) {
              return a.global_device_id.value() < b.global_device_id.value();
            });
  const int devices_per_node =
      EnvDevicesPerNodeOrDefault(static_cast<int>(devices.size()));
  absl::StatusOr<std::pair<int, int>> range =
      NodeScopedIndexRange(params, devices_per_node);
  if (!range.ok()) {
    return range.status();
  }

  ReplicaGroup group;
  for (int i = range->first; i < range->second; ++i) {
    group.add_replica_ids(devices[i].replica_id);
  }
  return group;
}

absl::StatusOr<int> NodeScopedGroupOrdinal(const CollectiveParams& params) {
  std::vector<AssignedDeviceEntry> devices = AssignedDevices(params);
  std::sort(devices.begin(), devices.end(),
            [](const AssignedDeviceEntry& a, const AssignedDeviceEntry& b) {
              return a.global_device_id.value() < b.global_device_id.value();
            });
  const int devices_per_node =
      EnvDevicesPerNodeOrDefault(static_cast<int>(devices.size()));
  absl::StatusOr<std::pair<int, int>> range =
      NodeScopedIndexRange(params, devices_per_node);
  if (!range.ok()) {
    return range.status();
  }
  return range->first / devices_per_node;
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

absl::StatusOr<GpuCliqueKey> AllAssignedDevicesCliqueKey(
    const CollectiveParams& params) {
  std::vector<ReplicaGroup> replica_groups = {
      AllAssignedDevicesReplicaGroup(params)};
  return GetGpuCliqueKey(
      params, replica_groups,
      CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID, false);
}

absl::StatusOr<GpuCliqueKey> AllAssignedDevicesP2PCliqueKey(
    const CollectiveParams& params) {
  std::vector<ReplicaGroup> replica_groups = {
      AllAssignedDevicesReplicaGroup(params)};
  return GetGpuCliqueKey(
      params, replica_groups,
      CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID,
      CommunicationId(1));
}

absl::StatusOr<GpuCliqueKey> NodeScopedCliqueKey(
    const CollectiveParams& params) {
  absl::StatusOr<ReplicaGroup> replica_group = NodeScopedReplicaGroup(params);
  if (!replica_group.ok()) {
    return replica_group.status();
  }
  std::vector<ReplicaGroup> replica_groups = {*replica_group};
  return GetGpuCliqueKey(
      params, replica_groups,
      CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID, false);
}

absl::StatusOr<GpuCliqueKey> NodeScopedP2PCliqueKey(
    const CollectiveParams& params) {
  absl::StatusOr<ReplicaGroup> replica_group = NodeScopedReplicaGroup(params);
  if (!replica_group.ok()) {
    return replica_group.status();
  }
  std::vector<ReplicaGroup> replica_groups = {*replica_group};
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
