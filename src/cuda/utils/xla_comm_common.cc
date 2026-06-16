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
// Shared helper definitions for the XLA communicator cuSOLVERMp backend.
//
// This file owns process-local synchronization state, common CUDA/cuSolver
// status conversion, scratch allocation, local clique construction, and small
// communicator utilities used by the solver-specific translation units.
//
// File workflow:
//   1. Translate CUDA/cuSOLVER return codes into absl::Status for FFI.
//   2. Allocate XLA-owned scratch buffers for fused native handlers.
//   3. Build all-assigned or node-scoped XLA collective clique keys from
//      CollectiveParams.

#include "../include/xla_comm_backend.h"

namespace xla::gpu {
namespace {

int EnvDevicesPerNodeOrDefault(int fallback) {
  // Diagnostic node-scoped communicators need the host-local group size when a
  // distributed launch exposes only one local device to each process.
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
  // Split the global device assignment into consecutive host-sized groups and
  // choose the group containing this rank's global device id.
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

absl::StatusOr<void*> AllocateFfiScratch(ffi::ScratchAllocator& scratch,
                                         size_t bytes, const char* name) {
  // XLA owns the lifetime of scratch allocations for the FFI call. Returning
  // the raw pointer is safe only within the current invocation; solver handlers
  // publish it through process-local state after all ranks have allocated.
  std::optional<void*> allocation = scratch.Allocate(bytes);
  if (!allocation.has_value()) {
    return absl::ResourceExhaustedError(absl::StrFormat(
        "Unable to allocate scratch memory for %s", name));
  }
  return *allocation;
}

ReplicaGroup AllAssignedDevicesReplicaGroup(const CollectiveParams& params) {
  // Keep every assigned replica in one group so the borrowed NCCL communicator
  // can represent the full cuSOLVERMp process grid when JAX is launched across
  // nodes.
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
  // Diagnostic helper: construct the replica group for only the current
  // host-local ranks.
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

}  // namespace xla::gpu
