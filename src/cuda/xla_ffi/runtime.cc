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
// Shared XLA FFI runtime helpers for the cuSOLVERMp backend.
//
// This file owns process-local synchronization state, common CUDA/cuSolver
// status conversion, scratch allocation, all-assigned clique construction, and
// small communicator utilities used by the solver-specific translation units.
//
// File workflow:
//   1. Translate CUDA/cuSOLVER return codes into absl::Status for FFI.
//   2. Allocate XLA-owned scratch buffers for fused native handlers.
//   3. Build all-assigned XLA collective clique keys from CollectiveParams.

#include "runtime.h"

namespace xla::gpu {
namespace {

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

}  // namespace

absl::Status CudaToStatus(cudaError_t err, const char* file, int line) {
  // Keep CUDA failures precise at the FFI boundary.  Returning the source file
  // and line has been much more useful than generic failed-precondition errors
  // when debugging remote Slurm runs.
  if (err == cudaSuccess) {
    return absl::OkStatus();
  }
  return absl::InternalError(absl::StrFormat(
      "CUDA error %d (%s) at %s:%d", static_cast<int>(err),
      cudaGetErrorString(err), file, line));
}

absl::Status CusolverToStatus(cusolverStatus_t err, const char* file,
                              int line) {
  // cuSOLVER/cuSOLVERMp status codes are numeric in several headers.  Preserve
  // the raw code and call site so Python tests can report the exact failing
  // native API call.
  if (err == CUSOLVER_STATUS_SUCCESS) {
    return absl::OkStatus();
  }
  return absl::InternalError(absl::StrFormat("cuSolver error %d at %s:%d",
                                             static_cast<int>(err), file,
                                             line));
}

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
  // Use sorted global device ids to build a stable group independent of the
  // order XLA exposes maps internally. This must match the rank order assumed
  // by the row-major/column-major cuSOLVERMp rank_map validation.
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

absl::StatusOr<GpuCliqueKey> AllAssignedDevicesCliqueKey(
    const CollectiveParams& params) {
  // All-reduce style clique over every assigned rank.
  std::vector<ReplicaGroup> replica_groups = {
      AllAssignedDevicesReplicaGroup(params)};
  return GetGpuCliqueKey(
      params, replica_groups,
      CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID, false);
}

absl::StatusOr<GpuCliqueKey> AllAssignedDevicesP2PCliqueKey(
    const CollectiveParams& params) {
  // Point-to-point clique over every assigned rank.  CommunicationId(1)
  // separates this P2P clique from the ordinary collective clique for the same
  // replica group.
  std::vector<ReplicaGroup> replica_groups = {
      AllAssignedDevicesReplicaGroup(params)};
  return GetGpuCliqueKey(
      params, replica_groups,
      CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID,
      CommunicationId(1));
}

absl::Status RequestAllAssignedP2PCommunicator(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests, const char* caller) {
  if (collective_params == nullptr || clique_requests == nullptr) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s requires XLA collective prepare contexts", caller));
  }

  // Production prepare path: ask XLA to build the communicator before runtime
  // dispatch. Dispatch cannot create this communicator lazily.
  absl::StatusOr<GpuCliqueKey> clique_key =
      AllAssignedDevicesP2PCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    return clique_key.status();
  }
  return clique_requests->RequestClique(
      *clique_key, {AllAssignedGlobalDeviceGroup(*collective_params)});
}

}  // namespace xla::gpu
