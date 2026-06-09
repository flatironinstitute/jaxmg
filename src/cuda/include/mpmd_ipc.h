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
// Utilities for the MPMD cuSolverMg compatibility layer.
//
// XLA/NCCL handles GPU-to-GPU communication, but cuSolverMg still needs one
// host process to build arrays of device pointers. In MPMD those pointers live
// in different processes, so the solver layer must exchange CUDA IPC handles
// and XLA buffer offsets before the rank-0 host call.
//
// Header workflow:
//   1. IpcHandleWithOffset describes a remote CUDA allocation plus the byte
//      offset of the actual JAX buffer inside that allocation.
//   2. MpmdProcessBarrier synchronizes participating local processes using
//      POSIX shared memory.
//   3. SharedMemoryArray<T> publishes one value per rank for handles, workspace
//      sizes, and solver statuses.
//   4. MpmdSolverExchange combines the barrier and shared arrays into the
//      pointer exchange protocol used by potrs, potri, and syevd.
//
// This layer is not a replacement for the XLA communicator. It only solves the
// host-side pointer visibility problem that cuSolverMg imposes in MPMD mode.

#ifndef JAXMG_MPMD_IPC_H_
#define JAXMG_MPMD_IPC_H_

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace xla::gpu {

struct IpcHandleWithOffset {
  cudaIpcMemHandle_t handle{};
  uintptr_t offset = 0;
};

struct OpenedIpcPointer {
  void* base = nullptr;
  void* ptr = nullptr;
};

// CUDA IPC helpers. ExportIpcHandleWithOffset converts a local JAX device
// pointer into a process-shareable handle. Open* maps the remote allocation in
// the rank-0 process and reconstructs the original pointer by reapplying the
// recorded offset.
absl::StatusOr<IpcHandleWithOffset> ExportIpcHandleWithOffset(void* ptr);
absl::StatusOr<OpenedIpcPointer> OpenIpcHandleWithOffset(
    const IpcHandleWithOffset& handle);
absl::StatusOr<OpenedIpcPointer> OpenIpcHandleWithOffsetOnDevice(
    const IpcHandleWithOffset& handle, int device);
absl::Status CloseIpcPointer(const OpenedIpcPointer& opened);

class MpmdProcessBarrier {
 public:
  MpmdProcessBarrier(int participants, const std::string& name);
  MpmdProcessBarrier(const MpmdProcessBarrier&) = delete;
  MpmdProcessBarrier& operator=(const MpmdProcessBarrier&) = delete;
  ~MpmdProcessBarrier();

  bool ok() const { return status_.ok(); }
  const absl::Status& status() const { return status_; }
  absl::Status ArriveAndWait();
  absl::Status CloseAndUnlink();

 private:
  struct Shared;

  std::string name_;
  Shared* shared_ = nullptr;
  bool owner_ = false;
  absl::Status status_;
};

// Small POSIX shared-memory array with rank-indexed entries. The owner process
// unlinks the shared memory name; all processes unmap their local view.
template <typename T>
class SharedMemoryArray {
 public:
  SharedMemoryArray(int rank, int count, const std::string& name);
  SharedMemoryArray(const SharedMemoryArray&) = delete;
  SharedMemoryArray& operator=(const SharedMemoryArray&) = delete;
  ~SharedMemoryArray();

  bool ok() const { return status_.ok(); }
  const absl::Status& status() const { return status_; }
  T* data() { return data_; }
  const T* data() const { return data_; }
  T& operator[](int index) { return data_[index]; }
  const T& operator[](int index) const { return data_[index]; }

  absl::Status CloseAndUnlink();

 private:
  std::string name_;
  T* data_ = nullptr;
  size_t bytes_ = 0;
  bool owner_ = false;
  absl::Status status_;
};

std::string MpmdSharedName(const char* prefix, int64_t run_id,
                           int node_group_id);

struct MpmdSolverExchangeConfig {
  int rank = 0;
  int count = 0;
  int64_t run_id = 0;
  int node_group_id = 0;
  const char* prefix = nullptr;
  bool include_b = false;
};

// Per-solver MPMD exchange object. The solver handler constructs one exchange
// per invocation, publishes its local pointers, waits for all local ranks, and
// lets rank 0 open the remote pointers for cuSolverMg.
class MpmdSolverExchange {
 public:
  explicit MpmdSolverExchange(const MpmdSolverExchangeConfig& config);
  MpmdSolverExchange(const MpmdSolverExchange&) = delete;
  MpmdSolverExchange& operator=(const MpmdSolverExchange&) = delete;

  bool ok() const { return status_.ok(); }
  const absl::Status& status() const { return status_; }

  absl::Status ArriveAndWait();
  absl::Status PublishA(void* ptr);
  absl::Status PublishB(void* ptr);
  absl::Status PublishWork(void* ptr);
  absl::StatusOr<void*> OpenAOnDevice(int device,
                                      OpenedIpcPointer* opened) const;
  absl::StatusOr<void*> OpenBOnDevice(int device,
                                      OpenedIpcPointer* opened) const;
  absl::StatusOr<void*> OpenWorkOnDevice(int device,
                                         OpenedIpcPointer* opened) const;

  void SetLwork(int device, int64_t lwork);
  int64_t Lwork(int device) const;
  void SetSolverStatus(int device, int32_t solver_status);
  int32_t SolverStatus(int device) const;

  absl::Status CloseAndUnlink();

 private:
  absl::Status PublishPointer(
      std::optional<SharedMemoryArray<IpcHandleWithOffset>>& handles,
      const char* label, void* ptr);
  absl::StatusOr<void*> OpenPointerOnDevice(
      const std::optional<SharedMemoryArray<IpcHandleWithOffset>>& handles,
      const char* label, int device, OpenedIpcPointer* opened) const;

  int rank_ = 0;
  int count_ = 0;
  absl::Status status_;
  std::optional<MpmdProcessBarrier> barrier_;
  std::optional<SharedMemoryArray<IpcHandleWithOffset>> a_ipc_;
  std::optional<SharedMemoryArray<IpcHandleWithOffset>> b_ipc_;
  std::optional<SharedMemoryArray<IpcHandleWithOffset>> work_ipc_;
  std::optional<SharedMemoryArray<int64_t>> lwork_;
  std::optional<SharedMemoryArray<int32_t>> solver_status_;
};

absl::Status CloseOpenedRemoteIpcPointers(
    int num_ranks, int current_device,
    std::initializer_list<OpenedIpcPointer*> opened_sets);

}  // namespace xla::gpu

#endif  // JAXMG_MPMD_IPC_H_
