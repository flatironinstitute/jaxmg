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

#ifndef JAXMG_MPMD_IPC_H_
#define JAXMG_MPMD_IPC_H_

#include <cstdint>
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

absl::StatusOr<IpcHandleWithOffset> ExportIpcHandleWithOffset(void* ptr);
absl::StatusOr<OpenedIpcPointer> OpenIpcHandleWithOffset(
    const IpcHandleWithOffset& handle);
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

std::string MpmdSharedName(const char* prefix, int run_id, int node_group_id);

}  // namespace xla::gpu

#endif  // JAXMG_MPMD_IPC_H_
