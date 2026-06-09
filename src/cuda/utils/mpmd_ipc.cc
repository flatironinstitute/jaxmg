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
// MPMD IPC implementation.
//
// Workflow:
//   1. Each process exports CUDA IPC handles for the JAX buffers and scratch
//      buffers it owns.
//   2. The handles are written into rank-indexed POSIX shared-memory arrays.
//   3. A shared-memory pthread barrier waits until all local ranks have
//      published their entries.
//   4. Rank 0 opens the remote CUDA IPC handles on the correct devices and
//      passes those pointers to cuSolverMg.
//   5. All mapped remote handles are closed before the FFI invocation returns.
//
// This file intentionally does not move matrix data. Data movement remains the
// responsibility of the XLA/NCCL communicator; this file only makes pointers
// visible to the single cuSolverMg host call.

#include "../include/mpmd_ipc.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "absl/strings/str_format.h"

namespace xla::gpu {
namespace {

absl::Status ErrnoStatus(const char* what) {
  return absl::InternalError(
      absl::StrFormat("%s: %s", what, std::strerror(errno)));
}

absl::Status CudaDriverStatus(CUresult result, const char* what) {
  if (result == CUDA_SUCCESS) {
    return absl::OkStatus();
  }
  const char* name = nullptr;
  const char* desc = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &desc);
  return absl::InternalError(absl::StrFormat(
      "%s failed: %s - %s", what, name ? name : "UNKNOWN",
      desc ? desc : "UNKNOWN"));
}

absl::Status CudaRuntimeStatus(cudaError_t result, const char* what) {
  if (result == cudaSuccess) {
    return absl::OkStatus();
  }
  return absl::InternalError(
      absl::StrFormat("%s failed: %s", what, cudaGetErrorString(result)));
}

template <typename T>
absl::StatusOr<T*> MapSharedArray(int count, const std::string& name,
                                  size_t* bytes, bool* owner) {
  // All ranks map the same named shared-memory object. The first rank to create
  // it sizes the object; later ranks wait for the size to become visible before
  // mmap so they do not fault on first access.
  if (count <= 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Shared memory array %s needs positive count, got %d",
                        name, count));
  }
  *bytes = sizeof(T) * static_cast<size_t>(count);
  int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
  if (fd >= 0) {
    *owner = true;
    if (ftruncate(fd, static_cast<off_t>(*bytes)) != 0) {
      close(fd);
      shm_unlink(name.c_str());
      return ErrnoStatus("ftruncate");
    }
  } else {
    if (errno != EEXIST) {
      return ErrnoStatus("shm_open create");
    }
    *owner = false;
    fd = shm_open(name.c_str(), O_RDWR, 0600);
    if (fd < 0) {
      return ErrnoStatus("shm_open existing");
    }
    // The name becomes visible before the creating process has necessarily
    // completed ftruncate. Wait until the object is large enough to avoid a
    // successful mmap followed by SIGBUS on first access.
    struct stat st;
    while (fstat(fd, &st) == 0 &&
           static_cast<size_t>(st.st_size) < *bytes) {
      sched_yield();
    }
    if (fstat(fd, &st) != 0) {
      close(fd);
      return ErrnoStatus("fstat");
    }
  }

  void* mapped = mmap(nullptr, *bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (mapped == MAP_FAILED) {
    return ErrnoStatus("mmap");
  }
  if (*owner) {
    std::memset(mapped, 0, *bytes);
  }
  return static_cast<T*>(mapped);
}

}  // namespace

struct MpmdProcessBarrier::Shared {
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  int participants;
  int remaining;
  int generation;
  int initialized;
};

// Export a pointer in a way that survives JAX allocator suballocation. CUDA IPC
// can export only the base allocation, so the byte offset to the requested
// pointer is stored alongside the handle.
absl::StatusOr<IpcHandleWithOffset> ExportIpcHandleWithOffset(void* ptr) {
  CUdeviceptr base = 0;
  size_t size = 0;
  CUresult range_status =
      cuMemGetAddressRange(&base, &size, reinterpret_cast<CUdeviceptr>(ptr));
  absl::Status status =
      CudaDriverStatus(range_status, "cuMemGetAddressRange");
  if (!status.ok()) {
    return status;
  }

  IpcHandleWithOffset result;
  status = CudaRuntimeStatus(
      cudaIpcGetMemHandle(&result.handle, reinterpret_cast<void*>(base)),
      "cudaIpcGetMemHandle");
  if (!status.ok()) {
    return status;
  }
  result.offset = reinterpret_cast<uintptr_t>(ptr) -
                  static_cast<uintptr_t>(base);
  return result;
}

absl::StatusOr<OpenedIpcPointer> OpenIpcHandleWithOffset(
    const IpcHandleWithOffset& handle) {
  // Open the remote base allocation and reconstruct the original pointer from
  // the recorded offset. The caller must eventually pass the returned object to
  // CloseIpcPointer.
  void* base = nullptr;
  absl::Status status = CudaRuntimeStatus(
      cudaIpcOpenMemHandle(&base, handle.handle, cudaIpcMemLazyEnablePeerAccess),
      "cudaIpcOpenMemHandle");
  if (!status.ok()) {
    return status;
  }
  auto* logical =
      static_cast<unsigned char*>(base) + static_cast<size_t>(handle.offset);
  return OpenedIpcPointer{base, logical};
}

absl::StatusOr<OpenedIpcPointer> OpenIpcHandleWithOffsetOnDevice(
    const IpcHandleWithOffset& handle, int device) {
  int current_device = 0;
  absl::Status status =
      CudaRuntimeStatus(cudaGetDevice(&current_device), "cudaGetDevice");
  if (!status.ok()) {
    return status;
  }
  status = CudaRuntimeStatus(cudaSetDevice(device), "cudaSetDevice");
  if (!status.ok()) {
    return status;
  }
  absl::StatusOr<OpenedIpcPointer> opened = OpenIpcHandleWithOffset(handle);
  status = CudaRuntimeStatus(cudaSetDevice(current_device), "cudaSetDevice");
  if (!status.ok()) {
    return status;
  }
  return opened;
}

absl::Status CloseIpcPointer(const OpenedIpcPointer& opened) {
  if (opened.base == nullptr) {
    return absl::OkStatus();
  }
  return CudaRuntimeStatus(cudaIpcCloseMemHandle(opened.base),
                           "cudaIpcCloseMemHandle");
}

MpmdProcessBarrier::MpmdProcessBarrier(int participants,
                                       const std::string& name)
    : name_(name), status_(absl::OkStatus()) {
  // The barrier lives in shared memory so independently launched JAX processes
  // can coordinate before rank 0 enters cuSolverMg.
  if (participants <= 0) {
    status_ = absl::InvalidArgumentError(absl::StrFormat(
        "MPMD process barrier %s needs positive participant count, got %d",
        name_, participants));
    return;
  }
  size_t bytes = 0;
  absl::StatusOr<Shared*> mapped =
      MapSharedArray<Shared>(/*count=*/1, name_, &bytes, &owner_);
  if (!mapped.ok()) {
    status_ = mapped.status();
    return;
  }
  shared_ = *mapped;

  if (owner_) {
    pthread_mutexattr_t mutex_attr;
    pthread_condattr_t cond_attr;
    pthread_mutexattr_init(&mutex_attr);
    pthread_mutexattr_setpshared(&mutex_attr, PTHREAD_PROCESS_SHARED);
    pthread_condattr_init(&cond_attr);
    pthread_condattr_setpshared(&cond_attr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(&shared_->mutex, &mutex_attr);
    pthread_cond_init(&shared_->cond, &cond_attr);
    pthread_mutexattr_destroy(&mutex_attr);
    pthread_condattr_destroy(&cond_attr);
    shared_->participants = participants;
    shared_->remaining = participants;
    shared_->generation = 0;
    __sync_synchronize();
    shared_->initialized = 1;
  } else {
    while (__atomic_load_n(&shared_->initialized, __ATOMIC_ACQUIRE) == 0) {
      sched_yield();
    }
  }
}

MpmdProcessBarrier::~MpmdProcessBarrier() {
  if (shared_ != nullptr) {
    munmap(shared_, sizeof(Shared));
    shared_ = nullptr;
  }
}

absl::Status MpmdProcessBarrier::ArriveAndWait() {
  if (!status_.ok()) {
    return status_;
  }
  if (shared_ == nullptr) {
    return absl::InternalError("MPMD process barrier is not initialized");
  }
  pthread_mutex_lock(&shared_->mutex);
  const int generation = shared_->generation;
  if (--shared_->remaining == 0) {
    shared_->remaining = shared_->participants;
    ++shared_->generation;
    pthread_cond_broadcast(&shared_->cond);
    pthread_mutex_unlock(&shared_->mutex);
    return absl::OkStatus();
  }
  while (generation == shared_->generation) {
    pthread_cond_wait(&shared_->cond, &shared_->mutex);
  }
  pthread_mutex_unlock(&shared_->mutex);
  return absl::OkStatus();
}

absl::Status MpmdProcessBarrier::CloseAndUnlink() {
  absl::Status status = ArriveAndWait();
  if (!status.ok()) {
    return status;
  }
  if (owner_) {
    shm_unlink(name_.c_str());
  }
  return ArriveAndWait();
}

template <typename T>
SharedMemoryArray<T>::SharedMemoryArray(int rank, int count,
                                        const std::string& name)
    : name_(name), status_(absl::OkStatus()) {
  if (rank < 0 || rank >= count) {
    status_ = absl::InvalidArgumentError(absl::StrFormat(
        "Rank %d is outside shared memory array %s count %d", rank, name_,
        count));
    return;
  }
  absl::StatusOr<T*> mapped =
      MapSharedArray<T>(count, name_, &bytes_, &owner_);
  if (mapped.ok()) {
    data_ = *mapped;
  } else {
    status_ = mapped.status();
  }
}

template <typename T>
SharedMemoryArray<T>::~SharedMemoryArray() {
  if (data_ != nullptr) {
    munmap(data_, bytes_);
    data_ = nullptr;
  }
}

template <typename T>
absl::Status SharedMemoryArray<T>::CloseAndUnlink() {
  if (!status_.ok()) {
    return status_;
  }
  if (owner_) {
    shm_unlink(name_.c_str());
  }
  return absl::OkStatus();
}

std::string MpmdSharedName(const char* prefix, int64_t run_id,
                           int node_group_id) {
  return absl::StrFormat("/jaxmg_%s_%lld_%d", prefix,
                         static_cast<long long>(run_id), node_group_id);
}

namespace {

std::string MpmdSolverSharedName(const char* prefix, const char* suffix,
                                 int64_t run_id, int node_group_id) {
  std::string full_prefix = absl::StrFormat("%s_%s", prefix, suffix);
  return MpmdSharedName(full_prefix.c_str(), run_id, node_group_id);
}

absl::Status FirstError(absl::Status current, absl::Status next) {
  if (!current.ok()) {
    return current;
  }
  return next;
}

}  // namespace

MpmdSolverExchange::MpmdSolverExchange(
    const MpmdSolverExchangeConfig& config)
    : rank_(config.rank), count_(config.count), status_(absl::OkStatus()) {
  // Build one shared-memory namespace per solver invocation. The run id and
  // node group id keep concurrent calls and separate nodes from colliding.
  if (rank_ < 0 || count_ <= 0 || rank_ >= count_) {
    status_ = absl::InvalidArgumentError(absl::StrFormat(
        "MPMD solver exchange rank/count mismatch: rank=%d count=%d", rank_,
        count_));
    return;
  }
  if (config.prefix == nullptr || config.prefix[0] == '\0') {
    status_ = absl::InvalidArgumentError(
        "MPMD solver exchange requires a non-empty prefix");
    return;
  }

  const int64_t run_id = config.run_id;
  const int node_group_id = config.node_group_id;
  barrier_.emplace(
      count_, MpmdSolverSharedName(config.prefix, "barrier", run_id,
                                   node_group_id));
  if (!barrier_->ok()) {
    status_ = barrier_->status();
    return;
  }
  a_ipc_.emplace(rank_, count_,
                 MpmdSolverSharedName(config.prefix, "a", run_id,
                                      node_group_id));
  if (config.include_b) {
    b_ipc_.emplace(rank_, count_,
                   MpmdSolverSharedName(config.prefix, "b", run_id,
                                        node_group_id));
  }
  work_ipc_.emplace(rank_, count_,
                    MpmdSolverSharedName(config.prefix, "work", run_id,
                                         node_group_id));
  lwork_.emplace(rank_, count_,
                 MpmdSolverSharedName(config.prefix, "lwork", run_id,
                                      node_group_id));
  solver_status_.emplace(
      rank_, count_,
      MpmdSolverSharedName(config.prefix, "status", run_id, node_group_id));

  if (!a_ipc_->ok()) status_ = a_ipc_->status();
  if (b_ipc_.has_value() && !b_ipc_->ok()) status_ = b_ipc_->status();
  if (!work_ipc_->ok()) status_ = work_ipc_->status();
  if (!lwork_->ok()) status_ = lwork_->status();
  if (!solver_status_->ok()) status_ = solver_status_->status();
}

absl::Status MpmdSolverExchange::ArriveAndWait() {
  if (!status_.ok()) {
    return status_;
  }
  if (!barrier_.has_value()) {
    return absl::InternalError("MPMD solver exchange barrier is not set");
  }
  return barrier_->ArriveAndWait();
}

absl::Status MpmdSolverExchange::PublishPointer(
    std::optional<SharedMemoryArray<IpcHandleWithOffset>>& handles,
    const char* label, void* ptr) {
  // Publish this rank's pointer after converting it to an IPC handle. The
  // actual cross-process synchronization is explicit through ArriveAndWait.
  if (!status_.ok()) {
    return status_;
  }
  if (!handles.has_value()) {
    return absl::InternalError(
        absl::StrFormat("MPMD solver exchange has no %s handle array", label));
  }
  absl::StatusOr<IpcHandleWithOffset> handle =
      ExportIpcHandleWithOffset(ptr);
  if (!handle.ok()) {
    return handle.status();
  }
  (*handles)[rank_] = *handle;
  return absl::OkStatus();
}

absl::Status MpmdSolverExchange::PublishA(void* ptr) {
  return PublishPointer(a_ipc_, "A", ptr);
}

absl::Status MpmdSolverExchange::PublishB(void* ptr) {
  return PublishPointer(b_ipc_, "B", ptr);
}

absl::Status MpmdSolverExchange::PublishWork(void* ptr) {
  return PublishPointer(work_ipc_, "workspace", ptr);
}

absl::StatusOr<void*> MpmdSolverExchange::OpenPointerOnDevice(
    const std::optional<SharedMemoryArray<IpcHandleWithOffset>>& handles,
    const char* label, int device, OpenedIpcPointer* opened) const {
  // Rank 0 calls this for each remote rank when assembling the cuSolverMg
  // pointer arrays. The device argument ensures the remote allocation is opened
  // in the CUDA context cuSolverMg expects for that entry.
  if (!status_.ok()) {
    return status_;
  }
  if (!handles.has_value()) {
    return absl::InternalError(
        absl::StrFormat("MPMD solver exchange has no %s handle array", label));
  }
  if (opened == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("MPMD %s open requires an output slot", label));
  }
  absl::StatusOr<OpenedIpcPointer> remote =
      OpenIpcHandleWithOffsetOnDevice((*handles)[device], device);
  if (!remote.ok()) {
    return remote.status();
  }
  *opened = *remote;
  return opened->ptr;
}

absl::StatusOr<void*> MpmdSolverExchange::OpenAOnDevice(
    int device, OpenedIpcPointer* opened) const {
  return OpenPointerOnDevice(a_ipc_, "A", device, opened);
}

absl::StatusOr<void*> MpmdSolverExchange::OpenBOnDevice(
    int device, OpenedIpcPointer* opened) const {
  return OpenPointerOnDevice(b_ipc_, "B", device, opened);
}

absl::StatusOr<void*> MpmdSolverExchange::OpenWorkOnDevice(
    int device, OpenedIpcPointer* opened) const {
  return OpenPointerOnDevice(work_ipc_, "workspace", device, opened);
}

void MpmdSolverExchange::SetLwork(int device, int64_t lwork) {
  (*lwork_)[device] = lwork;
}

int64_t MpmdSolverExchange::Lwork(int device) const {
  return (*lwork_)[device];
}

void MpmdSolverExchange::SetSolverStatus(int device, int32_t solver_status) {
  (*solver_status_)[device] = solver_status;
}

int32_t MpmdSolverExchange::SolverStatus(int device) const {
  return (*solver_status_)[device];
}

absl::Status MpmdSolverExchange::CloseAndUnlink() {
  absl::Status status = absl::OkStatus();
  if (a_ipc_.has_value()) status = FirstError(status, a_ipc_->CloseAndUnlink());
  if (b_ipc_.has_value()) status = FirstError(status, b_ipc_->CloseAndUnlink());
  if (work_ipc_.has_value()) {
    status = FirstError(status, work_ipc_->CloseAndUnlink());
  }
  if (lwork_.has_value()) status = FirstError(status, lwork_->CloseAndUnlink());
  if (solver_status_.has_value()) {
    status = FirstError(status, solver_status_->CloseAndUnlink());
  }
  if (barrier_.has_value()) {
    status = FirstError(status, barrier_->CloseAndUnlink());
  }
  return status;
}

absl::Status CloseOpenedRemoteIpcPointers(
    int num_ranks, int current_device,
    std::initializer_list<OpenedIpcPointer*> opened_sets) {
  for (int dev = 0; dev < num_ranks; ++dev) {
    if (dev == current_device) {
      continue;
    }
    for (OpenedIpcPointer* opened_set : opened_sets) {
      if (opened_set == nullptr) {
        continue;
      }
      absl::Status status = CloseIpcPointer(opened_set[dev]);
      if (!status.ok()) {
        return status;
      }
    }
  }
  return absl::OkStatus();
}

template class SharedMemoryArray<IpcHandleWithOffset>;
template class SharedMemoryArray<int64_t>;
template class SharedMemoryArray<int32_t>;

}  // namespace xla::gpu
