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

std::string MpmdSharedName(const char* prefix, int run_id, int node_group_id) {
  return absl::StrFormat("/jaxmg_%s_%d_%d", prefix, run_id, node_group_id);
}

template class SharedMemoryArray<IpcHandleWithOffset>;
template class SharedMemoryArray<int64_t>;
template class SharedMemoryArray<int32_t>;

}  // namespace xla::gpu
