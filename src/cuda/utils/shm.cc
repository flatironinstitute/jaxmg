// Own code
#include "utils/shm.h"
// C++
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#include <barrier>
// JAXlib
#include "jaxlib/gpu/vendor.h"
// CUDA
#include <third_party/gpus/cuda/include/cuComplex.h>
#include "third_party/gpus/cuda/include/cuda_runtime.h"
// Own code
#include "process_barrier.h"
#include "thread_barrier.h"

int sharedMemoryCreate(const char *name, size_t sz, sharedMemoryInfo *info)
{

  int status = 0;

  info->size = sz;
  /* Defensive initialization in case caller didn't initialize the struct. */
  info->shmFd = -1;
  info->addr = nullptr;

  /* Open or create the shared memory object. */
  info->shmFd = shm_open(name, O_RDWR | O_CREAT, 0777);
  if (info->shmFd < 0)
  {
    int err = errno;
    fprintf(stderr, "sharedMemoryCreate: shm_open('%s') failed: %s (errno=%d)\n",
            name, strerror(err), err);
    return err;
  }

  /* Resize. If ftruncate fails, close the fd to avoid leaking it. */
  status = ftruncate(info->shmFd, sz);
  if (status != 0)
  {
    int err = errno;
    close(info->shmFd);
    info->shmFd = -1;
    return err;
  }

  /* Map the region -- check against MAP_FAILED (not NULL). If mmap fails, close the fd. */
  info->addr = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shmFd, 0);
  if (info->addr == MAP_FAILED)
  {
    int err = errno;
    fprintf(stderr, "sharedMemoryCreate: mmap('%s', %zu) failed: %s (errno=%d)\n",
            name, sz, strerror(err), err);
    close(info->shmFd);
    info->shmFd = -1;
    info->addr = nullptr;
    return err;
  }

  return 0;
}

int sharedMemoryOpen(const char *name, size_t sz, sharedMemoryInfo *info)
{

  info->size = sz;
  /* Defensive initialization in case caller didn't initialize the struct. */
  info->shmFd = -1;
  info->addr = nullptr;

  info->shmFd = shm_open(name, O_RDWR, 0777);
  if (info->shmFd < 0)
  {
    int err = errno;
    fprintf(stderr, "sharedMemoryOpen: shm_open('%s') failed: %s (errno=%d)\n",
            name, strerror(err), err);
    return err;
  }

  info->addr = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shmFd, 0);
  if (info->addr == MAP_FAILED)
  {
    int err = errno;
    fprintf(stderr, "sharedMemoryOpen: mmap('%s', %zu) failed: %s (errno=%d)\n",
            name, sz, strerror(err), err);
    /* Close fd on error to avoid leaks. */
    close(info->shmFd);
    info->shmFd = -1;
    info->addr = nullptr;
    return err;
  }

  return 0;
}

void sharedMemoryClose(sharedMemoryInfo *info)
{

  /* munmap expects a valid mapping; mmap returns MAP_FAILED on error. */
  if (info->addr && info->addr != MAP_FAILED)
  {
    void *addr = info->addr;
    size_t sz = info->size;
    if (munmap(addr, sz) != 0)
    {
      int err = errno;
      fprintf(stderr, "sharedMemoryClose: munmap(%p, %zu) failed: %s (errno=%d)\n",
              addr, sz, strerror(err), err);
    }
    info->addr = nullptr;
  }

  /* File descriptor 0 is valid (stdin), so check >= 0. Clear it after close to avoid double-closing. */
  if (info->shmFd >= 0)
  {
    close(info->shmFd);
    info->shmFd = -1;
  }
}

void sharedMemoryUnlink(const char *name)
{
  if (shm_unlink(name) != 0)
  {
    int err = errno;
    if (err != ENOENT)
    {
      fprintf(stderr, "sharedMemoryUnlink: shm_unlink('%s') failed: %s (errno=%d)\n",
              name, strerror(err), err);
    }
  }
}

template <typename T>
T **get_shm_device_ptrs(int currentDevice, ThreadBarrier &sync_point, sharedMemoryInfo &info, const char *shmName)
{
  T **shm = nullptr;
  pid_t pid = getppid();
  const int MAX_NUM_DEVICES = 16;

  size_t shmSize = MAX_NUM_DEVICES * sizeof(T *);

  if (currentDevice == 0)
  {
    if (sharedMemoryCreate(shmName, shmSize, &info) != 0)
    {
      printf("Failed to create shared memory '%s'\n", shmName);
      exit(EXIT_FAILURE); // #TODO: replace this with proper JAX error handling
    }
    shm = (T **)info.addr;
    memset((void *)shm, 0, shmSize);
  }

  sync_point.arrive_and_wait(); // Barrier sync

  if (currentDevice != 0)
  {
    if (sharedMemoryOpen(shmName, shmSize, &info) != 0)
    {
      printf("Failed to open shared memory in dev %d\n", currentDevice);
      exit(EXIT_FAILURE);
    }
    shm = (T **)info.addr;
  }

  sync_point.arrive_and_wait();

  return shm;
}

template float **get_shm_device_ptrs<float>(int, ThreadBarrier &, sharedMemoryInfo &, const char *);
template double **get_shm_device_ptrs<double>(int, ThreadBarrier &, sharedMemoryInfo &, const char *);
template cuFloatComplex **get_shm_device_ptrs<cuFloatComplex>(int, ThreadBarrier &, sharedMemoryInfo &, const char *);
template cuDoubleComplex **get_shm_device_ptrs<cuDoubleComplex>(int, ThreadBarrier &, sharedMemoryInfo &, const char *);

template <typename T, typename barrier>
T *get_shm_lwork_ptr(int currentDevice, barrier &sync_point, sharedMemoryInfo &info, const char *shmName)
{
  pid_t pid = getppid();
  const int MAX_NUM_DEVICES = 16;

  size_t shmSize = MAX_NUM_DEVICES * sizeof(T);
  T *shm = nullptr;
  if (currentDevice == 0)
  {
    if (sharedMemoryCreate(shmName, shmSize, &info) != 0)
    {
      printf("Failed to create shared memory\n");
      exit(EXIT_FAILURE); // #TODO: Replace this with proper JAX error handling
    }
    shm = reinterpret_cast<T *>(info.addr);
    ;
    memset(shm, 0, shmSize);
  }

  sync_point.arrive_and_wait(); // Barrier sync

  if (currentDevice != 0)
  {
    if (sharedMemoryOpen(shmName, shmSize, &info) != 0)
    {
      printf("Failed to open shared memory '%s'\n", shmName);
      exit(EXIT_FAILURE);
    }
    shm = reinterpret_cast<T *>(info.addr);
  }

  sync_point.arrive_and_wait();

  return shm;
}

template int64_t *get_shm_lwork_ptr<int64_t, DynamicBarrier>(int, DynamicBarrier &, sharedMemoryInfo &, const char *);
template int32_t *get_shm_lwork_ptr<int32_t, DynamicBarrier>(int, DynamicBarrier &, sharedMemoryInfo &, const char *);
template uintptr_t *get_shm_lwork_ptr<uintptr_t, DynamicBarrier>(int, DynamicBarrier &, sharedMemoryInfo &, const char *);

template int64_t *get_shm_lwork_ptr<int64_t, ThreadBarrier>(int, ThreadBarrier &, sharedMemoryInfo &, const char *);
template int32_t *get_shm_lwork_ptr<int32_t, ThreadBarrier>(int, ThreadBarrier &, sharedMemoryInfo &, const char *);
template uintptr_t *get_shm_lwork_ptr<uintptr_t, ThreadBarrier>(int, ThreadBarrier &, sharedMemoryInfo &, const char *);

cudaIpcMemHandle_t *get_shm_ipc_handles(int currentDevice, DynamicBarrier &sync_point, sharedMemoryInfo &info, const char *shmName)
{
  const int MAX_NUM_DEVICES = 16;

  // Use a stable, shared name (not PID)
  size_t sz = sizeof(cudaIpcMemHandle_t) * MAX_NUM_DEVICES; // nbGpus must be visible here

  if (currentDevice == 0)
  {
    int rc = sharedMemoryCreate(shmName, sz, &info);
    if (rc)
    {

      printf("Failed to open shared memory IPC\n");
      std::abort();
    }
    memset(info.addr, 0, sz);
  }

  sync_point.arrive_and_wait();

  if (currentDevice != 0)
  {
    int rc = sharedMemoryOpen(shmName, sz, &info);
    if (rc)
    {
      printf("Failed to open shared memory IPC\n");
      std::abort();
    }
  }

  sync_point.arrive_and_wait();
  return static_cast<cudaIpcMemHandle_t *>(info.addr);
}
