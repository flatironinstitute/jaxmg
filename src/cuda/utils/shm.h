
#ifndef SHM_MULTIPROCESS_H
#define SHM_MULTIPROCESS_H

// C++
#include <vector>
#include <barrier>
#include <memory>
#include <cstddef>
#include <cstdint>
// CUDA
#include <cuda_runtime.h>
// Own code
#include "process_barrier.h"
#include "thread_barrier.h"

#define MAX_DEVICES 16

typedef struct sharedMemoryInfo_st
{
    void *addr;
    size_t size;
    int shmFd;
} sharedMemoryInfo;

int sharedMemoryCreate(const char *name, size_t sz, sharedMemoryInfo *info);

int sharedMemoryOpen(const char *name, size_t sz, sharedMemoryInfo *info);

void sharedMemoryClose(sharedMemoryInfo *info);

void sharedMemoryUnlink(const char *name);

template <typename T>
T **get_shm_device_ptrs(int currentDevice, ThreadBarrier &sync_point, sharedMemoryInfo &info, const char *shmName);

template <typename T, typename barrier>
T *get_shm_lwork_ptr(int currentDevice, barrier &sync_point, sharedMemoryInfo &info, const char *shmName);

cudaIpcMemHandle_t *get_shm_ipc_handles(int currentDevice, DynamicBarrier &sync_point, sharedMemoryInfo &info, const char *shmName);

typedef struct shmStruct_st
{
    size_t nprocesses;
    int barrier;
    int sense;
    int devices[16];
    cudaIpcMemHandle_t memHandle[16];
    cudaIpcEventHandle_t eventHandle[16];
} shmStruct;

#endif // SHM_MULTIPROCESS_H
