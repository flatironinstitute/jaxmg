/*
 * Copyright 2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

// C++
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <mutex>
#include <string>
#include <cstdio>
#include <iostream>
// Abseil
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
// Jaxlib
#include "jaxlib/ffi_helpers.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/vendor.h"
// XLA
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/api/c_api.h"
// CUDA
#include "third_party/gpus/cuda/include/cusolverMg.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/cuda_runtime.h"
#include "third_party/gpus/cuda/include/cuda.h"
// My Code
#include "utils/jax_utils.h"
#include "cusolver_utils.h"
#include "process_barrier.h"
#include "utils/shm.h"
#include "utils/ipc_utils.h"

namespace jax
{
    namespace JAX_GPU_NAMESPACE
    {
        namespace ffi = ::xla::ffi;

#define SOLVER_DISPATCH_IMPL(impl, ...)            \
    switch (dataType)                              \
    {                                              \
    case ffi::F32:                                 \
        return impl<float>(__VA_ARGS__);           \
    case ffi::F64:                                 \
        return impl<double>(__VA_ARGS__);          \
    case ffi::C64:                                 \
        return impl<cuFloatComplex>(__VA_ARGS__);  \
    case ffi::C128:                                \
        return impl<cuDoubleComplex>(__VA_ARGS__); \
    default:                                       \
        break;                                     \
    }

        template <typename data_type>
        ffi::Error SyevdMgImplMp(int64_t N, int64_t batch_a,
                                 gpuStream_t stream, ffi::ScratchAllocator &scratch,
                                 ffi::AnyBuffer a, int64_t tile_size,
                                 ffi::Result<ffi::AnyBuffer> eigenvalues,
                                 ffi::Result<ffi::AnyBuffer> V,
                                 ffi::Result<ffi::Buffer<ffi::S32>> status)
        {
            /* misc */
            const std::string &source = __FILE__;

            /* GPU */
            const int MAX_NUM_DEVICES = 16;
            int nbGpus = 0;
            int currentDevice = 0;
            CUDA_CHECK_OR_RETURN(cudaGetDeviceCount(&nbGpus));
            CUDA_CHECK_OR_RETURN(cudaGetDevice(&currentDevice));
            if (nbGpus > MAX_NUM_DEVICES)
            {
                return ffi::Error::InvalidArgument(
                    absl::StrFormat("%s: Number of Gpus must be <=16, received %d", source, nbGpus));
            }
            std::vector<int> deviceList(nbGpus);

            /* data */
            auto array_data_A = static_cast<data_type *>(a.untyped_data());
            auto array_data_eigenvalues = static_cast<data_type *>(eigenvalues->untyped_data());
            auto array_data_V = static_cast<data_type *>(V->untyped_data());
            std::vector<typename traits<data_type>::S> eigenvalues_host(N, 0.);

            /* Tiling sizes */
            const int IA = 1;
            const int JA = 1;
            const int T_A = std::min(tile_size, batch_a);

            /* CUDA */
            cudaDataType compute_type = traits<data_type>::cuda_data_type;
            cudaDataType eigenvalue_type = traits<typename traits<data_type>::S>::cuda_data_type;
            cudaLibMgMatrixDesc_t descrA;
            cudaLibMgGrid_t gridA;
            cusolverMgGridMapping_t mapping = CUDALIBMG_GRID_MAPPING_COL_MAJOR;
            cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
            cusolverMgHandle_t cusolverH = nullptr;
            int info = 0;
            cusolverStatus_t cusolver_status;
            auto status_data = status->typed_data();
            int64_t lwork_syevd = 0;

            /* Shared memory (multi-process barrier) */
            const pid_t ppid = getppid();
            const std::string barrier_name = "/jaxmgbarrier_" + std::to_string(static_cast<long long>(ppid));
            DynamicBarrier sync_point(nbGpus, barrier_name.c_str());
            sync_point.arrive_and_wait();
            CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());

            sharedMemoryInfo shminfoAipc;        // Shared memory info for device pointers to local matrices A
            sharedMemoryInfo shminfo_offsetA;    // Shared memory info for offsets of A pointers
            sharedMemoryInfo shminfoevipc;       // Shared memory info for device pointers to eigenvalue arrays
            sharedMemoryInfo shminfo_offsetev;   // Shared memory info for offsets of eigenvalue pointers
            sharedMemoryInfo shminfoVipc;        // Shared memory info for device pointers to eigenvector matrices
            sharedMemoryInfo shminfo_offsetV;    // Shared memory info for offsets of eigenvector pointers
            sharedMemoryInfo shminfoworkipc;     // Shared memory info for workspace pointers
            sharedMemoryInfo shminfo_offsetwork; // Shared memory info for offsets of workspace pointers
            sharedMemoryInfo shminfolwork;       // Shared memory info for lwork
            sharedMemoryInfo shminfo_csh;        // Shared memory info for cusolver status

            // Data handles A
            std::vector<data_type *> shmA(nbGpus, nullptr);
            IpcOpenResult<data_type> opened_ptrs_A;
            cudaIpcMemHandle_t *shmAipc = get_shm_ipc_handles(currentDevice, sync_point, shminfoAipc, "shmAipc");
            uintptr_t *shmoffsetA = get_shm_lwork_ptr<uintptr_t>(currentDevice, sync_point, shminfo_offsetA, "shmoffsetA");

            // Data handles eigenvalues
            std::vector<data_type *> shmev(nbGpus, nullptr);
            IpcOpenResult<data_type> opened_ptrs_ev;
            cudaIpcMemHandle_t *shmevipc = get_shm_ipc_handles(currentDevice, sync_point, shminfoevipc, "shmevipc");
            uintptr_t *shmoffsetev = get_shm_lwork_ptr<uintptr_t>(currentDevice, sync_point, shminfo_offsetev, "shmoffsetev");

            // Data handles V (eigenvectors)
            std::vector<data_type *> shmV(nbGpus, nullptr);
            IpcOpenResult<data_type> opened_ptrs_V;
            cudaIpcMemHandle_t *shmVipc = get_shm_ipc_handles(currentDevice, sync_point, shminfoVipc, "shmVipc");
            uintptr_t *shmoffsetV = get_shm_lwork_ptr<uintptr_t>(currentDevice, sync_point, shminfo_offsetV, "shmoffsetV");

            // Workspace
            std::vector<data_type *> shmwork(nbGpus, nullptr);
            IpcOpenResult<data_type> opened_ptrs_work;
            cudaIpcMemHandle_t *shmworkipc = get_shm_ipc_handles(currentDevice, sync_point, shminfoworkipc, "shmworkipc");
            uintptr_t *shmoffsetwork = get_shm_lwork_ptr<uintptr_t>(currentDevice, sync_point, shminfo_offsetwork, "shmoffsetwork");

            int32_t *cusolver_status_host = get_shm_lwork_ptr<int32_t>(currentDevice, sync_point, shminfo_csh, "shmcsh");
            int64_t *shmlwork = get_shm_lwork_ptr<int64_t>(currentDevice, sync_point, shminfolwork, "shmlwork");

            if (currentDevice == 0)
            {
                CUSOLVER_CHECK_OR_RETURN(cusolverMgCreate(&cusolverH));
                for (int j = 0; j < nbGpus; j++)
                {
                    deviceList[j] = j;
                    cudaDeviceProp prop;
                    CUDA_CHECK_OR_RETURN(cudaGetDeviceProperties(&prop, j));
                }

                CUSOLVER_CHECK_OR_RETURN(cusolverMgDeviceSelect(cusolverH, nbGpus, deviceList.data()));
                CUSOLVER_CHECK_OR_RETURN(cusolverMgCreateDeviceGrid(&gridA, 1, nbGpus, deviceList.data(), mapping));

                CUSOLVER_CHECK_OR_RETURN(cusolverMgCreateMatrixDesc(&descrA, N,
                                                                    N,
                                                                    N,
                                                                    T_A,
                                                                    compute_type, gridA));
            }

            CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            sync_point.arrive_and_wait();

            // Export local device pointers via CUDA IPC
            ipcGetHandleAndOffset(array_data_A, shmAipc[currentDevice], shmoffsetA[currentDevice]);
            ipcGetHandleAndOffset(array_data_eigenvalues, shmevipc[currentDevice], shmoffsetev[currentDevice]);
            ipcGetHandleAndOffset(array_data_V, shmVipc[currentDevice], shmoffsetV[currentDevice]);

            CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            sync_point.arrive_and_wait();

            if (currentDevice == 0)
            {
                opened_ptrs_A = ipcGetDevicePointers<data_type>(currentDevice, nbGpus, shmAipc, shmoffsetA);
                opened_ptrs_ev = ipcGetDevicePointers<data_type>(currentDevice, nbGpus, shmevipc, shmoffsetev);
                opened_ptrs_V = ipcGetDevicePointers<data_type>(currentDevice, nbGpus, shmVipc, shmoffsetV);

                for (int dev = 1; dev < nbGpus; ++dev)
                {
                    shmA[dev] = opened_ptrs_A.ptrs[dev];
                    shmev[dev] = opened_ptrs_ev.ptrs[dev];
                    shmV[dev] = opened_ptrs_V.ptrs[dev];
                }
                shmA[0] = array_data_A;
                shmev[0] = array_data_eigenvalues;
                shmV[0] = array_data_V;
            }

            CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            sync_point.arrive_and_wait();

            if (currentDevice == 0)
            {
                // Convert input layout to 1D block-cyclic layout across devices
                memcpyCyclicShard<data_type>(nbGpus, stream, deviceList.data(),
                                             N, batch_a, T_A,
                                             /* input */
                                             shmA.data(), false);
            }
            CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            sync_point.arrive_and_wait();

            if (currentDevice == 0)
            {
                CUSOLVER_CHECK_OR_RETURN(cusolverMgSyevd_bufferSize(cusolverH, jobz, CUBLAS_FILL_MODE_LOWER, N,
                                                                    reinterpret_cast<void **>(shmA.data()), IA, JA, descrA,
                                                                    reinterpret_cast<void *>(eigenvalues_host.data()),
                                                                    eigenvalue_type, compute_type,
                                                                    &lwork_syevd));

                for (int dev = 0; dev < nbGpus; dev++)
                {
                    shmlwork[dev] = lwork_syevd;
                }
            }

            sync_point.arrive_and_wait();
            CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());

            // Workspace size in bytes for current device
            size_t workspace_bytes = sizeof(data_type) * static_cast<size_t>(shmlwork[currentDevice]);

            sync_point.arrive_and_wait();
            CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());

            if (currentDevice == 0)
            {
                workspaceAlloc(nbGpus, deviceList.data(),
                               workspace_bytes,
                               reinterpret_cast<void **>(shmwork.data()));
            }
            CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            sync_point.arrive_and_wait();

            if (currentDevice == 0)
            {
                cusolver_status = cusolverMgSyevd(
                    cusolverH, jobz, CUBLAS_FILL_MODE_LOWER, N,
                    reinterpret_cast<void **>(shmA.data()), IA, JA,
                    descrA, reinterpret_cast<void **>(eigenvalues_host.data()),
                    eigenvalue_type, compute_type,
                    reinterpret_cast<void **>(shmwork.data()), shmlwork[currentDevice], &info);

                CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
                for (int dev = 0; dev < nbGpus; dev++)
                {
                    cusolver_status_host[dev] = static_cast<int32_t>(cusolver_status);
                }

                if (0 > info)
                {
                    return ffi::Error::Internal(
                        absl::StrFormat("unexpected error in cusolverMgSyevd, %d-th input parameter is wrong \n", -info));
                }
            }

            CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            sync_point.arrive_and_wait();

            // Write status per device
            int32_t status_val = static_cast<int32_t>(cusolver_status_host[currentDevice]);
            JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpy(status_data, &status_val, sizeof(status_val), gpuMemcpyHostToDevice));

            if (currentDevice == 0)
            {
                if (cusolver_status_host[0] == 0)
                {
                    // Copy eigenvalues to device 0 eigenvalue buffer
                    JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpy(
                        shmev[0], eigenvalues_host.data(), sizeof(data_type) * N, gpuMemcpyHostToDevice));
                    CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
                    // Broadcast eigenvalues to all other devices
                    for (int dev = 1; dev < nbGpus; dev++)
                    {
                        JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpy(shmev[dev], shmev[0], sizeof(data_type) * N, gpuMemcpyDeviceToDevice));
                    }
                }
            }
            CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            sync_point.arrive_and_wait();

            if (cusolver_status_host[currentDevice] == 0)
            {
                // Unshard eigenvectors from shmA back into V buffers
                if (currentDevice == 0)
                {
                    memcpyCyclicShard<data_type>(nbGpus, stream, deviceList.data(),
                                                 N, batch_a, T_A,
                                                 /* input */
                                                 shmA.data(), true);
                    // After unsharding, shmA[0] should contain the full matrix in device 0 layout.
                    // Set out pointers inplace
                    for (int dev = 0; dev < nbGpus; dev++)
                    {
                        shmV[dev]= shmA[dev];
                    }
                }
            }
            else
            {
                std::vector<typename traits<data_type>::T> host_ev_nan(N, traits<data_type>::nan());
                std::vector<typename traits<data_type>::T> host_V_nan(N * batch_a, traits<data_type>::nan());
                JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpy(array_data_eigenvalues, host_ev_nan.data(), sizeof(data_type) * N, gpuMemcpyHostToDevice));
                JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpy(array_data_V, host_V_nan.data(), sizeof(data_type) * N * batch_a, gpuMemcpyHostToDevice));
            }
            CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            sync_point.arrive_and_wait();

            if (currentDevice == 0)
            {
                CUSOLVER_CHECK_OR_RETURN(cusolverMgDestroyMatrixDesc(descrA));
                CUSOLVER_CHECK_OR_RETURN(cusolverMgDestroyGrid(gridA));
                CUSOLVER_CHECK_OR_RETURN(cusolverMgDestroy(cusolverH));

                sharedMemoryCleanup(&shminfo_offsetA, "shmoffsetA");
                sharedMemoryCleanup(&shminfoAipc, "shmAipc");
                sharedMemoryCleanup(&shminfo_offsetev, "shmoffsetev");
                sharedMemoryCleanup(&shminfoevipc, "shmevipc");
                sharedMemoryCleanup(&shminfo_offsetV, "shmoffsetV");
                sharedMemoryCleanup(&shminfoVipc, "shmVipc");
                sharedMemoryCleanup(&shminfoworkipc, "shmworkipc");
                sharedMemoryCleanup(&shminfo_offsetwork, "shmoffsetwork");
                sharedMemoryCleanup(&shminfolwork, "shmlwork");
                sharedMemoryCleanup(&shminfo_csh, "shmcsh");

                ipcCloseDevicePointers(currentDevice, opened_ptrs_A.bases, nbGpus);
                ipcCloseDevicePointers(currentDevice, opened_ptrs_ev.bases, nbGpus);
                ipcCloseDevicePointers(currentDevice, opened_ptrs_V.bases, nbGpus);
                workspaceFree(nbGpus, deviceList.data(), reinterpret_cast<void **>(shmwork.data()));
            }
            CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            sync_point.close_and_wait();

            return ffi::Error::Success();
        }

        ffi::Error SyevdMgMpDispatch(gpuStream_t stream, ffi::ScratchAllocator scratch,
                                     ffi::AnyBuffer a, int64_t tile_size,
                                     ffi::Result<ffi::AnyBuffer> eigenvalues,
                                     ffi::Result<ffi::AnyBuffer> V,
                                     ffi::Result<ffi::Buffer<ffi::S32>> status)
        {
            auto dataType = a.element_type();

            // Rows are batched
            FFI_ASSIGN_OR_RETURN((const auto [batch_a, N]), SplitBatch1D(a.dimensions()));

            if ((dataType != V->element_type()))
            {
                return ffi::Error::InvalidArgument(
                    "The input matrix and output eigenvector dtype of syevd must have the same element type");
            }
            FFI_RETURN_IF_ERROR(CheckShape(status->dimensions(), 1, "status", "syevd"));

            SOLVER_DISPATCH_IMPL(SyevdMgImplMp, N, batch_a, stream, scratch, a, tile_size, eigenvalues, V, status);

            return ffi::Error::InvalidArgument(absl::StrFormat(
                "Unsupported data type%s for syevd", absl::FormatStreamed(dataType)));
        }

        XLA_FFI_DEFINE_HANDLER_SYMBOL(SyevdMgMpFFI, SyevdMgMpDispatch,
                                      ffi::Ffi::Bind()
                                          .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                          .Ctx<ffi::ScratchAllocator>()
                                          .Arg<ffi::AnyBuffer>()        // A
                                          .Attr<int64_t>("T_A")         // tile size
                                          .Ret<ffi::AnyBuffer>()        // eigenvalues
                                          .Ret<ffi::AnyBuffer>()        // V
                                          .Ret<ffi::Buffer<ffi::S32>>() // status
        );

#undef SOLVER_DISPATCH_IMPL

    } // namespace JAX_GPU_NAMESPACE
} // namespace jax
