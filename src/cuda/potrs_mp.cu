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
#include <unistd.h>
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
        ffi::Error PotrsMgImpl(int64_t N, int64_t NRHS, int64_t batch_a,
                               gpuStream_t stream, ffi::ScratchAllocator &scratch,
                               ffi::AnyBuffer a, ffi::AnyBuffer b, int64_t tile_size,
                               ffi::Result<ffi::AnyBuffer> out_a,
                               ffi::Result<ffi::AnyBuffer> out_b,
                               ffi::Result<ffi::Buffer<ffi::S32>> status)
        {
            /* misc */
            const std::string &source = __FILE__; // file name for error messages

            /* GPU */
            const int MAX_NUM_DEVICES = 16; // cusolverMg can handle 16 GPUs at most
            int nbGpus = 0;                 // number of GPUs to use
            int currentDevice = 0;          // current GPU
            CUDA_CHECK_OR_RETURN(cudaGetDeviceCount(&nbGpus));
            CUDA_CHECK_OR_RETURN(cudaGetDevice(&currentDevice));

            if (nbGpus > MAX_NUM_DEVICES)
            {
                return ffi::Error::InvalidArgument(
                    absl::StrFormat("%s: Number of Gpus must be <=16, received %d", source, nbGpus));
            }
            std::vector<int> deviceList(nbGpus); // list of device IDs

            /* data */
            auto array_data_A = static_cast<data_type *>(a.untyped_data()); // XLA device pointer for a
            auto array_data_b = static_cast<data_type *>(b.untyped_data());
            auto out_data_a = static_cast<data_type *>(out_a->untyped_data());
            auto out_data_b = static_cast<data_type *>(out_b->untyped_data());

            /* Tiling sizes */
            const int IA = 1; // index within a global matrix, base-1 (not used)
            const int JA = 1;
            const int T_A = std::min(tile_size, batch_a); // tile size of A

            const int IB = 1; // index within a global matrix, base-1 (not used)
            const int JB = 1;

            if (NRHS > 256)
            {
                return ffi::Error::InvalidArgument(
                    absl::StrFormat("%s: Number of right hand sides must be <=256, received %d, "
                                    "this may be improved in the next release",
                                    source, NRHS));
            }
            const int T_B = static_cast<int>(NRHS); // tile size of B

            /* CUDA */
            cudaDataType compute_type = traits<data_type>::cuda_data_type; // Data type for computation
            cudaLibMgMatrixDesc_t descrA;                                  // CusolverMg matrix descriptors
            cudaLibMgMatrixDesc_t descrB;
            cudaLibMgGrid_t gridA; // CusolverMg grid descriptors
            cudaLibMgGrid_t gridB;
            cusolverMgGridMapping_t mapping = CUDALIBMG_GRID_MAPPING_COL_MAJOR; // Column major a la Scalapack
            cusolverMgHandle_t cusolverH = nullptr;                             // = cusolverHPool.get();
            int info = 0;                                                       // Info used by cusolverMg calls
            cusolverStatus_t cusolver_status;                                   // Return status of cusolverMg calls
            auto status_data = status->typed_data();                            // Status returned by potrf
            int64_t lwork_potrf = 0;                                            // Workspace size used by cusolverMg calls
            int64_t lwork_potrs = 0;

            /* Shared memory */
            const pid_t ppid = getppid();
            const std::string barrier_name = "/jaxmgbarrier_" + std::to_string(static_cast<long long>(ppid));
            const std::string shmAipc_name = absl::StrFormat("/jaxmg_potrs_mp_shmAipc_%d", ppid);
            const std::string shmoffsetA_name = absl::StrFormat("/jaxmg_potrs_mp_shmoffsetA_%d", ppid);
            const std::string shmBipc_name = absl::StrFormat("/jaxmg_potrs_mp_shmBipc_%d", ppid);
            const std::string shmoffsetB_name = absl::StrFormat("/jaxmg_potrs_mp_shmoffsetB_%d", ppid);
            const std::string shmoutdataaipc_name = absl::StrFormat("/jaxmg_potrs_mp_shmoutdataaipc_%d", ppid);
            const std::string shmoffsetoutdataa_name = absl::StrFormat("/jaxmg_potrs_mp_shmoffsetoutdataa_%d", ppid);
            const std::string shmoutdatabipc_name = absl::StrFormat("/jaxmg_potrs_mp_shmoutdatabipc_%d", ppid);
            const std::string shmoffsetoutdatab_name = absl::StrFormat("/jaxmg_potrs_mp_shmoffsetoutdatab_%d", ppid);
            const std::string shmworkipc_name = absl::StrFormat("/jaxmg_potrs_mp_shmworkipc_%d", ppid);
            const std::string shmoffsetwork_name = absl::StrFormat("/jaxmg_potrs_mp_shmoffsetwork_%d", ppid);
            const std::string shmlwork_name = absl::StrFormat("/jaxmg_potrs_mp_shmlwork_%d", ppid);
            const std::string shmcsh_name = absl::StrFormat("/jaxmg_potrs_mp_shmcsh_%d", ppid);
            DynamicBarrier sync_point(nbGpus, barrier_name.c_str());
            sync_point.arrive_and_wait();
            CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());

            sharedMemoryInfo shminfoAipc;            // Shared memory info for device pointers to local matrices
            sharedMemoryInfo shminfo_offsetA;        // Shared memory info for offsets of A pointers
            sharedMemoryInfo shminfoBipc;            // Shared memory info for device pointers to b matrices
            sharedMemoryInfo shminfo_offsetB;        // Shared memory info for offsets of b pointers
            sharedMemoryInfo shminfooutdataaipc;     // Shared memory info for device pointers to out_a matrices
            sharedMemoryInfo shminfooutdatabipc;     // Shared memory info for device pointers to out_b matrices
            sharedMemoryInfo shminfo_offsetoutdataa; // Shared memory info for offsets of out_a pointers
            sharedMemoryInfo shminfo_offsetoutdatab; // Shared memory info for offsets of out_b pointers
            sharedMemoryInfo shminfoworkipc;         // Shared memory info for workspace pointers
            sharedMemoryInfo shminfo_offsetwork;     // Shared memory info for offsets of workspace pointers
            sharedMemoryInfo shminfolwork;           // Shared memory info for lwork
            sharedMemoryInfo shminfo_csh;            // Shared memory info for cusolver status

            // Data handles A
            std::vector<data_type *> shmA(nbGpus, nullptr);
            IpcOpenResult<data_type> opened_ptrs_A;
            cudaIpcMemHandle_t *shmAipc = get_shm_ipc_handles(currentDevice, sync_point, shminfoAipc, shmAipc_name.c_str());
            uintptr_t *shmoffsetA = get_shm_lwork_ptr<uintptr_t>(currentDevice, sync_point, shminfo_offsetA, shmoffsetA_name.c_str());
            // Data handles b
            std::vector<data_type *> shmB(nbGpus, nullptr);
            IpcOpenResult<data_type> opened_ptrs_B;
            cudaIpcMemHandle_t *shmBipc = get_shm_ipc_handles(currentDevice, sync_point, shminfoBipc, shmBipc_name.c_str());
            uintptr_t *shmoffsetB = get_shm_lwork_ptr<uintptr_t>(currentDevice, sync_point, shminfo_offsetB, shmoffsetB_name.c_str());
            // Data handles out_data_a
            std::vector<data_type *> shmoutdataa(nbGpus, nullptr);
            IpcOpenResult<data_type> opened_ptrs_outdataa;
            cudaIpcMemHandle_t *shmoutdataaipc = get_shm_ipc_handles(currentDevice, sync_point, shminfooutdataaipc, shmoutdataaipc_name.c_str());
            uintptr_t *shmoffsetoutdataa = get_shm_lwork_ptr<uintptr_t>(currentDevice, sync_point, shminfo_offsetoutdataa, shmoffsetoutdataa_name.c_str());
            // Data handles out_data_b
            std::vector<data_type *> shmoutdatab(nbGpus, nullptr);
            IpcOpenResult<data_type> opened_ptrs_outdatab;
            cudaIpcMemHandle_t *shmoutdatabipc = get_shm_ipc_handles(currentDevice, sync_point, shminfooutdatabipc, shmoutdatabipc_name.c_str());
            uintptr_t *shmoffsetoutdatab = get_shm_lwork_ptr<uintptr_t>(currentDevice, sync_point, shminfo_offsetoutdatab, shmoffsetoutdatab_name.c_str());
            // Data handles shmwork
            std::vector<data_type *> shmwork(nbGpus, nullptr);
            IpcOpenResult<data_type> opened_ptrs_work;
            cudaIpcMemHandle_t *shmworkipc = get_shm_ipc_handles(currentDevice, sync_point, shminfoworkipc, shmworkipc_name.c_str());
            uintptr_t *shmoffsetwork = get_shm_lwork_ptr<uintptr_t>(currentDevice, sync_point, shminfo_offsetwork, shmoffsetwork_name.c_str());

            int32_t *cusolver_status_host = get_shm_lwork_ptr<int32_t>(currentDevice, sync_point, shminfo_csh, shmcsh_name.c_str());
            int64_t *shmlwork = get_shm_lwork_ptr<int64_t>(currentDevice, sync_point, shminfolwork, shmlwork_name.c_str());

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
                CUSOLVER_CHECK_OR_RETURN(cusolverMgCreateDeviceGrid(&gridB, 1, nbGpus, deviceList.data(), mapping));

                /* (global) A is N-by-N */
                CUSOLVER_CHECK_OR_RETURN(cusolverMgCreateMatrixDesc(&descrA, N, /* number of rows of (global) A */
                                                                    N,          /* number of columns of (global) A */
                                                                    N,          /* number or rows in a tile */
                                                                    T_A,        /* number of columns in a tile */
                                                                    compute_type, gridA));

                /* (global) B is N-by-NRHS */
                CUSOLVER_CHECK_OR_RETURN(cusolverMgCreateMatrixDesc(&descrB, N, /* number of rows of (global) B */
                                                                    NRHS,       /* number of columns of (global) B */
                                                                    N,          /* number or rows in a tile */
                                                                    T_B,        /* number of columns in a tile */
                                                                    compute_type, gridB));
            }

            CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            sync_point.arrive_and_wait();
            ipcGetHandleAndOffset(array_data_A, shmAipc[currentDevice], shmoffsetA[currentDevice]);
            ipcGetHandleAndOffset(array_data_b, shmBipc[currentDevice], shmoffsetB[currentDevice]);
            ipcGetHandleAndOffset(out_data_a, shmoutdataaipc[currentDevice], shmoffsetoutdataa[currentDevice]);
            ipcGetHandleAndOffset(out_data_b, shmoutdatabipc[currentDevice], shmoffsetoutdatab[currentDevice]);

            CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            sync_point.arrive_and_wait();

            // Gather all device pointers on rank 0
            if (currentDevice == 0)
            {
                opened_ptrs_A = ipcGetDevicePointers<data_type>(currentDevice, nbGpus, shmAipc, shmoffsetA);
                opened_ptrs_B = ipcGetDevicePointers<data_type>(currentDevice, nbGpus, shmBipc, shmoffsetB);
                opened_ptrs_outdataa = ipcGetDevicePointers<data_type>(currentDevice, nbGpus, shmoutdataaipc, shmoffsetoutdatab);
                opened_ptrs_outdatab = ipcGetDevicePointers<data_type>(currentDevice, nbGpus, shmoutdatabipc, shmoffsetoutdatab);

                for (int dev = 1; dev < nbGpus; ++dev)
                {
                    shmA[dev] = opened_ptrs_A.ptrs[dev];
                    shmB[dev] = opened_ptrs_B.ptrs[dev];
                    shmoutdataa[dev] = opened_ptrs_outdataa.ptrs[dev];
                    shmoutdatab[dev] = opened_ptrs_outdatab.ptrs[dev];
                }
                shmA[0] = array_data_A;
                shmB[0] = array_data_b;
                shmoutdataa[0] = out_data_a;
                shmoutdatab[0] = out_data_b;
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
                CUSOLVER_CHECK_OR_RETURN(cusolverMgPotrf_bufferSize(cusolverH, CUBLAS_FILL_MODE_LOWER, N,
                                                                    //   reinterpret_cast<void **>(array_d_A.data()), IA, /* base-1 */
                                                                    reinterpret_cast<void **>(shmA.data()), IA, /* base-1 */
                                                                    JA,                                         /* base-1 */
                                                                    descrA, compute_type, &lwork_potrf));
                CUSOLVER_CHECK_OR_RETURN(cusolverMgPotrs_bufferSize(cusolverH, CUBLAS_FILL_MODE_LOWER, N, NRHS, /* NRHS */
                                                                                                                //   reinterpret_cast<void **>(array_d_A.data()), IA, JA,
                                                                    reinterpret_cast<void **>(shmA.data()), IA, JA,
                                                                    //   descrA, reinterpret_cast<void **>(array_d_B.data()),
                                                                    descrA, reinterpret_cast<void **>(shmB.data()),
                                                                    IB, JB, descrB, compute_type,
                                                                    &lwork_potrs));

                for (int dev = 0; dev < nbGpus; ++dev)
                {
                    shmlwork[dev] = std::max(lwork_potrf, lwork_potrs);
                };
            }

            sync_point.arrive_and_wait();

            // Assign workspace
            size_t workspace_bytes = sizeof(data_type) * static_cast<size_t>(shmlwork[currentDevice]);

            sync_point.arrive_and_wait();
            CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());

            // Get all workspace pointers on rank 0.
            if (currentDevice == 0)
            {
                workspaceAlloc(nbGpus, deviceList.data(),
                               workspace_bytes, /* number of bytes per device */
                               reinterpret_cast<void **>(shmwork.data()));
            }
            // /* sync all devices */
            CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            sync_point.arrive_and_wait();

            if (currentDevice == 0)
            {
                cusolver_status = cusolverMgPotrf(
                    cusolverH, CUBLAS_FILL_MODE_LOWER, N,
                    reinterpret_cast<void **>(shmA.data()), IA, JA,
                    descrA, compute_type,
                    reinterpret_cast<void **>(shmwork.data()), shmlwork[currentDevice], &info);

                /* sync all devices */
                CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
                // Copy status to all devices
                for (int dev = 0; dev < nbGpus; dev++)
                {
                    cusolver_status_host[dev] = static_cast<int32_t>(cusolver_status);
                }
                /* check if A is singular */
                if (0 > info)
                {
                    return ffi::Error::Internal(
                        absl::StrFormat("unexpected error in cusolverMgPotrf, %d-th input parameter is wrong \n", -info));
                }

                // Check status, if 0, continue with Potrs
                if (cusolver_status_host[0] == 0)
                {
                    cusolver_status = cusolverMgPotrs(cusolverH, CUBLAS_FILL_MODE_LOWER, N, NRHS, /* NRHS */
                                                      reinterpret_cast<void **>(shmA.data()), IA, JA, descrA,
                                                      reinterpret_cast<void **>(shmB.data()), IB, JB, descrB,
                                                      compute_type,
                                                      reinterpret_cast<void **>(shmwork.data()), shmlwork[currentDevice],
                                                      &info);
                    /* sync all devices */
                    CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());

                    for (int dev = 0; dev < nbGpus; dev++)
                    {
                        cusolver_status_host[dev] = static_cast<int32_t>(cusolver_status);
                    }
                    /* check if parameters are valid */
                    if (0 > info)
                    {
                        return ffi::Error::Internal(
                            absl::StrFormat("unexpected error in cusolverMgPotrs, %d-th input parameter is wrong \n", -info));
                    }
                }
            }

            /* sync all devices */
            CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            sync_point.arrive_and_wait();

            if (currentDevice == 0)
            {
                std::vector<typename traits<data_type>::T> host_out(N);
                size_t numBytes = sizeof(data_type) * N;
                CUDA_CHECK_OR_RETURN(cudaMemcpy(host_out.data(), shmB[0], numBytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            }
            // Write status data
            int32_t status_val = static_cast<int32_t>(cusolver_status_host[currentDevice]);
            JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpy(status_data, &status_val, sizeof(status_val), gpuMemcpyHostToDevice));
            // Write solution to all shmBs
            if (currentDevice == 0)
            {
                for (int dev = 0; dev < nbGpus; dev++)
                {
                    JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpy(shmoutdatab[dev], shmB[0], b.size_bytes(), gpuMemcpyDeviceToDevice));
                }
            }
            CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            sync_point.arrive_and_wait();
            // Set output pointers for A
            if (cusolver_status_host[currentDevice] == 0)
            {
                shmoutdataa[currentDevice] = shmA[currentDevice];
            }
            // Fill nans if solver failed
            else
            {
                std::vector<typename traits<data_type>::T> host_nan(N * NRHS, traits<data_type>::nan());
                JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpy(out_data_b, host_nan.data(), sizeof(data_type) * N * NRHS, gpuMemcpyHostToDevice));
            }
            CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            sync_point.arrive_and_wait();
            // Close file descriptors
            sharedMemoryClose(&shminfo_offsetA);
            sharedMemoryClose(&shminfoAipc);
            sharedMemoryClose(&shminfo_offsetB);
            sharedMemoryClose(&shminfoBipc);
            sharedMemoryClose(&shminfo_offsetoutdataa);
            sharedMemoryClose(&shminfo_offsetoutdatab);
            sharedMemoryClose(&shminfooutdataaipc);
            sharedMemoryClose(&shminfooutdatabipc);
            sharedMemoryClose(&shminfoworkipc);
            sharedMemoryClose(&shminfo_offsetwork);
            sharedMemoryClose(&shminfolwork);
            sharedMemoryClose(&shminfo_csh);
            sync_point.arrive_and_wait();
            if (currentDevice == 0)
            {
                CUSOLVER_CHECK_OR_RETURN(cusolverMgDestroyMatrixDesc(descrA));
                CUSOLVER_CHECK_OR_RETURN(cusolverMgDestroyMatrixDesc(descrB));

                CUSOLVER_CHECK_OR_RETURN(cusolverMgDestroyGrid(gridA));
                CUSOLVER_CHECK_OR_RETURN(cusolverMgDestroyGrid(gridB));

                CUSOLVER_CHECK_OR_RETURN(cusolverMgDestroy(cusolverH));
                // Shared memory unlink
                sharedMemoryUnlink(shmoffsetA_name.c_str());
                sharedMemoryUnlink(shmAipc_name.c_str());
                sharedMemoryUnlink(shmoffsetB_name.c_str());
                sharedMemoryUnlink(shmBipc_name.c_str());
                sharedMemoryUnlink(shmoffsetoutdataa_name.c_str());
                sharedMemoryUnlink(shmoffsetoutdatab_name.c_str());
                sharedMemoryUnlink(shmoutdataaipc_name.c_str());
                sharedMemoryUnlink(shmoutdatabipc_name.c_str());
                sharedMemoryUnlink(shmworkipc_name.c_str());
                sharedMemoryUnlink(shmoffsetwork_name.c_str());
                sharedMemoryUnlink(shmlwork_name.c_str());
                sharedMemoryUnlink(shmcsh_name.c_str());
                // Close memory handles
                ipcCloseDevicePointers(currentDevice, opened_ptrs_A.bases, nbGpus);
                ipcCloseDevicePointers(currentDevice, opened_ptrs_B.bases, nbGpus);
                ipcCloseDevicePointers(currentDevice, opened_ptrs_outdataa.bases, nbGpus);
                ipcCloseDevicePointers(currentDevice, opened_ptrs_outdatab.bases, nbGpus);
                workspaceFree(nbGpus, deviceList.data(), reinterpret_cast<void **>(shmwork.data()));
            }
            CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            sync_point.close_and_wait();

            return ffi::Error::Success();
        }

        ffi::Error PotrsMgDispatch(gpuStream_t stream, ffi::ScratchAllocator scratch,
                                   ffi::AnyBuffer a, ffi::AnyBuffer b, int64_t tile_size,
                                   ffi::Result<ffi::AnyBuffer> out_a,
                                   ffi::Result<ffi::AnyBuffer> out_b,
                                   ffi::Result<ffi::Buffer<ffi::S32>> status)
        {
            auto dataType = a.element_type();

            // Rows are batched
            FFI_ASSIGN_OR_RETURN((const auto [batch_a, N]), SplitBatch1D(a.dimensions()));
            FFI_ASSIGN_OR_RETURN((const auto [N_b, NRHS]), SplitBatch1D(b.dimensions()));
            FFI_RETURN_IF_ERROR(CheckShape(b.dimensions(), {N, NRHS}, "b", "potrf"));

            if (dataType != out_a->element_type())
            {
                return ffi::Error::InvalidArgument(
                    "The input and output to potrs must have the same element type");
            }
            if (dataType != out_b->element_type())
            {
                return ffi::Error::InvalidArgument(
                    "The input and output to potrs must have the same element type");
            }
            if (dataType != b.element_type())
            {
                return ffi::Error::InvalidArgument(
                    "The input matrix a and output x of potrs must have the same element type");
            }
            FFI_RETURN_IF_ERROR(CheckShape(status->dimensions(), 1, "status", "potrf"));

            SOLVER_DISPATCH_IMPL(PotrsMgImpl, N, NRHS, batch_a, stream, scratch, a, b, tile_size, out_a, out_b, status);

            return ffi::Error::InvalidArgument(absl::StrFormat(
                "Unsupported data type%s for potrs", absl::FormatStreamed(dataType)));
        }

        XLA_FFI_DEFINE_HANDLER_SYMBOL(PotrsMgMpFFI, PotrsMgDispatch,
                                      ffi::Ffi::Bind()
                                          .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                          .Ctx<ffi::ScratchAllocator>()
                                          .Arg<ffi::AnyBuffer>()        // A
                                          .Arg<ffi::AnyBuffer>()        // b
                                          .Attr<int64_t>("T_A")         // tile size
                                          .Ret<ffi::AnyBuffer>()        // A_out
                                          .Ret<ffi::AnyBuffer>()        // b_out
                                          .Ret<ffi::Buffer<ffi::S32>>() // status
        );

#undef SOLVER_DISPATCH_IMPL

    } // namespace JAX_GPU_NAMESPACE
} // namespace jax