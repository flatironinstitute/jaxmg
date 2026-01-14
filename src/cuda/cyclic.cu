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
// CUDA
#include "third_party/gpus/cuda/include/cusolverMg.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/cuda_runtime.h"
// My Code
#include "utils/jax_utils.h"
#include "cusolver_utils.h"
#include "utils/shm.h"
#include "thread_barrier.h"

ThreadBarrier sync_point;

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
        ffi::Error CyclicMgImpl(int64_t N, int64_t batch_a, gpuStream_t stream, ffi::ScratchAllocator &scratch,
                                ffi::AnyBuffer a, int64_t tile_size, ffi::Result<ffi::AnyBuffer> out)
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
            // auto status_data = status->typed_data();                        // Status returned by potrf
            auto out_data = static_cast<data_type *>(out->untyped_data());

            /* Tiling sizes */
            const int T_A = std::min<int>(tile_size, static_cast<int>(N / nbGpus));
            if (tile_size >= N / nbGpus)
            {
                batch_a = T_A;
            }

            /* Shared memory */
            static std::once_flag barrier_initialized; // Initialize barrier once between threads
            std::call_once(barrier_initialized, [&]()
                           { sync_point.initialize(nbGpus); });
            CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            sync_point.arrive_and_wait();

            sharedMemoryInfo shminfoA; // Shared memory info for device pointers to local matrices
            sharedMemoryInfo shminfowork;
            sharedMemoryInfo shminfolwork; // Shared memory info for lwork space nbytes
            sharedMemoryInfo shmcsh;       // Shared memory info for cusolver status

            data_type **shmA = get_shm_device_ptrs<data_type>(currentDevice, sync_point, shminfoA, "shmA"); // Actual shared memory
            data_type **shmwork = get_shm_device_ptrs<data_type>(currentDevice, sync_point, shminfowork, "shmwork");

            int32_t *cusolver_status_host = get_shm_lwork_ptr<int32_t, ThreadBarrier>(currentDevice, sync_point, shmcsh, "shmcsh");
            int64_t *shmlwork = get_shm_lwork_ptr<int64_t, ThreadBarrier>(currentDevice, sync_point, shminfolwork, "shmlwork");

            if (currentDevice == 0)
            {
                for (int j = 0; j < nbGpus; j++)
                {
                    deviceList[j] = j;
                }
            }

            CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            sync_point.arrive_and_wait();

            shmA[currentDevice] = array_data_A;
            if (g_cusolver_utils_verbose)
            {
                std::vector<typename traits<data_type>::T> host(N * batch_a);
                size_t numBytes = sizeof(data_type) * N * batch_a;

                CUDA_CHECK_OR_RETURN(cudaMemcpy(host.data(), shmA[currentDevice], numBytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());

                printf("Dev %d\n", currentDevice);
                print_matrix(N, batch_a, host.data(), N);
            }
            sync_point.arrive_and_wait();

            if (currentDevice == 0)
            {
                memcpyCyclicShard<data_type>(nbGpus, stream, deviceList.data(),
                                             N, batch_a, T_A,
                                             /* input */
                                             shmA, false);
            }

            /* sync all devices */
            CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            sync_point.arrive_and_wait();

            if (g_cusolver_utils_verbose)
            {
                std::vector<typename traits<data_type>::T> host(N * batch_a);
                size_t numBytes = sizeof(data_type) * N * batch_a;
                CUDA_CHECK_OR_RETURN(cudaMemcpy(host.data(), shmA[currentDevice], numBytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
                printf("Dev %d\n", currentDevice);
                print_matrix(N, batch_a, host.data(), N);
            }
            out_data = shmA[currentDevice];
            return ffi::Error::Success();
        }

        ffi::Error CyclicMgDispatch(gpuStream_t stream, ffi::ScratchAllocator scratch,
                                    ffi::AnyBuffer a, int64_t tile_size, ffi::Result<ffi::AnyBuffer> out)
        {
            auto dataType = a.element_type();

            // Batching the rows of a symmetric matrix with row-major we get col-major columns.
            FFI_ASSIGN_OR_RETURN((const auto [batch_a, N]), SplitBatch1D(a.dimensions()));

            SOLVER_DISPATCH_IMPL(CyclicMgImpl, N, batch_a, stream, scratch, a, tile_size, out);

            return ffi::Error::InvalidArgument(absl::StrFormat(
                "Unsupported data type%s for potrs", absl::FormatStreamed(dataType)));
        }

        XLA_FFI_DEFINE_HANDLER_SYMBOL(CyclicMgFFI, CyclicMgDispatch,
                                      ffi::Ffi::Bind()
                                          .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                          .Ctx<ffi::ScratchAllocator>()
                                          .Arg<ffi::AnyBuffer>() // A
                                          .Attr<int64_t>("T_A")  // tile size
                                          .Ret<ffi::AnyBuffer>() // x

        );

#undef SOLVER_DISPATCH_IMPL

    } // namespace JAX_GPU_NAMESPACE
} // namespace jax