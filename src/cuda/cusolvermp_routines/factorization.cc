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
// Shared Cholesky and LU factorization steps for factor-only and solve paths.
// Inputs already use cuSOLVERMp's local column-major block-cyclic layout.
// Descriptors and persistent factors remain owned by the calling routine.

#include "factorization.h"

#include <cstdlib>

namespace xla::gpu {
namespace {

// Scope-bound temporary storage also releases partial allocations when a
// workspace query, allocation, or CUDA operation prevents factorization.
struct FactorizationWorkspace {
  size_t device_bytes = 0;
  size_t host_bytes = 0;
  void* device = nullptr;
  void* host = nullptr;
  int* info = nullptr;

  ~FactorizationWorkspace() {
    if (device != nullptr) cudaFree(device);
    if (info != nullptr) cudaFree(info);
    if (host != nullptr) std::free(host);
  }

  absl::Status Allocate(cudaStream_t stream, FactorizationStatusFields status) {
    status.device_workspace_kib = SizeToKiBForStatus(device_bytes);
    status.host_workspace_kib = SizeToKiBForStatus(host_bytes);
    if (device_bytes > 0 && cudaMalloc(&device, device_bytes) != cudaSuccess) {
      status.code = kDeviceAllocFailed;
      return absl::OkStatus();
    }
    if (host_bytes > 0) {
      host = std::malloc(host_bytes);
      if (host == nullptr) {
        status.code = kHostAllocFailed;
        return absl::OkStatus();
      }
    }
    if (cudaMalloc(reinterpret_cast<void**>(&info), sizeof(int)) != cudaSuccess) {
      status.code = kDeviceAllocFailed;
      return absl::OkStatus();
    }
    JAXMG_RETURN_IF_CUDA_ERROR(cudaMemsetAsync(info, 0, sizeof(int), stream));
    return absl::OkStatus();
  }

  absl::Status ReadInfo(cudaStream_t stream, int32_t& result) {
    int host_info = -1;
    JAXMG_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
        &host_info, info, sizeof(int), cudaMemcpyDeviceToHost, stream));
    JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(stream));
    result = host_info;
    return absl::OkStatus();
  }
};

}  // namespace

absl::Status RunCholeskyFactorization(
    const CusolverMpApi& api, cusolverMpHandle_t handle,
    cusolverMpMatrixDescriptor_t desc_a, cudaStream_t cuda_stream, int64_t n,
    void* a, cudaDataType_t dtype, FactorizationStatusFields status) {
  // Stage 1: query and allocate POTRF's temporary storage.
  FactorizationWorkspace work;
  cusolverStatus_t solver_status = api.potrf_buffer_size(
      handle, CUBLAS_FILL_MODE_LOWER, n, a, /*ia=*/1, /*ja=*/1, desc_a,
      dtype, &work.device_bytes, &work.host_bytes);
  if (solver_status != CUSOLVER_STATUS_SUCCESS) {
    status.code = kPotrfWorkspaceFailed;
    status.raw_status = static_cast<int32_t>(solver_status);
    return absl::OkStatus();
  }
  JAXMG_RETURN_IF_ERROR(work.Allocate(cuda_stream, status));
  if (status.code != kStatusOk) return absl::OkStatus();
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));

  // Stage 2: factor A in place and preserve numerical failure diagnostics.
  solver_status = api.potrf(
      handle, CUBLAS_FILL_MODE_LOWER, n, a, /*ia=*/1, /*ja=*/1, desc_a,
      dtype, work.device, work.device_bytes, work.host, work.host_bytes,
      work.info);
  if (solver_status != CUSOLVER_STATUS_SUCCESS) {
    status.code = kPotrfFailed;
    status.raw_status = static_cast<int32_t>(solver_status);
    return absl::OkStatus();
  }
  status.called = 1;
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));
  JAXMG_RETURN_IF_ERROR(work.ReadInfo(cuda_stream, status.info));
  if (status.info != 0) status.code = kPotrfInfoNonzero;
  return absl::OkStatus();
}

absl::Status RunLuFactorization(
    const CusolverMpApi& api, cusolverMpHandle_t handle,
    cusolverMpMatrixDescriptor_t desc_a, cudaStream_t cuda_stream, int64_t n,
    void* a, cudaDataType_t dtype, int64_t* ipiv, size_t ipiv_bytes,
    FactorizationStatusFields status) {
  // Stage 1: query and allocate GETRF's temporary storage. The pivots outlive
  // this workspace because the caller uses them in GETRS or returns them.
  FactorizationWorkspace work;
  cusolverStatus_t solver_status = api.getrf_buffer_size(
      handle, n, n, a, /*ia=*/1, /*ja=*/1, desc_a, ipiv, dtype,
      &work.device_bytes, &work.host_bytes);
  if (solver_status != CUSOLVER_STATUS_SUCCESS) {
    status.code = kGetrfWorkspaceFailed;
    status.raw_status = static_cast<int32_t>(solver_status);
    return absl::OkStatus();
  }
  JAXMG_RETURN_IF_ERROR(work.Allocate(cuda_stream, status));
  if (status.code != kStatusOk) return absl::OkStatus();
  // Unwritten entries remain identifiable when factor-only pivots are gathered.
  JAXMG_RETURN_IF_CUDA_ERROR(cudaMemsetAsync(ipiv, 0xFF, ipiv_bytes, cuda_stream));
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));

  // Stage 2: factor A in place and preserve numerical failure diagnostics.
  solver_status = api.getrf(
      handle, n, n, a, /*ia=*/1, /*ja=*/1, desc_a, ipiv, dtype,
      work.device, work.device_bytes, work.host, work.host_bytes, work.info);
  if (solver_status != CUSOLVER_STATUS_SUCCESS) {
    status.code = kGetrfFailed;
    status.raw_status = static_cast<int32_t>(solver_status);
    return absl::OkStatus();
  }
  status.called = 1;
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));
  JAXMG_RETURN_IF_ERROR(work.ReadInfo(cuda_stream, status.info));
  if (status.info != 0) status.code = kGetrfInfoNonzero;
  return absl::OkStatus();
}

}  // namespace xla::gpu
