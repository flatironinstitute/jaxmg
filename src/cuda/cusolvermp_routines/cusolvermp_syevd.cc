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
// Production cuSOLVERMp symmetric/Hermitian eigensolver FFI handler.
//
// This file owns the full fused vector-producing SYEVD native path. JAXMg does
// not expose a no-eigenvector mode: the production contract is to compute and
// return eigenvalues plus eigenvectors, matching the workflows this backend is
// designed to support.
//
// File workflow:
//   1. Validate padded local JAX buffers and output aliases.
//   2. Allocate one scratch buffer through the native memory_redist formula.
//   3. Convert A locally from JAX row-major to cuSOLVERMp column-major.
//   4. Redistribute A into local 2D block-cyclic cuSOLVERMp storage.
//   5. Borrow the XLA communicator's raw NCCL handle and run SYEVD.
//   6. Reverse-redistribute eigenvectors to the JAX-facing distribution.
//   7. Restore eigenvectors from column-major local storage to JAX row-major.

#include <array>
#include <cstdlib>

#include "cusolvermp_common.h"
#include "../include/xla_comm_backend.h"

namespace xla::gpu {
namespace {

// Run vector-producing SYEVD once memory_redist has converted JAX buffers
// into cuSOLVERMp local 2D block-cyclic storage.  The input matrix is
// copied/aliased into work_out because cuSOLVERMp overwrites d_A, while
// vectors_out is the separate d_Z eigenvector output.
template <typename DataType>
absl::Status RunCusolverMpSyevd(
    const CusolverMpApi& api, cusolverMpHandle_t handle,
    cusolverMpGrid_t grid, cudaStream_t cuda_stream, int64_t n,
    int64_t tile_size, int64_t local_physical_rows,
    int64_t local_physical_cols, int32_t process_row, int32_t process_col,
    ffi::AnyBuffer a, ffi::Result<ffi::AnyBuffer> eigenvalues_out,
    ffi::Result<ffi::AnyBuffer> work_out,
    ffi::Result<ffi::AnyBuffer> vectors_out,
    std::array<int32_t, kSyevdStatusSize>* status_words) {
  const int debug_rank = (*status_words)[2];
  const int32_t process_rows = (*status_words)[4];
  const int32_t process_cols = (*status_words)[5];
  const int64_t local_rows = LocalNumroc(n, tile_size, process_row,
                                         static_cast<int32_t>(process_rows));
  const int64_t local_cols = LocalNumroc(n, tile_size, process_col,
                                         static_cast<int32_t>(process_cols));
  (*status_words)[17] = static_cast<int32_t>(local_rows);
  (*status_words)[18] = static_cast<int32_t>(local_cols);
  if (local_physical_rows < local_rows || local_physical_cols < local_cols) {
    (*status_words)[0] = kOutputShapeMismatch;
    return absl::OkStatus();
  }
  CusolverMpDebug(
      debug_rank,
      "syevd enter n=%lld tile=%lld process_coord=(%d,%d) local=%lldx%lld "
      "physical=%lldx%lld dtype=%d",
      static_cast<long long>(n), static_cast<long long>(tile_size),
      process_row, process_col, static_cast<long long>(local_rows),
      static_cast<long long>(local_cols),
      static_cast<long long>(local_physical_rows),
      static_cast<long long>(local_physical_cols),
      static_cast<int>(SolverTraits<DataType>::cuda_data_type));

  cusolverMpMatrixDescriptor_t desc_a = nullptr;
  cusolverMpMatrixDescriptor_t desc_q = nullptr;
  CusolverMpDebug(debug_rank, "create desc_a begin");
  cusolverStatus_t status = api.create_matrix_desc(
      &desc_a, grid, SolverTraits<DataType>::cuda_data_type, n, n, tile_size,
      tile_size, /*RSRC_A=*/0, /*CSRC_A=*/0, local_physical_rows);
  CusolverMpDebug(debug_rank, "create desc_a end status=%d desc=%p",
                  static_cast<int>(status), desc_a);
  if (status != CUSOLVER_STATUS_SUCCESS || desc_a == nullptr) {
    (*status_words)[0] = kCreateMatrixDescFailed;
    (*status_words)[11] = static_cast<int32_t>(status);
    return absl::OkStatus();
  }
  (*status_words)[10] = 1;

  CusolverMpDebug(debug_rank, "create desc_q begin");
  status = api.create_matrix_desc(
      &desc_q, grid, SolverTraits<DataType>::cuda_data_type, n, n, tile_size,
      tile_size, /*RSRC_Q=*/0, /*CSRC_Q=*/0, local_physical_rows);
  CusolverMpDebug(debug_rank, "create desc_q end status=%d desc=%p",
                  static_cast<int>(status), desc_q);
  if (status != CUSOLVER_STATUS_SUCCESS || desc_q == nullptr) {
    (*status_words)[0] = kCreateMatrixDescFailed;
    (*status_words)[11] = static_cast<int32_t>(status);
    api.destroy_matrix_desc(desc_a);
    return absl::OkStatus();
  }
  (*status_words)[27] = 1;

  void* d_work = nullptr;
  void* h_work = nullptr;
  int* d_info = nullptr;
  auto cleanup = [&]() {
    if (d_work != nullptr) cudaFree(d_work);
    if (d_info != nullptr) cudaFree(d_info);
    if (h_work != nullptr) std::free(h_work);
    api.destroy_matrix_desc(desc_q);
    api.destroy_matrix_desc(desc_a);
  };

  // cuSOLVERMp SYEVD overwrites d_A and writes eigenvectors to d_Z when
  // requested. Keep those buffers distinct: the donated JAX input aliases the
  // work output used as d_A, while the public eigenvector result is d_Z.
  CusolverMpDebug(debug_rank, "copy input to work begin a=%p work=%p bytes=%lld",
                  a.untyped_data(), work_out->untyped_data(),
                  static_cast<long long>(a.size_bytes()));
  JAXMG_RETURN_IF_ERROR(CopyAnyBufferToOutputIfNeeded(cuda_stream, a, work_out));
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));
  CusolverMpDebug(debug_rank, "copy input to work end");

  size_t workspace_device = 0;
  size_t workspace_host = 0;
  // cuSOLVERMp 0.7.x exposes this selector as `compz` and NVIDIA's
  // distributed SYEVD sample uses 'Z' for the eigenvector-producing path. Keep
  // this aligned with the installed header/runtime rather than the newer
  // `jobz='V'` wording in some versions of the public documentation.
  char compz[] = {'Z', '\0'};
  void* z_data = vectors_out->untyped_data();
  CusolverMpDebug(
      debug_rank,
      "syevd_buffer_size begin compz=%c a=%p evals=%p z=%p desc_a=%p desc_q=%p",
      compz[0], work_out->untyped_data(), eigenvalues_out->untyped_data(),
      z_data, desc_a, desc_q);
  status = api.syevd_buffer_size(
      handle, compz, CUBLAS_FILL_MODE_LOWER, n, work_out->untyped_data(),
      /*IA=*/1, /*JA=*/1, desc_a, eigenvalues_out->untyped_data(),
      z_data, /*IQ=*/1, /*JQ=*/1, desc_q,
      SolverTraits<DataType>::cuda_data_type, &workspace_device,
      &workspace_host);
  CusolverMpDebug(debug_rank,
                  "syevd_buffer_size end status=%d workspace_device=%zu "
                  "workspace_host=%zu",
                  static_cast<int>(status), workspace_device, workspace_host);
  if (status != CUSOLVER_STATUS_SUCCESS) {
    (*status_words)[0] = kSyevdWorkspaceFailed;
    (*status_words)[11] = static_cast<int32_t>(status);
    cleanup();
    return absl::OkStatus();
  }
  (*status_words)[21] = SizeToKiBForStatus(workspace_device);
  (*status_words)[22] = SizeToKiBForStatus(workspace_host);

  cudaError_t cuda_status = cudaSuccess;
  if (workspace_device > 0) {
    CusolverMpDebug(debug_rank, "cudaMalloc d_work begin bytes=%zu",
                    workspace_device);
    cuda_status = cudaMalloc(&d_work, workspace_device);
    CusolverMpDebug(debug_rank, "cudaMalloc d_work end status=%d ptr=%p",
                    static_cast<int>(cuda_status), d_work);
    if (cuda_status != cudaSuccess) {
      (*status_words)[0] = kDeviceAllocFailed;
      cleanup();
      return absl::OkStatus();
    }
  }
  if (workspace_host > 0) {
    CusolverMpDebug(debug_rank, "malloc h_work begin bytes=%zu", workspace_host);
    h_work = std::malloc(workspace_host);
    CusolverMpDebug(debug_rank, "malloc h_work end ptr=%p", h_work);
    if (h_work == nullptr) {
      (*status_words)[0] = kHostAllocFailed;
      cleanup();
      return absl::OkStatus();
    }
  }
  CusolverMpDebug(debug_rank, "cudaMalloc d_info begin");
  cuda_status = cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int));
  CusolverMpDebug(debug_rank, "cudaMalloc d_info end status=%d ptr=%p",
                  static_cast<int>(cuda_status), d_info);
  if (cuda_status != cudaSuccess) {
    (*status_words)[0] = kDeviceAllocFailed;
    cleanup();
    return absl::OkStatus();
  }
  JAXMG_RETURN_IF_CUDA_ERROR(
      cudaMemsetAsync(d_info, 0, sizeof(int), cuda_stream));
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));

  CusolverMpDebug(debug_rank, "syevd begin");
  status = api.syevd(handle, compz, CUBLAS_FILL_MODE_LOWER, n,
                     work_out->untyped_data(), /*IA=*/1, /*JA=*/1, desc_a,
                     eigenvalues_out->untyped_data(), z_data, /*IQ=*/1,
                     /*JQ=*/1, desc_q,
                     SolverTraits<DataType>::cuda_data_type, d_work,
                     workspace_device, h_work, workspace_host, d_info);
  CusolverMpDebug(debug_rank, "syevd end status=%d", static_cast<int>(status));
  if (status != CUSOLVER_STATUS_SUCCESS) {
    (*status_words)[0] = kSyevdFailed;
    (*status_words)[11] = static_cast<int32_t>(status);
    cleanup();
    return absl::OkStatus();
  }
  (*status_words)[23] = 1;
  CusolverMpDebug(debug_rank, "stream sync after syevd begin");
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));
  CusolverMpDebug(debug_rank, "stream sync after syevd end");

  int h_info = -1;
  JAXMG_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
      &h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost, cuda_stream));
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));
  CusolverMpDebug(debug_rank, "syevd info=%d", h_info);
  (*status_words)[24] = h_info;
  if (h_info != 0) {
    (*status_words)[0] = kSyevdInfoNonzero;
  }

  cleanup();
  return absl::OkStatus();
}


absl::Status RunCusolverMpSyevdSolver(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    int64_t process_rows, int64_t process_cols, int64_t n,
    int64_t tile_size, int64_t grid_mapping,
    absl::Span<const int64_t> rank_map, ffi::AnyBuffer a,
    ffi::Result<ffi::AnyBuffer> eigenvalues_out,
    ffi::Result<ffi::AnyBuffer> work_out,
    ffi::Result<ffi::AnyBuffer> vectors_out,
    ffi::Result<ffi::BufferR1<S32>> status_out,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  (void)comm_stream;
  // Stage 1: validate the local FFI buffers.  SYEVD has three matrix buffers:
  // the redistributed input A, work_out used as cuSOLVERMp's overwritten d_A,
  // and vectors_out used as cuSOLVERMp's d_Z eigenvector output.
  if (stream == nullptr || cuda_stream == nullptr) {
    return absl::InvalidArgumentError(
        "cusolvermp_syevd requires XLA and CUDA streams");
  }
  if (a.dimensions().size() != 2 || work_out->dimensions().size() != 2 ||
      vectors_out->dimensions().size() != 2 ||
      eigenvalues_out->dimensions().size() != 1) {
    return absl::InvalidArgumentError(
        "cusolvermp_syevd expects rank-2 matrix buffers and rank-1 "
        "eigenvalue output");
  }
  if (a.element_type() != work_out->element_type() ||
      a.element_type() != vectors_out->element_type()) {
    return absl::InvalidArgumentError(
        "cusolvermp_syevd requires matching matrix/work/vector dtypes");
  }
  const PrimitiveType expected_eigen_type =
      (a.element_type() == F32 || a.element_type() == C64) ? F32 : F64;
  if (eigenvalues_out->element_type() != expected_eigen_type) {
    return absl::InvalidArgumentError(
        "cusolvermp_syevd received an eigenvalue output with the wrong dtype");
  }
  if (a.dimensions()[0] != work_out->dimensions()[0] ||
      a.dimensions()[1] != work_out->dimensions()[1] ||
      a.dimensions()[0] != vectors_out->dimensions()[0] ||
      a.dimensions()[1] != vectors_out->dimensions()[1]) {
    return absl::InvalidArgumentError(
        "cusolvermp_syevd input/output matrix shapes must match");
  }
  if (eigenvalues_out->dimensions()[0] != n) {
    return absl::InvalidArgumentError(
        "cusolvermp_syevd eigenvalue output length must match n");
  }
  if (status_out->dimensions().size() != 1 ||
      status_out->dimensions()[0] != kSyevdStatusSize) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "cusolvermp_syevd expects status shape (%d,)", kSyevdStatusSize));
  }

  std::array<int32_t, kSyevdStatusSize> status_words = {
      kStatusOk,
      -1,  // CUDA device selected for this FFI invocation.
      -1,  // NCCL rank reported by the borrowed communicator.
      -1,  // NCCL communicator size.
      static_cast<int32_t>(process_rows),
      static_cast<int32_t>(process_cols),
      -1,  // cuSOLVERMp version, if available.
      0,   // libcusolverMp linked runtime available.
      0,   // cuSOLVERMp handle created.
      0,   // cuSOLVERMp grid created.
      0,   // matrix descriptor for A created.
      0,   // raw cuSOLVER status from the failing call, if any.
      static_cast<int32_t>(a.size_bytes()),
      static_cast<int32_t>(n),
      static_cast<int32_t>(tile_size),
      static_cast<int32_t>(a.dimensions()[0]),
      static_cast<int32_t>(a.dimensions()[1]),
      -1,  // local NUMROC rows.
      -1,  // local NUMROC cols.
      static_cast<int32_t>(eigenvalues_out->size_bytes()),
      1,   // vector-producing SYEVD path.
      -1,  // syevd device workspace, KiB.
      -1,  // syevd host workspace, KiB.
      0,   // cusolverMpSyevd called.
      -1,  // syevd info value copied from device.
      -1,  // dtype code.
      static_cast<int32_t>(grid_mapping),
      0,   // matrix descriptor for Q created.
      0,   // matrix came from native JAX-buffer redistribution.
      0,   // reserved.
      0,   // reserved.
      0,   // reserved.
      0,   // reserved.
      0,   // reserved.
      0,   // reserved.
      0,   // reserved.
  };

  // Stage 2: bind cuSOLVERMp to the CUDA device that owns this rank's local
  // matrix shard.  This mirrors POTRS and avoids stale current-device state in
  // multi-device JAX callbacks.
  absl::StatusOr<int> buffer_device = DeviceForCudaPointer(a.untyped_data());
  if (!buffer_device.ok()) {
    status_words[0] = kCudaDeviceFailed;
    return CopySyevdStatusToDevice(stream, status_words, status_out);
  }
  int cuda_device = *buffer_device;
  cudaError_t cuda_status = cudaSetDevice(cuda_device);
  if (cuda_status != cudaSuccess) {
    status_words[0] = kCudaDeviceFailed;
    return CopySyevdStatusToDevice(stream, status_words, status_out);
  }
  status_words[1] = cuda_device;
  CusolverMpDebug(-1, "syevd dispatch device=%d n=%lld tile=%lld grid=%lldx%lld "
                      "vectors=1",
                  cuda_device, static_cast<long long>(n),
                  static_cast<long long>(tile_size),
                  static_cast<long long>(process_rows),
                  static_cast<long long>(process_cols));

  // Stage 3: borrow the XLA-created NCCL communicator for this rank.  The
  // communicator rank is then converted to a cuSOLVERMp process-grid
  // coordinate using the requested row-major or column-major mapping.
  if (collective_params == nullptr || collective_cliques == nullptr) {
    status_words[0] = kCollectiveContextMissing;
    return CopySyevdStatusToDevice(stream, status_words, status_out);
  }

  absl::StatusOr<GpuCliqueKey> clique_key =
      AllAssignedDevicesP2PCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    status_words[0] = kCliqueKeyFailed;
    return CopySyevdStatusToDevice(stream, status_words, status_out);
  }

  absl::StatusOr<GpuCommunicator*> gpu_comm = collective_cliques->GetComm(
      *clique_key, collective_params->global_device_id);
  if (!gpu_comm.ok() || *gpu_comm == nullptr) {
    status_words[0] = kCommunicatorMissing;
    return CopySyevdStatusToDevice(stream, status_words, status_out);
  }

  void* platform_handle = (*gpu_comm)->platform_comm().handle;
  if (platform_handle == nullptr) {
    status_words[0] = kNcclHandleMissing;
    return CopySyevdStatusToDevice(stream, status_words, status_out);
  }
  ncclComm_t nccl_comm = reinterpret_cast<ncclComm_t>(platform_handle);

  int nccl_rank = -1;
  int nccl_count = -1;
  ncclResult_t rank_status = ncclCommUserRank(nccl_comm, &nccl_rank);
  ncclResult_t count_status = ncclCommCount(nccl_comm, &nccl_count);
  if (rank_status != ncclSuccess || count_status != ncclSuccess) {
    status_words[0] = kNcclRankMismatch;
    return CopySyevdStatusToDevice(stream, status_words, status_out);
  }
  status_words[2] = nccl_rank;
  status_words[3] = nccl_count;
  CusolverMpDebug(nccl_rank, "nccl communicator rank=%d count=%d",
                  nccl_rank, nccl_count);

  // Stage 4: validate the grid metadata.  cuSOLVERMp is a direct native
  // dependency now, so missing symbols fail during build/import rather than
  // inside this runtime path.
  if (process_rows <= 0 || process_cols <= 0 ||
      process_rows * process_cols != nccl_count || n <= 0 ||
      tile_size <= 0) {
    status_words[0] = kGridShapeMismatch;
    return CopySyevdStatusToDevice(stream, status_words, status_out);
  }
  JAXMG_RETURN_IF_ERROR(
      ValidateCusolverMpGridMapping("cusolvermp_syevd", grid_mapping));
  if (!rank_map.empty()) {
    JAXMG_RETURN_IF_ERROR(ValidateStandardRankMapForGridMapping(
        "cusolvermp_syevd", rank_map, process_rows, process_cols,
        grid_mapping));
  }

  CusolverMpApi api = LinkedCusolverMpApi(&status_words);

  // Stage 5: create the cuSOLVERMp handle and grid against the borrowed NCCL
  // communicator.  The heavy local-buffer redistribution has already happened
  // in memory_redist; this stage only defines cuSOLVERMp's view of the grid.
  cusolverMpHandle_t handle = nullptr;
  CusolverMpDebug(nccl_rank, "cusolverMpCreate begin device=%d stream=%p",
                  cuda_device, reinterpret_cast<void*>(cuda_stream));
  cusolverStatus_t cusolver_status =
      api.create(&handle, cuda_device, cuda_stream);
  CusolverMpDebug(nccl_rank, "cusolverMpCreate end status=%d handle=%p",
                  static_cast<int>(cusolver_status), handle);
  if (cusolver_status != CUSOLVER_STATUS_SUCCESS || handle == nullptr) {
    status_words[0] = kCreateHandleFailed;
    status_words[11] = static_cast<int32_t>(cusolver_status);
    return CopySyevdStatusToDevice(stream, status_words, status_out);
  }
  status_words[8] = 1;

  int version = -1;
  cusolver_status = api.get_version(handle, &version);
  if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
    status_words[0] = kGetVersionFailed;
    status_words[11] = static_cast<int32_t>(cusolver_status);
    api.destroy(handle);
    return CopySyevdStatusToDevice(stream, status_words, status_out);
  }
  status_words[6] = version;
  CusolverMpDebug(nccl_rank, "cusolverMp version=%d", version);

  cusolverMpGrid_t grid = nullptr;
  CusolverMpDebug(nccl_rank,
                  "cusolverMpCreateDeviceGrid begin rows=%lld cols=%lld "
                  "mapping=%lld",
                  static_cast<long long>(process_rows),
                  static_cast<long long>(process_cols),
                  static_cast<long long>(grid_mapping));
  cusolver_status = api.create_grid(
      handle, &grid, nccl_comm, static_cast<int32_t>(process_rows),
      static_cast<int32_t>(process_cols),
      ToCusolverMpGridMapping(grid_mapping));
  CusolverMpDebug(nccl_rank, "cusolverMpCreateDeviceGrid end status=%d grid=%p",
                  static_cast<int>(cusolver_status), grid);
  if (cusolver_status != CUSOLVER_STATUS_SUCCESS || grid == nullptr) {
    status_words[0] = kCreateGridFailed;
    status_words[11] = static_cast<int32_t>(cusolver_status);
    api.destroy(handle);
    return CopySyevdStatusToDevice(stream, status_words, status_out);
  }
  status_words[9] = 1;

  const auto [process_row, process_col] = ProcessCoordFromRank(
      nccl_rank, process_rows, process_cols, grid_mapping);
  CusolverMpDebug(nccl_rank, "dispatch process_coord=(%d,%d) element_type=%d",
                  process_row, process_col, static_cast<int>(a.element_type()));

  // Stage 6: dispatch on dtype and run the shared SYEVD helper.  The helper
  // keeps d_A and d_Z separate because vector-producing SYEVD overwrites A.
  absl::Status syevd_status;
  switch (a.element_type()) {
    case F32:
      status_words[25] = 1;
      syevd_status = RunCusolverMpSyevd<float>(
          api, handle, grid, cuda_stream, n, tile_size, a.dimensions()[0],
          a.dimensions()[1], process_row, process_col, a, eigenvalues_out,
          work_out, vectors_out, &status_words);
      break;
    case F64:
      status_words[25] = 2;
      syevd_status = RunCusolverMpSyevd<double>(
          api, handle, grid, cuda_stream, n, tile_size, a.dimensions()[0],
          a.dimensions()[1], process_row, process_col, a, eigenvalues_out,
          work_out, vectors_out, &status_words);
      break;
    case C64:
      status_words[25] = 3;
      syevd_status = RunCusolverMpSyevd<cuFloatComplex>(
          api, handle, grid, cuda_stream, n, tile_size, a.dimensions()[0],
          a.dimensions()[1], process_row, process_col, a, eigenvalues_out,
          work_out, vectors_out, &status_words);
      break;
    case C128:
      status_words[25] = 4;
      syevd_status = RunCusolverMpSyevd<cuDoubleComplex>(
          api, handle, grid, cuda_stream, n, tile_size, a.dimensions()[0],
          a.dimensions()[1], process_row, process_col, a, eigenvalues_out,
          work_out, vectors_out, &status_words);
      break;
    default:
      status_words[0] = kUnsupportedDtype;
      break;
  }
  if (!syevd_status.ok()) {
    api.destroy_grid(grid);
    api.destroy(handle);
    return syevd_status;
  }

  // Stage 7: release cuSOLVERMp objects and return status to Python.
  if (status_words[0] == kStatusOk) {
    cusolver_status = api.destroy_grid(grid);
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
      status_words[0] = kDestroyGridFailed;
      status_words[11] = static_cast<int32_t>(cusolver_status);
    }
  } else {
    api.destroy_grid(grid);
  }

  if (status_words[0] == kStatusOk) {
    cusolver_status = api.destroy(handle);
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
      status_words[0] = kDestroyHandleFailed;
      status_words[11] = static_cast<int32_t>(cusolver_status);
    }
  } else {
    api.destroy(handle);
  }
  return CopySyevdStatusToDevice(stream, status_words, status_out);
}


}  // namespace

// Prepare only requests the communicator clique.  The dispatch path later
// reuses that XLA-owned communicator for redistribution and passes its raw NCCL
// handle into cuSOLVERMp.
absl::Status XlaCusolverMpSyevdPrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests) {
  return RequestAllAssignedP2PCommunicator(
      collective_params, clique_requests, "cusolvermp_syevd");
}

absl::Status XlaCusolverMpSyevdDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    se::OwningScratchAllocator<> scratch, int64_t process_rows,
    int64_t process_cols, int64_t n, int64_t tile_size,
    int64_t grid_mapping, absl::Span<const int64_t> rank_map, ffi::AnyBuffer a,
    ffi::Result<ffi::AnyBuffer> eigenvalues,
    ffi::Result<ffi::AnyBuffer> work, ffi::Result<ffi::AnyBuffer> vectors,
    ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  // Stage 1: validate the local FFI buffers.  The public Python wrapper has
  // already padded A to tile-aligned capacity and allocated work/vectors with
  // the same local matrix capacity.
  if (a.dimensions().size() != 2 || work->dimensions().size() != 2 ||
      vectors->dimensions().size() != 2) {
    return absl::InvalidArgumentError(
        "cusolvermp_syevd expects rank-2 matrix buffers");
  }
  if (a.element_type() != work->element_type() ||
      a.element_type() != vectors->element_type()) {
    return absl::InvalidArgumentError(
        "cusolvermp_syevd requires matching matrix dtypes");
  }
  if (a.dimensions()[0] != work->dimensions()[0] ||
      a.dimensions()[1] != work->dimensions()[1] ||
      a.dimensions()[0] != vectors->dimensions()[0] ||
      a.dimensions()[1] != vectors->dimensions()[1]) {
    return absl::InvalidArgumentError(
        "cusolvermp_syevd input/output matrix shapes must match");
  }

  // Stage 2: allocate the one scratch window used for local layout conversion,
  // edge-padding compaction, and 2D block-cyclic redistribution in both
  // directions.
  const size_t element_bytes =
      a.size_bytes() / static_cast<size_t>(a.element_count());
  const std::array<Padded2DRedistScratchRequest, 1> scratch_requests = {{
      Padded2DRedistScratchRequest{
          /*process_rows=*/process_rows,
          /*process_cols=*/process_cols,
          /*tile_rows=*/tile_size,
          /*tile_cols=*/tile_size,
          /*logical_rows=*/n,
          /*logical_cols=*/n,
          /*local_rows=*/a.dimensions()[0],
          /*local_cols=*/a.dimensions()[1],
          /*rank_map=*/rank_map,
      },
  }};
  absl::StatusOr<Padded2DRedistScratch> redistribution_scratch =
      AllocatePadded2DRedistScratch(
          scratch, element_bytes, absl::MakeConstSpan(scratch_requests),
          "cusolvermp_syevd_redistribution");
  if (!redistribution_scratch.ok()) {
    return redistribution_scratch.status();
  }
  se::DeviceAddressBase scratch_base = redistribution_scratch->base;
  const int64_t scratch_elements = redistribution_scratch->elements;

  ffi::AnyBuffer a_forward_input = a;
  // Stage 3: copy/alias the input into work and convert the local storage from
  // JAX row-major to cuSOLVERMp column-major in place.  SYEVD overwrites d_A, so
  // the public input and the solver work buffer must stay conceptually
  // separate even when XLA can alias storage. If XLA cannot alias, `work`
  // remains the single full-size storage slot used by the fused call.
  JAXMG_RETURN_IF_ERROR(CopyMatrixIfNeeded(cuda_stream, a, work));
  a_forward_input = *work;
  JAXMG_RETURN_IF_ERROR(ConvertRowMajorToColumnMajorInPlace(
      cuda_stream, "cusolvermp_syevd/a_layout_convert", a_forward_input,
      scratch_base, scratch_elements));

  // Stage 4: redistribute A into cuSOLVERMp's 2D block-cyclic layout.
  JAXMG_RETURN_IF_ERROR(ExecutePadded2DNativePlanRaw(
      "cusolvermp_syevd/a_forward", stream, comm_stream, cuda_stream,
      process_rows, process_cols, tile_size, tile_size, n, n, /*reverse=*/0,
      rank_map, a_forward_input, work->device_memory(), scratch_base,
      scratch_elements, collective_params, collective_cliques));

  // Preserve the old split-call sequencing semantics inside the fused handler:
  // the host-side cuSOLVERMp call must not observe the work buffer until all
  // pack/NCCL/unpack redistribution work is complete.
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));

  ffi::AnyBuffer a_cyclic = *work;
  // Stage 5: run vector-producing SYEVD.  The helper creates cuSOLVERMp
  // descriptors for A and Q and writes eigenvectors into `vectors`.
  JAXMG_RETURN_IF_ERROR(RunCusolverMpSyevdSolver(
      stream, comm_stream, cuda_stream, process_rows, process_cols, n,
      tile_size, grid_mapping, rank_map, a_cyclic, eigenvalues, work, vectors,
      status, collective_params, collective_cliques));

  // SYEVD writes the eigenvectors in cuSOLVERMp layout.  Keep the reverse
  // redistribution after a completed solver call, mirroring the old separate
  // FFI-call pipeline.
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));

  ffi::AnyBuffer vectors_cyclic = *vectors;
  // Stage 6: reverse-redistribute eigenvectors to the original JAX-facing shard
  // placement.  Eigenvalues are already a replicated rank-1 output from
  // cuSOLVERMp and do not need matrix redistribution.
  JAXMG_RETURN_IF_ERROR(ExecutePadded2DNativePlanRaw(
      "cusolvermp_syevd/vectors_reverse", stream, comm_stream,
      cuda_stream, process_rows, process_cols, tile_size, tile_size, n, n,
      /*reverse=*/1, rank_map, vectors_cyclic, vectors->device_memory(),
      scratch_base, scratch_elements, collective_params, collective_cliques));

  // Stage 7: restore JAX's row-major local storage for the user-visible
  // eigenvector shard.
  JAXMG_RETURN_IF_ERROR(ConvertColumnMajorToRowMajorInPlace(
      cuda_stream, "cusolvermp_syevd/vectors_layout_restore", *vectors,
      scratch_base, scratch_elements));
  return absl::OkStatus();
}

}  // namespace xla::gpu
