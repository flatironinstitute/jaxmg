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
// Production cuSOLVERMp LU solve FFI handler.
//
// This file owns the full fused LU solve native path. It starts from padded JAX
// local shards, converts their local storage to cuSOLVERMp's column-major view,
// performs edge-padding compaction and 2D block-cyclic redistribution, calls
// cuSOLVERMp GETRF/GETRS through the borrowed XLA-owned NCCL communicator, and
// redistributes the solved right-hand side back to the original JAX layout.
//
// File workflow:
//   1. Validate padded local JAX buffers and output aliases.
//   2. Allocate one scratch buffer through the native memory_redist formula.
//   3. Convert A and B locally from JAX row-major to cuSOLVERMp column-major.
//   4. Redistribute A and B into local 2D block-cyclic cuSOLVERMp storage.
//   5. Borrow the XLA communicator's raw NCCL handle and run GETRF/GETRS.
//   6. Reverse-redistribute solved B back to the JAX-facing distribution.
//   7. Restore solved B from column-major local storage to JAX row-major.

#include <array>
#include <cstdlib>

#include "cusolvermp_common.h"
#include "cusolvermp_routines.h"

namespace xla::gpu {
namespace {

// Executes the cuSOLVERMp LU factorization and linear solve.
//
// Inputs have already been converted into cuSOLVERMp local 2D block-cyclic
// storage by memory_redist. This helper owns the solver descriptors,
// workspaces, info buffers, and GETRF/GETRS calls. It intentionally does not
// gather a residual on the host; production correctness is reported through the
// cuSOLVERMp info/status vector, while numerical checks belong in validation
// jobs.
template <typename DataType>
absl::Status RunCusolverMpDistributedLuSolve(
    const CusolverMpApi& api, cusolverMpHandle_t handle,
    cusolverMpGrid_t grid, cudaStream_t cuda_stream, int64_t n,
    int64_t nrhs, int64_t tile_size, int64_t local_physical_rows,
    int64_t local_physical_cols_a, int64_t local_physical_cols_b,
    int32_t process_row, int32_t process_col, int32_t nccl_rank,
    ffi::AnyBuffer a, ffi::AnyBuffer b, ffi::Result<ffi::AnyBuffer> a_out,
    ffi::Result<ffi::AnyBuffer> b_out,
    std::array<int32_t, kLuSolveStatusSize>* status_words) {
  const int32_t process_rows = (*status_words)[4];
  const int32_t process_cols = (*status_words)[5];
  const int64_t local_rows = LocalNumroc(n, tile_size, process_row,
                                         static_cast<int32_t>(process_rows));
  const int64_t local_cols_a = LocalNumroc(n, tile_size, process_col,
                                           static_cast<int32_t>(process_cols));
  const int64_t local_cols_b = LocalNumroc(nrhs, tile_size, process_col,
                                           static_cast<int32_t>(process_cols));
  (*status_words)[18] = static_cast<int32_t>(local_rows);
  (*status_words)[19] = static_cast<int32_t>(local_cols_a);
  (*status_words)[20] = static_cast<int32_t>(local_rows);
  (*status_words)[21] = static_cast<int32_t>(local_cols_b);
  (*status_words)[36] = static_cast<int32_t>(nrhs);
  (*status_words)[37] = 1;  // A came from JAX/native redistribution.
  (*status_words)[38] = 1;  // B came from JAX/native redistribution.
  if (local_physical_rows < local_rows ||
      local_physical_cols_a < local_cols_a ||
      local_physical_cols_b < local_cols_b) {
    (*status_words)[0] = kOutputShapeMismatch;
    return absl::OkStatus();
  }

  cusolverMpMatrixDescriptor_t desc_a = nullptr;
  cusolverMpMatrixDescriptor_t desc_b = nullptr;
  cusolverStatus_t status = api.create_matrix_desc(
      &desc_a, grid, SolverTraits<DataType>::cuda_data_type, n, n, tile_size,
      tile_size, /*RSRC_A=*/0, /*CSRC_A=*/0, local_physical_rows);
  if (status != CUSOLVER_STATUS_SUCCESS || desc_a == nullptr) {
    (*status_words)[0] = kCreateMatrixDescFailed;
    (*status_words)[11] = static_cast<int32_t>(status);
    return absl::OkStatus();
  }
  (*status_words)[10] = 1;

  status = api.create_matrix_desc(
      &desc_b, grid, SolverTraits<DataType>::cuda_data_type, n, nrhs,
      tile_size, tile_size, /*RSRC_B=*/0, /*CSRC_B=*/0,
      local_physical_rows);
  if (status != CUSOLVER_STATUS_SUCCESS || desc_b == nullptr) {
    (*status_words)[0] = kCreateMatrixDescFailed;
    (*status_words)[11] = static_cast<int32_t>(status);
    api.destroy_matrix_desc(desc_a);
    return absl::OkStatus();
  }

  int64_t* d_ipiv = nullptr;
  void* d_getrf_work = nullptr;
  void* d_getrs_work = nullptr;
  void* h_getrf_work = nullptr;
  void* h_getrs_work = nullptr;
  int* d_getrf_info = nullptr;
  int* d_getrs_info = nullptr;
  auto cleanup = [&]() {
    if (d_ipiv != nullptr) cudaFree(d_ipiv);
    if (d_getrf_work != nullptr) cudaFree(d_getrf_work);
    if (d_getrs_work != nullptr) cudaFree(d_getrs_work);
    if (d_getrf_info != nullptr) cudaFree(d_getrf_info);
    if (d_getrs_info != nullptr) cudaFree(d_getrs_info);
    if (h_getrf_work != nullptr) std::free(h_getrf_work);
    if (h_getrs_work != nullptr) std::free(h_getrs_work);
    api.destroy_matrix_desc(desc_b);
    api.destroy_matrix_desc(desc_a);
  };

  // cuSOLVERMp GETRF writes one local pivot entry per locally owned matrix
  // column.  Older documentation has described this as a local-row quantity,
  // but NVIDIA's sample code and forum clarification use LOCc(N_A).  This is
  // especially important for degenerate P x 1 grids, where every rank owns all
  // columns and a row-sized pivot buffer can be much too small.
  const int64_t ipiv_len = local_cols_a;
  const size_t ipiv_bytes =
      static_cast<size_t>(ipiv_len) * sizeof(int64_t);
  (*status_words)[33] = static_cast<int32_t>(ipiv_len);
  (*status_words)[40] = SizeToKiBForStatus(ipiv_bytes);
  cudaError_t cuda_status =
      cudaMalloc(reinterpret_cast<void**>(&d_ipiv), ipiv_bytes);
  if (cuda_status != cudaSuccess) {
    (*status_words)[0] = kDeviceAllocFailed;
    cleanup();
    return absl::OkStatus();
  }

  size_t getrf_workspace_device = 0;
  size_t getrf_workspace_host = 0;
  size_t getrs_workspace_device = 0;
  size_t getrs_workspace_host = 0;
  status = api.getrf_buffer_size(
      handle, n, n, a_out->untyped_data(), /*ia=*/1, /*ja=*/1, desc_a,
      d_ipiv, SolverTraits<DataType>::cuda_data_type,
      &getrf_workspace_device, &getrf_workspace_host);
  if (status != CUSOLVER_STATUS_SUCCESS) {
    (*status_words)[0] = kGetrfWorkspaceFailed;
    (*status_words)[11] = static_cast<int32_t>(status);
    cleanup();
    return absl::OkStatus();
  }
  (*status_words)[22] = SizeToKiBForStatus(getrf_workspace_device);
  (*status_words)[23] = SizeToKiBForStatus(getrf_workspace_host);

  status = api.getrs_buffer_size(
      handle, CUBLAS_OP_N, n, nrhs, a_out->untyped_data(), /*ia=*/1,
      /*ja=*/1, desc_a, d_ipiv, b_out->untyped_data(), /*ib=*/1, /*jb=*/1,
      desc_b, SolverTraits<DataType>::cuda_data_type,
      &getrs_workspace_device, &getrs_workspace_host);
  if (status != CUSOLVER_STATUS_SUCCESS) {
    (*status_words)[0] = kGetrsWorkspaceFailed;
    (*status_words)[11] = static_cast<int32_t>(status);
    cleanup();
    return absl::OkStatus();
  }
  (*status_words)[24] = SizeToKiBForStatus(getrs_workspace_device);
  (*status_words)[25] = SizeToKiBForStatus(getrs_workspace_host);

  if (getrf_workspace_device > 0) {
    cuda_status = cudaMalloc(&d_getrf_work, getrf_workspace_device);
    if (cuda_status != cudaSuccess) {
      (*status_words)[0] = kDeviceAllocFailed;
      cleanup();
      return absl::OkStatus();
    }
  }
  if (getrs_workspace_device > 0) {
    cuda_status = cudaMalloc(&d_getrs_work, getrs_workspace_device);
    if (cuda_status != cudaSuccess) {
      (*status_words)[0] = kDeviceAllocFailed;
      cleanup();
      return absl::OkStatus();
    }
  }
  cuda_status = cudaMalloc(reinterpret_cast<void**>(&d_getrf_info),
                           sizeof(int));
  if (cuda_status != cudaSuccess) {
    (*status_words)[0] = kDeviceAllocFailed;
    cleanup();
    return absl::OkStatus();
  }
  cuda_status = cudaMalloc(reinterpret_cast<void**>(&d_getrs_info),
                           sizeof(int));
  if (cuda_status != cudaSuccess) {
    (*status_words)[0] = kDeviceAllocFailed;
    cleanup();
    return absl::OkStatus();
  }
  if (getrf_workspace_host > 0) {
    h_getrf_work = std::malloc(getrf_workspace_host);
    if (h_getrf_work == nullptr) {
      (*status_words)[0] = kHostAllocFailed;
      cleanup();
      return absl::OkStatus();
    }
  }
  if (getrs_workspace_host > 0) {
    h_getrs_work = std::malloc(getrs_workspace_host);
    if (h_getrs_work == nullptr) {
      (*status_words)[0] = kHostAllocFailed;
      cleanup();
      return absl::OkStatus();
    }
  }

  JAXMG_RETURN_IF_ERROR(CopyAnyBufferToOutputIfNeeded(cuda_stream, a, a_out));
  JAXMG_RETURN_IF_ERROR(CopyAnyBufferToOutputIfNeeded(cuda_stream, b, b_out));
  JAXMG_RETURN_IF_CUDA_ERROR(
      cudaMemsetAsync(d_ipiv, 0, ipiv_bytes, cuda_stream));
  JAXMG_RETURN_IF_CUDA_ERROR(
      cudaMemsetAsync(d_getrf_info, 0, sizeof(int), cuda_stream));
  JAXMG_RETURN_IF_CUDA_ERROR(
      cudaMemsetAsync(d_getrs_info, 0, sizeof(int), cuda_stream));
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));

  status = api.getrf(handle, n, n, a_out->untyped_data(), /*ia=*/1,
                     /*ja=*/1, desc_a, d_ipiv,
                     SolverTraits<DataType>::cuda_data_type, d_getrf_work,
                     getrf_workspace_device, h_getrf_work,
                     getrf_workspace_host, d_getrf_info);
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));
  if (status != CUSOLVER_STATUS_SUCCESS) {
    (*status_words)[0] = kGetrfFailed;
    (*status_words)[11] = static_cast<int32_t>(status);
    cleanup();
    return absl::OkStatus();
  }
  (*status_words)[26] = 1;

  int h_getrf_info = -1;
  JAXMG_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
      &h_getrf_info, d_getrf_info, sizeof(int), cudaMemcpyDeviceToHost,
      cuda_stream));
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));
  (*status_words)[27] = h_getrf_info;
  if (h_getrf_info != 0) {
    (*status_words)[0] = kGetrfInfoNonzero;
    cleanup();
    return absl::OkStatus();
  }

  status = api.getrs(handle, CUBLAS_OP_N, n, nrhs, a_out->untyped_data(),
                     /*ia=*/1, /*ja=*/1, desc_a, d_ipiv,
                     b_out->untyped_data(), /*ib=*/1, /*jb=*/1, desc_b,
                     SolverTraits<DataType>::cuda_data_type, d_getrs_work,
                     getrs_workspace_device, h_getrs_work,
                     getrs_workspace_host, d_getrs_info);
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));
  if (status != CUSOLVER_STATUS_SUCCESS) {
    (*status_words)[0] = kGetrsFailed;
    (*status_words)[11] = static_cast<int32_t>(status);
    cleanup();
    return absl::OkStatus();
  }
  (*status_words)[28] = 1;

  int h_getrs_info = -1;
  JAXMG_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
      &h_getrs_info, d_getrs_info, sizeof(int), cudaMemcpyDeviceToHost,
      cuda_stream));
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));
  (*status_words)[29] = h_getrs_info;
  if (h_getrs_info != 0) {
    (*status_words)[0] = kGetrsInfoNonzero;
    cleanup();
    return absl::OkStatus();
  }


  cleanup();
  return absl::OkStatus();
}


// Creates the borrowed-XLA-communicator cuSOLVERMp handle/grid and dispatches
// the dtype-specific GETRF/GETRS helper on redistributed A and B buffers.
absl::Status RunCusolverMpLuSolveSolver(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    int64_t process_rows, int64_t process_cols, int64_t n, int64_t nrhs,
    int64_t tile_size, int64_t grid_mapping,
    absl::Span<const int64_t> rank_map, ffi::AnyBuffer a, ffi::AnyBuffer b,
    ffi::Result<ffi::AnyBuffer> a_out, ffi::Result<ffi::AnyBuffer> b_out,
    ffi::Result<ffi::BufferR1<S32>> status_out,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  (void)comm_stream;
  // Stage 1: validate the static FFI contract.  By this point Python/JAX has
  // already padded the user arrays and memory_redist has produced local
  // column-major 2D block-cyclic buffers.  The checks here are deliberately
  // about local capacities and dtype consistency, not global mathematical
  // correctness.
  if (stream == nullptr || cuda_stream == nullptr) {
    return absl::InvalidArgumentError(
        "cusolvermp_lu_solve requires XLA and CUDA streams");
  }
  if (a.dimensions().size() != 2 || b.dimensions().size() != 2 ||
      a_out->dimensions().size() != 2 || b_out->dimensions().size() != 2) {
    return absl::InvalidArgumentError(
        "cusolvermp_lu_solve expects rank-2 A and B buffers");
  }
  if (a.element_type() != b.element_type() ||
      a.element_type() != a_out->element_type() ||
      b.element_type() != b_out->element_type()) {
    return absl::InvalidArgumentError(
        "cusolvermp_lu_solve requires matching A/B dtypes");
  }
  if (a.dimensions()[0] != a_out->dimensions()[0] ||
      a.dimensions()[1] != a_out->dimensions()[1] ||
      b.dimensions()[0] != b_out->dimensions()[0] ||
      b.dimensions()[1] != b_out->dimensions()[1]) {
    return absl::InvalidArgumentError(
        "cusolvermp_lu_solve input/output shapes must match");
  }
  if (a.dimensions()[0] != b.dimensions()[0]) {
    return absl::InvalidArgumentError(
        "cusolvermp_lu_solve expects A and B to have the same "
        "local row capacity");
  }
  if (status_out->dimensions().size() != 1 ||
      status_out->dimensions()[0] != kLuSolveStatusSize) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "cusolvermp_lu_solve expects status shape (%d,)",
        kLuSolveStatusSize));
  }

  std::array<int32_t, kLuSolveStatusSize> status_words = {
      kStatusOk,
      -1,  // CUDA device selected for this FFI invocation.
      -1,  // NCCL rank reported by the borrowed communicator.
      -1,  // NCCL communicator size.
      static_cast<int32_t>(process_rows),
      static_cast<int32_t>(process_cols),
      -1,  // cuSOLVERMp version, if available.
      0,   // libcusolverMp linked runtime available.
      0,   // cusolverMpHandle_t created.
      0,   // cusolverMpGrid_t created.
      0,   // cusolverMpMatrixDescriptor_t for A created.
      0,   // raw cuSOLVER status from the failing call, if any.
      static_cast<int32_t>(a.size_bytes()),
      static_cast<int32_t>(n),
      static_cast<int32_t>(tile_size),
      static_cast<int32_t>(a.dimensions()[0]),
      static_cast<int32_t>(a.dimensions()[1]),
      static_cast<int32_t>(b.dimensions()[0]),
      -1,  // local NUMROC rows for A.
      -1,  // local NUMROC cols for A.
      -1,  // local NUMROC rows for B.
      -1,  // local NUMROC cols for B.
      -1,  // GETRF device workspace, KiB.
      -1,  // GETRF host workspace, KiB.
      -1,  // GETRS device workspace, KiB.
      -1,  // GETRS host workspace, KiB.
      0,   // cusolverMpGetrf called.
      -1,  // GETRF info value copied from device.
      0,   // cusolverMpGetrs called.
      -1,  // GETRS info value copied from device.
      0,   // A native redistribution reached cuSOLVERMp layout.
      0,   // B native redistribution reached cuSOLVERMp layout.
      0,   // B reverse redistribution completed.
      -1,  // pivot vector length.
      -1,  // dtype code.
      static_cast<int32_t>(b.dimensions()[1]),
      static_cast<int32_t>(nrhs),
      0,  // A native redistribution flag.
      0,  // B native redistribution flag.
      -1,  // cuSOLVERMp grid mapping.
      -1,  // pivot vector allocation, KiB.
  };

  // Stage 2: bind CUDA work to the device that owns this rank's local A shard.
  // This avoids relying on ambient host-thread CUDA state, which is not stable
  // across JAX's multi-device FFI callbacks.
  absl::StatusOr<int> buffer_device = DeviceForCudaPointer(a.untyped_data());
  if (!buffer_device.ok()) {
    status_words[0] = kCudaDeviceFailed;
    return CopyLuSolveStatusToDevice(stream, status_words, status_out);
  }
  int cuda_device = *buffer_device;
  cudaError_t cuda_status = cudaSetDevice(cuda_device);
  if (cuda_status != cudaSuccess) {
    status_words[0] = kCudaDeviceFailed;
    return CopyLuSolveStatusToDevice(stream, status_words, status_out);
  }
  status_words[1] = cuda_device;

  // Stage 3: retrieve the communicator that XLA already created for the
  // compiled program.  cuSOLVERMp receives the raw NCCL handle from that
  // communicator; JAXMg does not create a separate NCCL communicator here.
  if (collective_params == nullptr || collective_cliques == nullptr) {
    status_words[0] = kCollectiveContextMissing;
    return CopyLuSolveStatusToDevice(stream, status_words, status_out);
  }

  absl::StatusOr<GpuCliqueKey> clique_key =
      AllAssignedDevicesP2PCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    status_words[0] = kCliqueKeyFailed;
    return CopyLuSolveStatusToDevice(stream, status_words, status_out);
  }

  absl::StatusOr<GpuCommunicator*> gpu_comm = collective_cliques->GetComm(
      *clique_key, collective_params->global_device_id);
  if (!gpu_comm.ok() || *gpu_comm == nullptr) {
    status_words[0] = kCommunicatorMissing;
    return CopyLuSolveStatusToDevice(stream, status_words, status_out);
  }

  void* platform_handle = (*gpu_comm)->platform_comm().handle;
  if (platform_handle == nullptr) {
    status_words[0] = kNcclHandleMissing;
    return CopyLuSolveStatusToDevice(stream, status_words, status_out);
  }
  ncclComm_t nccl_comm = reinterpret_cast<ncclComm_t>(platform_handle);

  int nccl_rank = -1;
  int nccl_count = -1;
  ncclResult_t rank_status = ncclCommUserRank(nccl_comm, &nccl_rank);
  ncclResult_t count_status = ncclCommCount(nccl_comm, &nccl_count);
  if (rank_status != ncclSuccess || count_status != ncclSuccess) {
    status_words[0] = kNcclRankMismatch;
    return CopyLuSolveStatusToDevice(stream, status_words, status_out);
  }
  status_words[2] = nccl_rank;
  status_words[3] = nccl_count;

  // Stage 4: validate that the requested cuSOLVERMp process grid matches the
  // communicator size and that any explicit Python rank map is one of the two
  // dense mappings cuSOLVERMp can describe directly.
  if (process_rows <= 0 || process_cols <= 0 ||
      process_rows * process_cols != nccl_count || n <= 0 || nrhs <= 0 ||
      tile_size <= 0) {
    status_words[0] = kGridShapeMismatch;
    return CopyLuSolveStatusToDevice(stream, status_words, status_out);
  }
  absl::Status grid_mapping_status =
      ValidateCusolverMpGridMapping("cusolvermp_lu_solve", grid_mapping);
  if (!grid_mapping_status.ok()) {
    return grid_mapping_status;
  }
  status_words[39] = static_cast<int32_t>(grid_mapping);
  if (!rank_map.empty()) {
    absl::Status rank_map_status = ValidateStandardRankMapForGridMapping(
        "cusolvermp_lu_solve", rank_map, process_rows, process_cols,
        grid_mapping);
    if (!rank_map_status.ok()) {
      return rank_map_status;
    }
  }

  // Stage 5: create the cuSOLVERMp handle/grid for this rank. The grid mapping
  // value is passed through to cuSOLVERMp so row-major and
  // column-major JAX mesh orders remain supported without an extra remap layer.
  CusolverMpApi api = LinkedCusolverMpApi(&status_words);

  cusolverMpHandle_t handle = nullptr;
  cusolverStatus_t cusolver_status =
      api.create(&handle, cuda_device, cuda_stream);
  if (cusolver_status != CUSOLVER_STATUS_SUCCESS || handle == nullptr) {
    status_words[0] = kCreateHandleFailed;
    status_words[11] = static_cast<int32_t>(cusolver_status);
    return CopyLuSolveStatusToDevice(stream, status_words, status_out);
  }
  status_words[8] = 1;

  int version = -1;
  cusolver_status = api.get_version(handle, &version);
  if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
    status_words[0] = kGetVersionFailed;
    status_words[11] = static_cast<int32_t>(cusolver_status);
    api.destroy(handle);
    return CopyLuSolveStatusToDevice(stream, status_words, status_out);
  }
  status_words[6] = version;

  cusolverMpGrid_t grid = nullptr;
  cusolver_status = api.create_grid(
      handle, &grid, nccl_comm, static_cast<int32_t>(process_rows),
      static_cast<int32_t>(process_cols),
      ToCusolverMpGridMapping(grid_mapping));
  if (cusolver_status != CUSOLVER_STATUS_SUCCESS || grid == nullptr) {
    status_words[0] = kCreateGridFailed;
    status_words[11] = static_cast<int32_t>(cusolver_status);
    api.destroy(handle);
    return CopyLuSolveStatusToDevice(stream, status_words, status_out);
  }
  status_words[9] = 1;

  const auto [process_row, process_col] = ProcessCoordFromRank(
      nccl_rank, process_rows, process_cols, grid_mapping);

  // Stage 6: dispatch on the XLA primitive dtype and run GETRF/GETRS.
  absl::Status lu_solve_status;
  switch (a.element_type()) {
    case F32:
      status_words[34] = 1;
      lu_solve_status = RunCusolverMpDistributedLuSolve<float>(
          api, handle, grid, cuda_stream, n, nrhs, tile_size,
          a.dimensions()[0], a.dimensions()[1], b.dimensions()[1],
          process_row, process_col, nccl_rank, a, b, a_out, b_out, &status_words);
      break;
    case F64:
      status_words[34] = 2;
      lu_solve_status = RunCusolverMpDistributedLuSolve<double>(
          api, handle, grid, cuda_stream, n, nrhs, tile_size,
          a.dimensions()[0], a.dimensions()[1], b.dimensions()[1],
          process_row, process_col, nccl_rank, a, b, a_out, b_out, &status_words);
      break;
    case C64:
      status_words[34] = 3;
      lu_solve_status = RunCusolverMpDistributedLuSolve<cuFloatComplex>(
          api, handle, grid, cuda_stream, n, nrhs, tile_size,
          a.dimensions()[0], a.dimensions()[1], b.dimensions()[1],
          process_row, process_col, nccl_rank, a, b, a_out, b_out, &status_words);
      break;
    case C128:
      status_words[34] = 4;
      lu_solve_status = RunCusolverMpDistributedLuSolve<cuDoubleComplex>(
          api, handle, grid, cuda_stream, n, nrhs, tile_size,
          a.dimensions()[0], a.dimensions()[1], b.dimensions()[1],
          process_row, process_col, nccl_rank, a, b, a_out, b_out, &status_words);
      break;
    default:
      status_words[0] = kUnsupportedDtype;
      break;
  }
  if (!lu_solve_status.ok()) {
    api.destroy_grid(grid);
    api.destroy(handle);
    return lu_solve_status;
  }

  // Stage 7: tear down cuSOLVERMp resources in reverse construction order and
  // return the per-rank status vector to Python.
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
  return CopyLuSolveStatusToDevice(stream, status_words, status_out);
}


}  // namespace

// Prepare is intentionally limited to communicator setup.  XLA constructs the
// all-assigned P2P clique during compilation; dispatch later retrieves the
// communicator and uses it for both redistribution and cuSOLVERMp.
absl::Status XlaCusolverMpLuSolvePrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests) {
  return RequestAllAssignedP2PCommunicator(
      collective_params, clique_requests, "cusolvermp_lu_solve");
}

// Fused LU solve FFI entry point.
//
// This is the production native workflow called from Python: allocate one XLA
// scratch window, convert local JAX layout to cuSOLVERMp layout, run forward
// edge-padding/block-cyclic redistribution for A and B, call cuSOLVERMp, move
// the solved B back to the JAX distribution, and restore B's local row-major
// layout for the user-visible output.
absl::Status XlaCusolverMpLuSolveDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    se::OwningScratchAllocator<> scratch, int64_t process_rows,
    int64_t process_cols, int64_t n, int64_t nrhs,
    int64_t b_distribution_cols, int64_t tile_size, int64_t grid_mapping,
    absl::Span<const int64_t> rank_map, ffi::AnyBuffer a, ffi::AnyBuffer b,
    ffi::Result<ffi::AnyBuffer> a_work, ffi::Result<ffi::AnyBuffer> b_out,
    ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  // Stage 1: validate the local FFI buffers.  Python has already padded A/B to
  // the user-visible storage capacity needed by the requested process grid and
  // tile size; native code only checks the local contracts it will dereference.
  if (a.dimensions().size() != 2 || b.dimensions().size() != 2 ||
      a_work->dimensions().size() != 2 || b_out->dimensions().size() != 2) {
    return absl::InvalidArgumentError(
        "cusolvermp_lu_solve expects rank-2 A/B buffers");
  }
  if (a.element_type() != b.element_type() ||
      a.element_type() != a_work->element_type() ||
      b.element_type() != b_out->element_type()) {
    return absl::InvalidArgumentError(
        "cusolvermp_lu_solve requires matching A/B dtypes");
  }
  if (a.dimensions()[0] != a_work->dimensions()[0] ||
      a.dimensions()[1] != a_work->dimensions()[1] ||
      b.dimensions()[0] != b_out->dimensions()[0] ||
      b.dimensions()[1] != b_out->dimensions()[1]) {
    return absl::InvalidArgumentError(
        "cusolvermp_lu_solve input/output shapes must match");
  }
  if (a.dimensions()[0] != b.dimensions()[0]) {
    return absl::InvalidArgumentError(
        "cusolvermp_lu_solve expects A and B to have matching local row "
        "capacity after padding");
  }
  if (b_distribution_cols < nrhs) {
    return absl::InvalidArgumentError(
        "cusolvermp_lu_solve requires b_distribution_cols >= nrhs");
  }
  // Stage 2: size one reusable scratch allocation for the whole fused call.
  // The memory_redist layer owns the formula so the solver wrapper does not
  // duplicate layout-conversion, edge-padding, or 2D cyclic scratch rules.
  // A and B can have different logical column counts. For B there are two
  // counts: `nrhs` is the real RHS width passed to cuSOLVERMp, while
  // `b_distribution_cols` is the JAX-visible routing width used to make skinny
  // RHS matrices shardable over process-grid columns. Describe both native
  // redistribution requests and let memory_redist allocate one reusable scratch
  // buffer large enough for every layout-conversion, padding, and block-cyclic
  // movement in this fused FFI call.
  const size_t element_bytes =
      a.size_bytes() / static_cast<size_t>(a.element_count());
  const std::array<Padded2DRedistScratchRequest, 2> scratch_requests = {{
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
      Padded2DRedistScratchRequest{
          /*process_rows=*/process_rows,
          /*process_cols=*/process_cols,
          /*tile_rows=*/tile_size,
          /*tile_cols=*/tile_size,
          /*logical_rows=*/n,
          /*logical_cols=*/b_distribution_cols,
          /*local_rows=*/b.dimensions()[0],
          /*local_cols=*/b.dimensions()[1],
          /*rank_map=*/rank_map,
      },
  }};
  absl::StatusOr<Padded2DRedistScratch> redistribution_scratch_status =
      AllocatePadded2DRedistScratch(
          cuda_stream, element_bytes, absl::MakeConstSpan(scratch_requests),
          "cusolvermp_lu_solve_redistribution");
  if (!redistribution_scratch_status.ok()) {
    return redistribution_scratch_status.status();
  }
  Padded2DRedistScratch redistribution_scratch =
      *redistribution_scratch_status;
  se::DeviceAddressBase scratch_base = redistribution_scratch.base;
  const int64_t scratch_elements = redistribution_scratch.elements;

  bool scratch_freed = false;
  auto free_redistribution_scratch = [&]() -> absl::Status {
    if (scratch_freed) {
      return absl::OkStatus();
    }
    absl::Status free_status = FreePadded2DRedistScratch(
        cuda_stream, redistribution_scratch,
        "cusolvermp_lu_solve_redistribution");
    if (!free_status.ok()) {
      return free_status;
    }
    scratch_freed = true;
    return absl::OkStatus();
  };
  auto return_after_cleanup = [&](absl::Status status) -> absl::Status {
    absl::Status free_status = free_redistribution_scratch();
    if (!status.ok()) {
      return status;
    }
    return free_status;
  };
  auto cuda_status_after_cleanup =
      [&](cudaError_t status, const char* caller) -> absl::Status {
    if (status == cudaSuccess) {
      return absl::OkStatus();
    }
    return return_after_cleanup(absl::InternalError(absl::StrFormat(
        "%s failed: cuda status %d (%s)", caller, static_cast<int>(status),
        cudaGetErrorString(status))));
  };

  ffi::AnyBuffer a_forward_input = a;
  ffi::AnyBuffer b_forward_input = b;
  // Stage 3: make the local buffer storage convention match cuSOLVERMp.  JAX
  // enters with row-major local shards; cuSOLVERMp descriptors interpret local
  // memory as column-major.  This in-place conversion avoids XLA's full-size
  // layout-copy allocation. When XLA aliases input/output buffers the copy is a
  // no-op; otherwise the explicit copy preserves correctness without a second
  // native matrix allocation.
  if (absl::Status status = CopyMatrixIfNeeded(cuda_stream, a, a_work);
      !status.ok()) {
    return return_after_cleanup(status);
  }
  if (absl::Status status = CopyMatrixIfNeeded(cuda_stream, b, b_out);
      !status.ok()) {
    return return_after_cleanup(status);
  }
  a_forward_input = *a_work;
  b_forward_input = *b_out;
  if (absl::Status status = ConvertRowMajorToColumnMajorInPlace(
          cuda_stream, "cusolvermp_lu_solve/a_layout_convert",
          a_forward_input, scratch_base, scratch_elements);
      !status.ok()) {
    return return_after_cleanup(status);
  }
  if (absl::Status status = ConvertRowMajorToColumnMajorInPlace(
          cuda_stream, "cusolvermp_lu_solve/b_layout_convert",
          b_forward_input, scratch_base, scratch_elements);
      !status.ok()) {
    return return_after_cleanup(status);
  }

  // Stage 4: move local shards into cuSOLVERMp's 2D block-cyclic distribution.
  // A is routed over the full n x n logical matrix; B is routed over the padded
  // distribution width but only the first nrhs columns are solved.
  if (absl::Status status = ExecutePadded2DNativePlanRaw(
          "cusolvermp_lu_solve/a_forward", stream, comm_stream, cuda_stream,
          process_rows, process_cols, tile_size, tile_size, n, n,
          /*reverse=*/0, rank_map, a_forward_input, a_work->device_memory(),
          scratch_base, scratch_elements, collective_params,
          collective_cliques);
      !status.ok()) {
    return return_after_cleanup(status);
  }
  if (absl::Status status = ExecutePadded2DNativePlanRaw(
          "cusolvermp_lu_solve/b_forward", stream, comm_stream, cuda_stream,
          process_rows, process_cols, tile_size, tile_size, n,
          b_distribution_cols, /*reverse=*/0, rank_map, b_forward_input,
          b_out->device_memory(), scratch_base, scratch_elements,
          collective_params, collective_cliques);
      !status.ok()) {
    return return_after_cleanup(status);
  }

  // The previous production prototype ran redistribution and cuSOLVERMp as
  // separate FFI calls, which gave XLA a hard sequencing boundary.  The fused
  // handler must provide the same safety explicitly before the host-side
  // cuSOLVERMp call reads the redistributed buffers.  This can be relaxed to an
  // event dependency after the correctness matrix is stable.
  if (absl::Status status = cuda_status_after_cleanup(
          cudaStreamSynchronize(cuda_stream),
          "cusolvermp_lu_solve forward stream synchronize");
      !status.ok()) {
    return status;
  }

  ffi::AnyBuffer a_cyclic = *a_work;
  ffi::AnyBuffer b_cyclic = *b_out;
  // Stage 5: run GETRF/GETRS on the redistributed buffers.  The shared helper
  // handles the cuSOLVERMp ABI boundary and borrowed NCCL communicator.
  if (absl::Status solver_call_status = RunCusolverMpLuSolveSolver(
          stream, comm_stream, cuda_stream, process_rows, process_cols, n,
          nrhs, tile_size, grid_mapping, rank_map, a_cyclic, b_cyclic,
          a_work, b_out, status, collective_params, collective_cliques);
      !solver_call_status.ok()) {
    return return_after_cleanup(solver_call_status);
  }

  // cuSOLVERMp is issued on the same stream, but keep the fused reverse
  // redistribution boundary explicit for the same reason as the forward solve
  // boundary above.
  if (absl::Status status = cuda_status_after_cleanup(
          cudaStreamSynchronize(cuda_stream),
          "cusolvermp_lu_solve solver stream synchronize");
      !status.ok()) {
    return status;
  }

  ffi::AnyBuffer b_solved_cyclic = *b_out;
  // Stage 6: return only B to the JAX-facing distribution.  A is a factorized
  // LU work buffer after GETRF and is returned only for donation/aliasing.
  if (absl::Status status = ExecutePadded2DNativePlanRaw(
          "cusolvermp_lu_solve/b_reverse", stream, comm_stream, cuda_stream,
          process_rows, process_cols, tile_size, tile_size, n,
          b_distribution_cols, /*reverse=*/1, rank_map, b_solved_cyclic,
          b_out->device_memory(), scratch_base, scratch_elements,
          collective_params, collective_cliques);
      !status.ok()) {
    return return_after_cleanup(status);
  }

  // Stage 7: restore the local storage convention for the user-visible output.
  if (absl::Status status = ConvertColumnMajorToRowMajorInPlace(
          cuda_stream, "cusolvermp_lu_solve/b_layout_restore", *b_out,
          scratch_base, scratch_elements);
      !status.ok()) {
    return return_after_cleanup(status);
  }
  return return_after_cleanup(absl::OkStatus());
}

}  // namespace xla::gpu
