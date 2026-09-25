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
// cuSOLVERMp least-squares FFI handler.
//
// This file implements the fused overdetermined GELS path. It converts padded
// JAX shards to column-major storage, redistributes rectangular A and B into
// cuSOLVERMp's 2D block-cyclic layout, solves min ||A X - B||_2, and restores
// the overwritten B work buffer to its original JAX layout.
//
// File workflow:
//   1. Validate the padded A/B and output-buffer contracts.
//   2. Allocate one scratch buffer shared by every redistribution stage.
//   3. Convert local A/B storage from row-major to column-major.
//   4. Redistribute both buffers into 2D block-cyclic storage.
//   5. Borrow XLA's NCCL communicator and run cusolverMpGels.
//   6. Reverse the B redistribution and restore row-major storage.

#include <array>
#include <cstdlib>

#include "cusolvermp_common.h"
#include "cusolvermp_routines.h"

namespace xla::gpu {
namespace {

// Runs GELS after memory_redist has placed A and B in local 2D block-cyclic
// storage. The solution overwrites the first N rows of B, matching the
// cuSOLVERMp in-place contract.
template <typename DataType>
absl::Status RunCusolverMpDistributedGels(
    const CusolverMpApi& api, cusolverMpHandle_t handle,
    cusolverMpGrid_t grid, cudaStream_t cuda_stream, int64_t m, int64_t n,
    int64_t nrhs, int64_t tile_size, int64_t local_physical_rows_a,
    int64_t local_physical_cols_a, int64_t local_physical_rows_b,
    int64_t local_physical_cols_b, int32_t process_row,
    int32_t process_col, ffi::AnyBuffer a, ffi::AnyBuffer b,
    ffi::Result<ffi::AnyBuffer> a_out, ffi::Result<ffi::AnyBuffer> b_out,
    std::array<int32_t, kGelsStatusSize>* status_words) {
  const int32_t process_rows = (*status_words)[4];
  const int32_t process_cols = (*status_words)[5];
  const int64_t local_rows_a =
      LocalNumroc(m, tile_size, process_row, process_rows);
  const int64_t local_cols_a =
      LocalNumroc(n, tile_size, process_col, process_cols);
  const int64_t local_rows_b =
      LocalNumroc(m, tile_size, process_row, process_rows);
  const int64_t local_cols_b =
      LocalNumroc(nrhs, tile_size, process_col, process_cols);
  (*status_words)[20] = static_cast<int32_t>(local_rows_a);
  (*status_words)[21] = static_cast<int32_t>(local_cols_a);
  (*status_words)[22] = static_cast<int32_t>(local_rows_b);
  (*status_words)[23] = static_cast<int32_t>(local_cols_b);
  (*status_words)[28] = 1;
  (*status_words)[29] = 1;

  if (local_physical_rows_a < local_rows_a ||
      local_physical_cols_a < local_cols_a ||
      local_physical_rows_b < local_rows_b ||
      local_physical_cols_b < local_cols_b) {
    (*status_words)[0] = kOutputShapeMismatch;
    return absl::OkStatus();
  }

  cusolverMpMatrixDescriptor_t desc_a = nullptr;
  cusolverMpMatrixDescriptor_t desc_b = nullptr;
  cusolverStatus_t solver_status = api.create_matrix_desc(
      &desc_a, grid, SolverTraits<DataType>::cuda_data_type, m, n, tile_size,
      tile_size, /*RSRC_A=*/0, /*CSRC_A=*/0, local_physical_rows_a);
  if (solver_status != CUSOLVER_STATUS_SUCCESS || desc_a == nullptr) {
    (*status_words)[0] = kCreateMatrixDescFailed;
    (*status_words)[11] = static_cast<int32_t>(solver_status);
    return absl::OkStatus();
  }
  (*status_words)[10] = 1;

  solver_status = api.create_matrix_desc(
      &desc_b, grid, SolverTraits<DataType>::cuda_data_type, m, nrhs,
      tile_size, tile_size, /*RSRC_B=*/0, /*CSRC_B=*/0,
      local_physical_rows_b);
  if (solver_status != CUSOLVER_STATUS_SUCCESS || desc_b == nullptr) {
    (*status_words)[0] = kCreateMatrixDescFailed;
    (*status_words)[11] = static_cast<int32_t>(solver_status);
    api.destroy_matrix_desc(desc_a);
    return absl::OkStatus();
  }

  void* d_work = nullptr;
  void* h_work = nullptr;
  int* d_info = nullptr;
  auto cleanup = [&]() {
    if (d_work != nullptr) cudaFree(d_work);
    if (h_work != nullptr) std::free(h_work);
    if (d_info != nullptr) cudaFree(d_info);
    api.destroy_matrix_desc(desc_b);
    api.destroy_matrix_desc(desc_a);
  };
  auto check_cuda = [&](cudaError_t result, const char* caller) -> absl::Status {
    if (result == cudaSuccess) return absl::OkStatus();
    cleanup();
    return absl::InternalError(absl::StrFormat(
        "%s failed: cuda status %d (%s)", caller,
        static_cast<int>(result), cudaGetErrorString(result)));
  };

  size_t workspace_device = 0;
  size_t workspace_host = 0;
  solver_status = api.gels_buffer_size(
      handle, CUBLAS_OP_N, m, n, nrhs, a_out->untyped_data(), /*ia=*/1,
      /*ja=*/1, desc_a, b_out->untyped_data(), /*ib=*/1, /*jb=*/1, desc_b,
      SolverTraits<DataType>::cuda_data_type, &workspace_device,
      &workspace_host);
  if (solver_status != CUSOLVER_STATUS_SUCCESS) {
    (*status_words)[0] = kGelsWorkspaceFailed;
    (*status_words)[11] = static_cast<int32_t>(solver_status);
    cleanup();
    return absl::OkStatus();
  }
  (*status_words)[24] = SizeToKiBForStatus(workspace_device);
  (*status_words)[25] = SizeToKiBForStatus(workspace_host);

  cudaError_t cuda_status = cudaSuccess;
  if (workspace_device > 0) {
    cuda_status = cudaMalloc(&d_work, workspace_device);
    if (cuda_status != cudaSuccess) {
      (*status_words)[0] = kDeviceAllocFailed;
      cleanup();
      return absl::OkStatus();
    }
  }
  if (workspace_host > 0) {
    h_work = std::malloc(workspace_host);
    if (h_work == nullptr) {
      (*status_words)[0] = kHostAllocFailed;
      cleanup();
      return absl::OkStatus();
    }
  }
  cuda_status = cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int));
  if (cuda_status != cudaSuccess) {
    (*status_words)[0] = kDeviceAllocFailed;
    cleanup();
    return absl::OkStatus();
  }

  if (absl::Status result =
          CopyAnyBufferToOutputIfNeeded(cuda_stream, a, a_out);
      !result.ok()) {
    cleanup();
    return result;
  }
  if (absl::Status result =
          CopyAnyBufferToOutputIfNeeded(cuda_stream, b, b_out);
      !result.ok()) {
    cleanup();
    return result;
  }
  if (absl::Status result = check_cuda(
          cudaMemsetAsync(d_info, 0, sizeof(int), cuda_stream),
          "cusolvermp_gels info initialization");
      !result.ok()) {
    return result;
  }
  if (absl::Status result = check_cuda(
          cudaStreamSynchronize(cuda_stream),
          "cusolvermp_gels pre-solve synchronization");
      !result.ok()) {
    return result;
  }

  solver_status = api.gels(
      handle, CUBLAS_OP_N, m, n, nrhs, a_out->untyped_data(), /*ia=*/1,
      /*ja=*/1, desc_a, b_out->untyped_data(), /*ib=*/1, /*jb=*/1, desc_b,
      SolverTraits<DataType>::cuda_data_type, d_work, workspace_device,
      h_work, workspace_host, d_info);
  if (absl::Status result = check_cuda(
          cudaStreamSynchronize(cuda_stream),
          "cusolvermp_gels solve synchronization");
      !result.ok()) {
    return result;
  }
  if (solver_status != CUSOLVER_STATUS_SUCCESS) {
    (*status_words)[0] = kGelsFailed;
    (*status_words)[11] = static_cast<int32_t>(solver_status);
    cleanup();
    return absl::OkStatus();
  }
  (*status_words)[26] = 1;

  int h_info = -1;
  if (absl::Status result = check_cuda(
          cudaMemcpyAsync(&h_info, d_info, sizeof(int),
                          cudaMemcpyDeviceToHost, cuda_stream),
          "cusolvermp_gels info copy");
      !result.ok()) {
    return result;
  }
  if (absl::Status result = check_cuda(
          cudaStreamSynchronize(cuda_stream),
          "cusolvermp_gels info synchronization");
      !result.ok()) {
    return result;
  }
  (*status_words)[27] = h_info;
  if (h_info != 0) {
    (*status_words)[0] = kGelsInfoNonzero;
  }

  cleanup();
  return absl::OkStatus();
}


// Creates the borrowed-communicator cuSOLVERMp context and dispatches the
// dtype-specific GELS implementation on redistributed A and B buffers.
absl::Status RunCusolverMpGelsSolver(
    se::Stream* stream, cudaStream_t cuda_stream, int64_t process_rows,
    int64_t process_cols, int64_t m, int64_t n, int64_t nrhs,
    int64_t tile_size, int64_t grid_mapping,
    absl::Span<const int64_t> rank_map, ffi::AnyBuffer a, ffi::AnyBuffer b,
    ffi::Result<ffi::AnyBuffer> a_out, ffi::Result<ffi::AnyBuffer> b_out,
    ffi::Result<ffi::BufferR1<S32>> status_out,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  if (stream == nullptr || cuda_stream == nullptr) {
    return absl::InvalidArgumentError(
        "cusolvermp_gels requires XLA and CUDA streams");
  }
  if (a.dimensions().size() != 2 || b.dimensions().size() != 2 ||
      a_out->dimensions().size() != 2 || b_out->dimensions().size() != 2) {
    return absl::InvalidArgumentError(
        "cusolvermp_gels expects rank-2 A and B buffers");
  }
  if (a.element_type() != b.element_type() ||
      a.element_type() != a_out->element_type() ||
      b.element_type() != b_out->element_type()) {
    return absl::InvalidArgumentError(
        "cusolvermp_gels requires matching A/B dtypes");
  }
  if (a.dimensions()[0] != a_out->dimensions()[0] ||
      a.dimensions()[1] != a_out->dimensions()[1] ||
      b.dimensions()[0] != b_out->dimensions()[0] ||
      b.dimensions()[1] != b_out->dimensions()[1]) {
    return absl::InvalidArgumentError(
        "cusolvermp_gels input/output shapes must match");
  }
  if (a.dimensions()[0] != b.dimensions()[0]) {
    return absl::InvalidArgumentError(
        "cusolvermp_gels expects matching local row capacity for A and B");
  }
  if (status_out->dimensions().size() != 1 ||
      status_out->dimensions()[0] != kGelsStatusSize) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "cusolvermp_gels expects status shape (%d,)", kGelsStatusSize));
  }

  std::array<int32_t, kGelsStatusSize> status_words = {
      kStatusOk,
      -1,  // CUDA device.
      -1,  // NCCL rank.
      -1,  // NCCL rank count.
      static_cast<int32_t>(process_rows),
      static_cast<int32_t>(process_cols),
      -1,  // cuSOLVERMp version.
      0,   // linked runtime available.
      0,   // handle created.
      0,   // grid created.
      0,   // A descriptor created.
      0,   // raw cuSOLVER status.
      static_cast<int32_t>(a.size_bytes()),
      static_cast<int32_t>(m),
      static_cast<int32_t>(n),
      static_cast<int32_t>(tile_size),
      static_cast<int32_t>(a.dimensions()[0]),
      static_cast<int32_t>(a.dimensions()[1]),
      static_cast<int32_t>(b.dimensions()[0]),
      static_cast<int32_t>(b.dimensions()[1]),
      -1,  // local NUMROC rows for A.
      -1,  // local NUMROC cols for A.
      -1,  // local NUMROC rows for B.
      -1,  // local NUMROC cols for B.
      -1,  // GELS device workspace, KiB.
      -1,  // GELS host workspace, KiB.
      0,   // cusolverMpGels called.
      -1,  // GELS info.
      0,   // A reached native distribution.
      0,   // B reached native distribution.
      0,   // B reverse redistribution completed.
      -1,  // dtype code.
      static_cast<int32_t>(nrhs),
      -1,  // grid mapping.
      static_cast<int32_t>(b.size_bytes()),
  };

  // Stage 1: select the CUDA device owning this rank's local matrix shard.
  absl::StatusOr<int> buffer_device = DeviceForCudaPointer(a.untyped_data());
  if (!buffer_device.ok() || cudaSetDevice(*buffer_device) != cudaSuccess) {
    status_words[0] = kCudaDeviceFailed;
    return CopyGelsStatusToDevice(stream, status_words, status_out);
  }
  const int cuda_device = *buffer_device;
  status_words[1] = cuda_device;

  // Stage 2: borrow the all-assigned NCCL communicator created by XLA.
  if (collective_params == nullptr || collective_cliques == nullptr) {
    status_words[0] = kCollectiveContextMissing;
    return CopyGelsStatusToDevice(stream, status_words, status_out);
  }
  absl::StatusOr<GpuCliqueKey> clique_key =
      AllAssignedDevicesP2PCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    status_words[0] = kCliqueKeyFailed;
    return CopyGelsStatusToDevice(stream, status_words, status_out);
  }
  absl::StatusOr<GpuCommunicator*> gpu_comm = collective_cliques->GetComm(
      *clique_key, collective_params->global_device_id);
  if (!gpu_comm.ok() || *gpu_comm == nullptr) {
    status_words[0] = kCommunicatorMissing;
    return CopyGelsStatusToDevice(stream, status_words, status_out);
  }
  void* platform_handle = (*gpu_comm)->platform_comm().handle;
  if (platform_handle == nullptr) {
    status_words[0] = kNcclHandleMissing;
    return CopyGelsStatusToDevice(stream, status_words, status_out);
  }
  ncclComm_t nccl_comm = reinterpret_cast<ncclComm_t>(platform_handle);

  int nccl_rank = -1;
  int nccl_count = -1;
  if (ncclCommUserRank(nccl_comm, &nccl_rank) != ncclSuccess ||
      ncclCommCount(nccl_comm, &nccl_count) != ncclSuccess) {
    status_words[0] = kNcclRankMismatch;
    return CopyGelsStatusToDevice(stream, status_words, status_out);
  }
  status_words[2] = nccl_rank;
  status_words[3] = nccl_count;

  // Stage 3: validate the overdetermined no-transpose GELS contract and grid.
  if (process_rows <= 0 || process_cols <= 0 ||
      process_rows * process_cols != nccl_count || m <= 0 || n <= 0 ||
      m < n || nrhs <= 0 || tile_size <= 0) {
    status_words[0] = kGridShapeMismatch;
    return CopyGelsStatusToDevice(stream, status_words, status_out);
  }
  JAXMG_RETURN_IF_ERROR(
      ValidateCusolverMpGridMapping("cusolvermp_gels", grid_mapping));
  status_words[33] = static_cast<int32_t>(grid_mapping);
  if (!rank_map.empty()) {
    JAXMG_RETURN_IF_ERROR(ValidateStandardRankMapForGridMapping(
        "cusolvermp_gels", rank_map, process_rows, process_cols,
        grid_mapping));
  }

  // Stage 4: create the cuSOLVERMp handle and process grid.
  CusolverMpApi api = LinkedCusolverMpApi(&status_words);
  cusolverMpHandle_t handle = nullptr;
  cusolverStatus_t solver_status =
      api.create(&handle, cuda_device, cuda_stream);
  if (solver_status != CUSOLVER_STATUS_SUCCESS || handle == nullptr) {
    status_words[0] = kCreateHandleFailed;
    status_words[11] = static_cast<int32_t>(solver_status);
    return CopyGelsStatusToDevice(stream, status_words, status_out);
  }
  status_words[8] = 1;

  int version = -1;
  solver_status = api.get_version(handle, &version);
  if (solver_status != CUSOLVER_STATUS_SUCCESS) {
    status_words[0] = kGetVersionFailed;
    status_words[11] = static_cast<int32_t>(solver_status);
    api.destroy(handle);
    return CopyGelsStatusToDevice(stream, status_words, status_out);
  }
  status_words[6] = version;

  cusolverMpGrid_t grid = nullptr;
  solver_status = api.create_grid(
      handle, &grid, nccl_comm, static_cast<int32_t>(process_rows),
      static_cast<int32_t>(process_cols),
      ToCusolverMpGridMapping(grid_mapping));
  if (solver_status != CUSOLVER_STATUS_SUCCESS || grid == nullptr) {
    status_words[0] = kCreateGridFailed;
    status_words[11] = static_cast<int32_t>(solver_status);
    api.destroy(handle);
    return CopyGelsStatusToDevice(stream, status_words, status_out);
  }
  status_words[9] = 1;

  const auto [process_row, process_col] = ProcessCoordFromRank(
      nccl_rank, process_rows, process_cols, grid_mapping);

  // Stage 5: dispatch GELS using the matrix element type.
  absl::Status gels_status;
  switch (a.element_type()) {
    case F32:
      status_words[31] = 1;
      gels_status = RunCusolverMpDistributedGels<float>(
          api, handle, grid, cuda_stream, m, n, nrhs, tile_size,
          a.dimensions()[0], a.dimensions()[1], b.dimensions()[0],
          b.dimensions()[1], process_row, process_col, a, b, a_out, b_out,
          &status_words);
      break;
    case F64:
      status_words[31] = 2;
      gels_status = RunCusolverMpDistributedGels<double>(
          api, handle, grid, cuda_stream, m, n, nrhs, tile_size,
          a.dimensions()[0], a.dimensions()[1], b.dimensions()[0],
          b.dimensions()[1], process_row, process_col, a, b, a_out, b_out,
          &status_words);
      break;
    case C64:
      status_words[31] = 3;
      gels_status = RunCusolverMpDistributedGels<cuFloatComplex>(
          api, handle, grid, cuda_stream, m, n, nrhs, tile_size,
          a.dimensions()[0], a.dimensions()[1], b.dimensions()[0],
          b.dimensions()[1], process_row, process_col, a, b, a_out, b_out,
          &status_words);
      break;
    case C128:
      status_words[31] = 4;
      gels_status = RunCusolverMpDistributedGels<cuDoubleComplex>(
          api, handle, grid, cuda_stream, m, n, nrhs, tile_size,
          a.dimensions()[0], a.dimensions()[1], b.dimensions()[0],
          b.dimensions()[1], process_row, process_col, a, b, a_out, b_out,
          &status_words);
      break;
    default:
      status_words[0] = kUnsupportedDtype;
      gels_status = absl::OkStatus();
      break;
  }

  if (!gels_status.ok()) {
    api.destroy_grid(grid);
    api.destroy(handle);
    return gels_status;
  }

  // Stage 6: destroy native resources and publish per-rank diagnostics.
  if (api.destroy_grid(grid) != CUSOLVER_STATUS_SUCCESS &&
      status_words[0] == kStatusOk) {
    status_words[0] = kDestroyGridFailed;
  }
  if (api.destroy(handle) != CUSOLVER_STATUS_SUCCESS &&
      status_words[0] == kStatusOk) {
    status_words[0] = kDestroyHandleFailed;
  }
  return CopyGelsStatusToDevice(stream, status_words, status_out);
}

}  // namespace


absl::Status XlaCusolverMpGelsPrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests) {
  return RequestAllAssignedP2PCommunicator(
      collective_params, clique_requests, "cusolvermp_gels");
}


absl::Status XlaCusolverMpGelsDispatch(
    se::Stream* stream, cudaStream_t cuda_stream, int64_t process_rows,
    int64_t process_cols, int64_t m, int64_t n, int64_t nrhs,
    int64_t b_distribution_cols, int64_t tile_size, int64_t grid_mapping,
    absl::Span<const int64_t> rank_map, ffi::AnyBuffer a, ffi::AnyBuffer b,
    ffi::Result<ffi::AnyBuffer> a_work, ffi::Result<ffi::AnyBuffer> b_out,
    ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  se::Stream* comm_stream = nullptr;

  // Stage 1: validate the local padded buffers supplied by Python/JAX.
  if (a.dimensions().size() != 2 || b.dimensions().size() != 2 ||
      a_work->dimensions().size() != 2 || b_out->dimensions().size() != 2) {
    return absl::InvalidArgumentError(
        "cusolvermp_gels expects rank-2 A/B buffers");
  }
  if (a.element_type() != b.element_type() ||
      a.element_type() != a_work->element_type() ||
      b.element_type() != b_out->element_type()) {
    return absl::InvalidArgumentError(
        "cusolvermp_gels requires matching A/B dtypes");
  }
  if (a.dimensions()[0] != a_work->dimensions()[0] ||
      a.dimensions()[1] != a_work->dimensions()[1] ||
      b.dimensions()[0] != b_out->dimensions()[0] ||
      b.dimensions()[1] != b_out->dimensions()[1]) {
    return absl::InvalidArgumentError(
        "cusolvermp_gels input/output shapes must match");
  }
  if (a.dimensions()[0] != b.dimensions()[0] ||
      b_distribution_cols < nrhs || m < n) {
    return absl::InvalidArgumentError(
        "cusolvermp_gels received incompatible A/B dimensions");
  }

  // Stage 2: allocate one scratch region large enough for either matrix.
  const size_t element_bytes =
      a.size_bytes() / static_cast<size_t>(a.element_count());
  const std::array<Padded2DRedistScratchRequest, 2> scratch_requests = {{
      Padded2DRedistScratchRequest{
          process_rows, process_cols, tile_size, tile_size, m, n,
          a.dimensions()[0], a.dimensions()[1], rank_map,
      },
      Padded2DRedistScratchRequest{
          process_rows, process_cols, tile_size, tile_size, m,
          b_distribution_cols, b.dimensions()[0], b.dimensions()[1], rank_map,
      },
  }};
  absl::StatusOr<Padded2DRedistScratch> scratch_status =
      AllocatePadded2DRedistScratch(
          cuda_stream, element_bytes, absl::MakeConstSpan(scratch_requests),
          "cusolvermp_gels_redistribution");
  if (!scratch_status.ok()) return scratch_status.status();
  Padded2DRedistScratch scratch = *scratch_status;
  const se::DeviceAddressBase scratch_base = scratch.base;
  const int64_t scratch_elements = scratch.elements;

  bool scratch_freed = false;
  auto free_scratch = [&]() -> absl::Status {
    if (scratch_freed) return absl::OkStatus();
    absl::Status result = FreePadded2DRedistScratch(
        cuda_stream, scratch, "cusolvermp_gels_redistribution");
    if (result.ok()) scratch_freed = true;
    return result;
  };
  auto return_after_cleanup = [&](absl::Status result) -> absl::Status {
    absl::Status free_status = free_scratch();
    return result.ok() ? free_status : result;
  };
  auto check_cuda = [&](cudaError_t result, const char* caller) -> absl::Status {
    if (result == cudaSuccess) return absl::OkStatus();
    return return_after_cleanup(absl::InternalError(absl::StrFormat(
        "%s failed: cuda status %d (%s)", caller, static_cast<int>(result),
        cudaGetErrorString(result))));
  };

  // Stage 3: reuse donated outputs as row-to-column-major work buffers.
  if (absl::Status result = CopyMatrixIfNeeded(cuda_stream, a, a_work);
      !result.ok()) {
    return return_after_cleanup(result);
  }
  if (absl::Status result = CopyMatrixIfNeeded(cuda_stream, b, b_out);
      !result.ok()) {
    return return_after_cleanup(result);
  }
  ffi::AnyBuffer a_forward = *a_work;
  ffi::AnyBuffer b_forward = *b_out;
  if (absl::Status result = ConvertRowMajorToColumnMajorInPlace(
          cuda_stream, "cusolvermp_gels/a_layout_convert", a_forward,
          scratch_base, scratch_elements);
      !result.ok()) {
    return return_after_cleanup(result);
  }
  if (absl::Status result = ConvertRowMajorToColumnMajorInPlace(
          cuda_stream, "cusolvermp_gels/b_layout_convert", b_forward,
          scratch_base, scratch_elements);
      !result.ok()) {
    return return_after_cleanup(result);
  }

  // Stage 4: redistribute A and B into cuSOLVERMp's block-cyclic layout.
  if (absl::Status result = ExecutePadded2DNativePlanRaw(
          "cusolvermp_gels/a_forward", stream, comm_stream, cuda_stream,
          process_rows, process_cols, tile_size, tile_size, m, n,
          /*reverse=*/0, rank_map, a_forward, a_work->device_memory(),
          scratch_base, scratch_elements, collective_params,
          collective_cliques);
      !result.ok()) {
    return return_after_cleanup(result);
  }
  if (absl::Status result = ExecutePadded2DNativePlanRaw(
          "cusolvermp_gels/b_forward", stream, comm_stream, cuda_stream,
          process_rows, process_cols, tile_size, tile_size, m,
          b_distribution_cols, /*reverse=*/0, rank_map, b_forward,
          b_out->device_memory(), scratch_base, scratch_elements,
          collective_params, collective_cliques);
      !result.ok()) {
    return return_after_cleanup(result);
  }
  if (absl::Status result = check_cuda(
          cudaStreamSynchronize(cuda_stream),
          "cusolvermp_gels forward stream synchronize");
      !result.ok()) {
    return result;
  }

  // Stage 5: execute GELS through the borrowed XLA communicator.
  if (absl::Status result = RunCusolverMpGelsSolver(
          stream, cuda_stream, process_rows, process_cols, m, n, nrhs,
          tile_size, grid_mapping, rank_map, *a_work, *b_out, a_work, b_out,
          status, collective_params, collective_cliques);
      !result.ok()) {
    return return_after_cleanup(result);
  }
  if (absl::Status result = check_cuda(
          cudaStreamSynchronize(cuda_stream),
          "cusolvermp_gels solver stream synchronize");
      !result.ok()) {
    return result;
  }

  // Stage 6: restore B; Python returns only its first N solution rows.
  if (absl::Status result = ExecutePadded2DNativePlanRaw(
          "cusolvermp_gels/b_reverse", stream, comm_stream, cuda_stream,
          process_rows, process_cols, tile_size, tile_size, m,
          b_distribution_cols, /*reverse=*/1, rank_map, *b_out,
          b_out->device_memory(), scratch_base, scratch_elements,
          collective_params, collective_cliques);
      !result.ok()) {
    return return_after_cleanup(result);
  }
  if (absl::Status result = ConvertColumnMajorToRowMajorInPlace(
          cuda_stream, "cusolvermp_gels/b_layout_restore", *b_out,
          scratch_base, scratch_elements);
      !result.ok()) {
    return return_after_cleanup(result);
  }
  return return_after_cleanup(absl::OkStatus());
}

}  // namespace xla::gpu
