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
// cuSOLVERMp reduced-QR FFI handler.
//
// This file implements the fused reduced decomposition A = Q * R for tall or
// square matrices. GEQRF stores R and Householder reflectors together in the
// donated A buffer. R is copied to its output before ORGQR overwrites that
// packed representation with the explicit reduced Q.
//
// File workflow:
//   1. Convert A to local column-major storage and redistribute it into the
//      cuSOLVERMp 2D block-cyclic layout.
//   2. Run GEQRF and preserve its distributed upper triangle as R.
//   3. Run ORGQR to replace the packed factorization with Q.
//   4. Reverse-redistribute Q and R and restore row-major JAX storage.

#include <algorithm>
#include <array>
#include <cstdlib>
#include <vector>

#include "cusolvermp_common.h"
#include "cusolvermp_routines.h"
#include "qr_extract.h"

namespace xla::gpu {
namespace {

template <typename DataType>
absl::Status RunCusolverMpQr(
    const CusolverMpApi& api, cusolverMpHandle_t handle, cusolverMpGrid_t grid,
    cudaStream_t cuda_stream, int64_t m, int64_t n, int64_t tile_size,
    int32_t process_row, int32_t process_col,
    ffi::Result<ffi::AnyBuffer> q_out, ffi::Result<ffi::AnyBuffer> r_out,
    std::array<int32_t, kQrStatusSize>* status_words) {
  const int debug_rank = (*status_words)[2];
  const int32_t process_rows = (*status_words)[4];
  const int32_t process_cols = (*status_words)[5];

  const int64_t q_numroc_rows =
      LocalNumroc(m, tile_size, process_row, process_rows);
  const int64_t q_numroc_cols =
      LocalNumroc(n, tile_size, process_col, process_cols);
  const int64_t r_numroc_rows =
      LocalNumroc(n, tile_size, process_row, process_rows);
  const int64_t r_numroc_cols = q_numroc_cols;
  (*status_words)[19] = static_cast<int32_t>(q_numroc_rows);
  (*status_words)[20] = static_cast<int32_t>(q_numroc_cols);
  (*status_words)[21] = static_cast<int32_t>(r_numroc_rows);
  (*status_words)[22] = static_cast<int32_t>(r_numroc_cols);
  (*status_words)[23] = static_cast<int32_t>(q_numroc_cols);
  if (q_out->dimensions()[0] < q_numroc_rows ||
      q_out->dimensions()[1] < q_numroc_cols ||
      r_out->dimensions()[0] < r_numroc_rows ||
      r_out->dimensions()[1] < r_numroc_cols) {
    (*status_words)[0] = kOutputShapeMismatch;
    return absl::OkStatus();
  }

  CusolverMpDebug(debug_rank,
                  "qr enter m=%lld n=%lld tile=%lld process_coord=(%d,%d) "
                  "dtype=%d",
                  static_cast<long long>(m), static_cast<long long>(n),
                  static_cast<long long>(tile_size), process_row, process_col,
                  static_cast<int>(SolverTraits<DataType>::cuda_data_type));

  cusolverMpMatrixDescriptor_t desc_q = nullptr;
  cusolverMpMatrixDescriptor_t desc_r = nullptr;
  void* d_tau = nullptr;
  void* d_work = nullptr;
  void* h_work = nullptr;
  int* d_info = nullptr;

  auto cleanup = [&]() {
    if (d_tau != nullptr) cudaFree(d_tau);
    if (d_work != nullptr) cudaFree(d_work);
    if (d_info != nullptr) cudaFree(d_info);
    if (h_work != nullptr) std::free(h_work);
    auto record_destroy_failure = [&](cusolverStatus_t destroy_status) {
      if (destroy_status != CUSOLVER_STATUS_SUCCESS &&
          (*status_words)[0] == kStatusOk) {
        (*status_words)[0] = kDestroyMatrixDescFailed;
        (*status_words)[12] = static_cast<int32_t>(destroy_status);
      }
    };
    if (desc_r != nullptr)
      record_destroy_failure(api.destroy_matrix_desc(desc_r));
    if (desc_q != nullptr)
      record_destroy_failure(api.destroy_matrix_desc(desc_q));
  };
  auto cuda_error_after_cleanup = [&](cudaError_t cuda_status,
                                      const char* caller) -> absl::Status {
    cleanup();
    return absl::InternalError(absl::StrFormat(
        "%s failed: cuda status %d (%s)", caller, static_cast<int>(cuda_status),
        cudaGetErrorString(cuda_status)));
  };

  cusolverStatus_t solver_status = api.create_matrix_desc(
      &desc_q, grid, SolverTraits<DataType>::cuda_data_type, m, n, tile_size,
      tile_size, /*RSRC_Q=*/0, /*CSRC_Q=*/0, q_out->dimensions()[0]);
  if (solver_status != CUSOLVER_STATUS_SUCCESS || desc_q == nullptr) {
    (*status_words)[0] = kCreateMatrixDescFailed;
    (*status_words)[12] = static_cast<int32_t>(solver_status);
    cleanup();
    return absl::OkStatus();
  }
  (*status_words)[10] = 1;

  solver_status = api.create_matrix_desc(
      &desc_r, grid, SolverTraits<DataType>::cuda_data_type, n, n, tile_size,
      tile_size, /*RSRC_R=*/0, /*CSRC_R=*/0, r_out->dimensions()[0]);
  if (solver_status != CUSOLVER_STATUS_SUCCESS || desc_r == nullptr) {
    (*status_words)[0] = kCreateMatrixDescFailed;
    (*status_words)[12] = static_cast<int32_t>(solver_status);
    cleanup();
    return absl::OkStatus();
  }
  (*status_words)[11] = 1;

  cudaError_t cuda_status = cudaMalloc(
      &d_tau, std::max<int64_t>(1, q_numroc_cols) * sizeof(DataType));
  if (cuda_status != cudaSuccess) {
    (*status_words)[0] = kDeviceAllocFailed;
    cleanup();
    return absl::OkStatus();
  }

  size_t geqrf_device = 0;
  size_t geqrf_host = 0;
  solver_status = api.geqrf_buffer_size(
      handle, m, n, q_out->untyped_data(), /*ia=*/1, /*ja=*/1, desc_q,
      SolverTraits<DataType>::cuda_data_type, &geqrf_device, &geqrf_host);
  if (solver_status != CUSOLVER_STATUS_SUCCESS) {
    (*status_words)[0] = kGeqrfWorkspaceFailed;
    (*status_words)[12] = static_cast<int32_t>(solver_status);
    cleanup();
    return absl::OkStatus();
  }
  (*status_words)[24] = SizeToKiBForStatus(geqrf_device);
  (*status_words)[25] = SizeToKiBForStatus(geqrf_host);

  size_t orgqr_device = 0;
  size_t orgqr_host = 0;
  solver_status = api.orgqr_buffer_size(
      handle, m, n, n, q_out->untyped_data(), /*ia=*/1, /*ja=*/1, desc_q,
      d_tau, SolverTraits<DataType>::cuda_data_type, &orgqr_device,
      &orgqr_host);
  if (solver_status != CUSOLVER_STATUS_SUCCESS) {
    (*status_words)[0] = kOrgqrWorkspaceFailed;
    (*status_words)[12] = static_cast<int32_t>(solver_status);
    cleanup();
    return absl::OkStatus();
  }
  (*status_words)[26] = SizeToKiBForStatus(orgqr_device);
  (*status_words)[27] = SizeToKiBForStatus(orgqr_host);

  const size_t workspace_device = std::max(geqrf_device, orgqr_device);
  const size_t workspace_host = std::max(geqrf_host, orgqr_host);
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

  auto reset_info = [&]() -> absl::Status {
    const cudaError_t status =
        cudaMemsetAsync(d_info, 0, sizeof(int), cuda_stream);
    if (status == cudaSuccess) return absl::OkStatus();
    return cuda_error_after_cleanup(status, "cusolvermp_qr info initialization");
  };
  auto read_info = [&](int32_t* destination,
                       const char* caller) -> absl::Status {
    cudaError_t status = cudaStreamSynchronize(cuda_stream);
    if (status != cudaSuccess) return cuda_error_after_cleanup(status, caller);
    int h_info = -1;
    status = cudaMemcpyAsync(&h_info, d_info, sizeof(int),
                             cudaMemcpyDeviceToHost, cuda_stream);
    if (status != cudaSuccess) return cuda_error_after_cleanup(status, caller);
    status = cudaStreamSynchronize(cuda_stream);
    if (status != cudaSuccess) return cuda_error_after_cleanup(status, caller);
    *destination = h_info;
    return absl::OkStatus();
  };

  // Stage 1: factor A into packed Householder reflectors and upper-triangular R.
  JAXMG_RETURN_IF_ERROR(reset_info());
  solver_status = api.geqrf(
      handle, m, n, q_out->untyped_data(), /*ia=*/1, /*ja=*/1, desc_q, d_tau,
      SolverTraits<DataType>::cuda_data_type, d_work, workspace_device, h_work,
      workspace_host, d_info);
  if (solver_status != CUSOLVER_STATUS_SUCCESS) {
    (*status_words)[0] = kGeqrfFailed;
    (*status_words)[12] = static_cast<int32_t>(solver_status);
    cleanup();
    return absl::OkStatus();
  }
  (*status_words)[28] = 1;
  JAXMG_RETURN_IF_ERROR(
      read_info(&(*status_words)[29], "cusolvermp_qr GEQRF synchronize"));
  if ((*status_words)[29] != 0) {
    (*status_words)[0] = kGeqrfInfoNonzero;
    cleanup();
    return absl::OkStatus();
  }

  // Stage 2: preserve R before ORGQR overwrites the packed GEQRF output.
  cuda_status = ExtractDistributedQrR(
      cuda_stream, SolverTraits<DataType>::cuda_data_type,
      q_out->untyped_data(), r_out->untyped_data(), n, tile_size, process_rows,
      process_cols, process_row, process_col, q_out->dimensions()[0],
      r_out->dimensions()[0], r_numroc_rows, r_numroc_cols);
  if (cuda_status != cudaSuccess) {
    (*status_words)[0] = kQrExtractFailed;
    cleanup();
    return absl::OkStatus();
  }
  cuda_status = cudaStreamSynchronize(cuda_stream);
  if (cuda_status != cudaSuccess) {
    return cuda_error_after_cleanup(cuda_status,
                                    "cusolvermp_qr R extraction synchronize");
  }
  (*status_words)[30] = 1;

  // Stage 3: materialize the reduced M-by-N Q in the donated A buffer.
  JAXMG_RETURN_IF_ERROR(reset_info());
  solver_status = api.orgqr(
      handle, m, n, n, q_out->untyped_data(), /*ia=*/1, /*ja=*/1, desc_q,
      d_tau, SolverTraits<DataType>::cuda_data_type, d_work, workspace_device,
      h_work, workspace_host, d_info);
  if (solver_status != CUSOLVER_STATUS_SUCCESS) {
    (*status_words)[0] = kOrgqrFailed;
    (*status_words)[12] = static_cast<int32_t>(solver_status);
    cleanup();
    return absl::OkStatus();
  }
  (*status_words)[31] = 1;
  JAXMG_RETURN_IF_ERROR(
      read_info(&(*status_words)[32], "cusolvermp_qr ORGQR synchronize"));
  if ((*status_words)[32] != 0) (*status_words)[0] = kOrgqrInfoNonzero;

  cleanup();
  return absl::OkStatus();
}

absl::Status RunCusolverMpQrSolver(
    se::Stream* stream, cudaStream_t cuda_stream, int64_t process_rows,
    int64_t process_cols, int64_t m, int64_t n, int64_t tile_size,
    int64_t grid_mapping, absl::Span<const int64_t> rank_map,
    ffi::Result<ffi::AnyBuffer> q, ffi::Result<ffi::AnyBuffer> r,
    ffi::Result<ffi::BufferR1<S32>> status_out,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  if (stream == nullptr || cuda_stream == nullptr) {
    return absl::InvalidArgumentError("cusolvermp_qr requires XLA and CUDA streams");
  }
  if (q->dimensions().size() != 2 || r->dimensions().size() != 2 ||
      q->element_type() != r->element_type()) {
    return absl::InvalidArgumentError(
        "cusolvermp_qr expects rank-2 Q/R outputs with matching dtypes");
  }
  if (status_out->dimensions().size() != 1 ||
      status_out->dimensions()[0] != kQrStatusSize) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "cusolvermp_qr expects status shape (%d,)", kQrStatusSize));
  }

  std::array<int32_t, kQrStatusSize> status_words = {
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
      0,   // Q descriptor created.
      0,   // R descriptor created.
      0,   // raw cuSOLVER status.
      static_cast<int32_t>(q->size_bytes()),
      static_cast<int32_t>(m),
      static_cast<int32_t>(n),
      static_cast<int32_t>(tile_size),
      static_cast<int32_t>(q->dimensions()[0]),
      static_cast<int32_t>(q->dimensions()[1]),
      -1,  // Q NUMROC rows.
      -1,  // Q NUMROC cols.
      -1,  // R NUMROC rows.
      -1,  // R NUMROC cols.
      -1,  // local tau elements.
      -1,  // GEQRF device workspace, KiB.
      -1,  // GEQRF host workspace, KiB.
      -1,  // ORGQR device workspace, KiB.
      -1,  // ORGQR host workspace, KiB.
      0,   // GEQRF called.
      -1,  // GEQRF info.
      0,   // R extracted.
      0,   // ORGQR called.
      -1,  // ORGQR info.
      -1,  // dtype code.
      static_cast<int32_t>(grid_mapping),
      0,  // reserved.
  };

  absl::StatusOr<int> buffer_device = DeviceForCudaPointer(q->untyped_data());
  if (!buffer_device.ok() || cudaSetDevice(*buffer_device) != cudaSuccess) {
    status_words[0] = kCudaDeviceFailed;
    return CopyQrStatusToDevice(stream, status_words, status_out);
  }
  const int cuda_device = *buffer_device;
  status_words[1] = cuda_device;

  if (collective_params == nullptr || collective_cliques == nullptr) {
    status_words[0] = kCollectiveContextMissing;
    return CopyQrStatusToDevice(stream, status_words, status_out);
  }
  absl::StatusOr<GpuCliqueKey> clique_key =
      AllAssignedDevicesP2PCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    status_words[0] = kCliqueKeyFailed;
    return CopyQrStatusToDevice(stream, status_words, status_out);
  }
  absl::StatusOr<GpuCommunicator*> gpu_comm = collective_cliques->GetComm(
      *clique_key, collective_params->global_device_id);
  if (!gpu_comm.ok() || *gpu_comm == nullptr) {
    status_words[0] = kCommunicatorMissing;
    return CopyQrStatusToDevice(stream, status_words, status_out);
  }
  void* platform_handle = (*gpu_comm)->platform_comm().handle;
  if (platform_handle == nullptr) {
    status_words[0] = kNcclHandleMissing;
    return CopyQrStatusToDevice(stream, status_words, status_out);
  }
  ncclComm_t nccl_comm = reinterpret_cast<ncclComm_t>(platform_handle);
  int nccl_rank = -1;
  int nccl_count = -1;
  if (ncclCommUserRank(nccl_comm, &nccl_rank) != ncclSuccess ||
      ncclCommCount(nccl_comm, &nccl_count) != ncclSuccess) {
    status_words[0] = kNcclRankMismatch;
    return CopyQrStatusToDevice(stream, status_words, status_out);
  }
  status_words[2] = nccl_rank;
  status_words[3] = nccl_count;

  if (process_rows <= 0 || process_cols <= 0 ||
      process_rows * process_cols != nccl_count || m < n || n <= 0 ||
      tile_size <= 0) {
    status_words[0] = kGridShapeMismatch;
    return CopyQrStatusToDevice(stream, status_words, status_out);
  }
  JAXMG_RETURN_IF_ERROR(ValidateCusolverMpGridMapping("cusolvermp_qr",
                                                      grid_mapping));
  if (!rank_map.empty()) {
    JAXMG_RETURN_IF_ERROR(ValidateStandardRankMapForGridMapping(
        "cusolvermp_qr", rank_map, process_rows, process_cols, grid_mapping));
  }

  CusolverMpApi api = LinkedCusolverMpApi(&status_words);
  cusolverMpHandle_t handle = nullptr;
  cusolverStatus_t solver_status =
      api.create(&handle, cuda_device, cuda_stream);
  if (solver_status != CUSOLVER_STATUS_SUCCESS || handle == nullptr) {
    status_words[0] = kCreateHandleFailed;
    status_words[12] = static_cast<int32_t>(solver_status);
    return CopyQrStatusToDevice(stream, status_words, status_out);
  }
  status_words[8] = 1;

  int version = -1;
  solver_status = api.get_version(handle, &version);
  if (solver_status != CUSOLVER_STATUS_SUCCESS) {
    status_words[0] = kGetVersionFailed;
    status_words[12] = static_cast<int32_t>(solver_status);
    api.destroy(handle);
    return CopyQrStatusToDevice(stream, status_words, status_out);
  }
  status_words[6] = version;

  cusolverMpGrid_t grid = nullptr;
  solver_status = api.create_grid(handle, &grid, nccl_comm,
                                  static_cast<int32_t>(process_rows),
                                  static_cast<int32_t>(process_cols),
                                  ToCusolverMpGridMapping(grid_mapping));
  if (solver_status != CUSOLVER_STATUS_SUCCESS || grid == nullptr) {
    status_words[0] = kCreateGridFailed;
    status_words[12] = static_cast<int32_t>(solver_status);
    api.destroy(handle);
    return CopyQrStatusToDevice(stream, status_words, status_out);
  }
  status_words[9] = 1;

  const auto [process_row, process_col] =
      ProcessCoordFromRank(nccl_rank, process_rows, process_cols, grid_mapping);
  absl::Status qr_status;
  switch (q->element_type()) {
    case F32:
      status_words[33] = 1;
      qr_status = RunCusolverMpQr<float>(
          api, handle, grid, cuda_stream, m, n, tile_size, process_row,
          process_col, q, r, &status_words);
      break;
    case F64:
      status_words[33] = 2;
      qr_status = RunCusolverMpQr<double>(
          api, handle, grid, cuda_stream, m, n, tile_size, process_row,
          process_col, q, r, &status_words);
      break;
    case C64:
      status_words[33] = 3;
      qr_status = RunCusolverMpQr<cuFloatComplex>(
          api, handle, grid, cuda_stream, m, n, tile_size, process_row,
          process_col, q, r, &status_words);
      break;
    case C128:
      status_words[33] = 4;
      qr_status = RunCusolverMpQr<cuDoubleComplex>(
          api, handle, grid, cuda_stream, m, n, tile_size, process_row,
          process_col, q, r, &status_words);
      break;
    default:
      status_words[0] = kUnsupportedDtype;
      qr_status = absl::OkStatus();
      break;
  }
  if (!qr_status.ok()) {
    api.destroy_grid(grid);
    api.destroy(handle);
    return qr_status;
  }

  if (status_words[0] == kStatusOk) {
    solver_status = api.destroy_grid(grid);
    if (solver_status != CUSOLVER_STATUS_SUCCESS) {
      status_words[0] = kDestroyGridFailed;
      status_words[12] = static_cast<int32_t>(solver_status);
    }
  } else {
    api.destroy_grid(grid);
  }
  if (status_words[0] == kStatusOk) {
    solver_status = api.destroy(handle);
    if (solver_status != CUSOLVER_STATUS_SUCCESS) {
      status_words[0] = kDestroyHandleFailed;
      status_words[12] = static_cast<int32_t>(solver_status);
    }
  } else {
    api.destroy(handle);
  }
  return CopyQrStatusToDevice(stream, status_words, status_out);
}

absl::Status RunCusolverMpQrDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    int64_t process_rows, int64_t process_cols, int64_t m, int64_t n,
    int64_t tile_size, absl::Span<const int64_t> partition_slots,
    ffi::AnyBuffer a, ffi::Result<ffi::AnyBuffer> q,
    ffi::Result<ffi::AnyBuffer> r,
    ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  absl::StatusOr<ResolvedProcessGrid> process_grid = ResolveProcessGrid(
      "cusolvermp_qr", collective_params, partition_slots, process_rows,
      process_cols);
  if (!process_grid.ok()) return process_grid.status();
  const int64_t grid_mapping = process_grid->grid_mapping;
  const absl::Span<const int64_t> rank_map = process_grid->rank_map;
  if (a.dimensions().size() != 2 || q->dimensions().size() != 2 ||
      r->dimensions().size() != 2) {
    return absl::InvalidArgumentError("cusolvermp_qr expects rank-2 buffers");
  }
  if (a.element_type() != q->element_type() ||
      a.element_type() != r->element_type()) {
    return absl::InvalidArgumentError(
        "cusolvermp_qr requires matching matrix dtypes");
  }
  if (a.dimensions()[0] != q->dimensions()[0] ||
      a.dimensions()[1] != q->dimensions()[1]) {
    return absl::InvalidArgumentError(
        "cusolvermp_qr input and Q shapes must match");
  }

  const size_t element_bytes =
      a.size_bytes() / static_cast<size_t>(a.element_count());
  const std::array<Padded2DRedistScratchRequest, 2> scratch_requests = {{
      {process_rows, process_cols, tile_size, tile_size, m, n,
       a.dimensions()[0], a.dimensions()[1], rank_map},
      {process_rows, process_cols, tile_size, tile_size, n, n,
       r->dimensions()[0], r->dimensions()[1], rank_map},
  }};
  absl::StatusOr<Padded2DRedistScratch> scratch_status =
      AllocatePadded2DRedistScratch(cuda_stream, element_bytes,
                                    absl::MakeConstSpan(scratch_requests),
                                    "cusolvermp_qr_redistribution");
  if (!scratch_status.ok()) return scratch_status.status();
  Padded2DRedistScratch scratch = *scratch_status;
  bool scratch_freed = false;
  auto return_after_cleanup = [&](absl::Status result) -> absl::Status {
    if (!scratch_freed) {
      absl::Status free_status = FreePadded2DRedistScratch(
          cuda_stream, scratch, "cusolvermp_qr_redistribution");
      scratch_freed = free_status.ok();
      if (result.ok() && !free_status.ok()) return free_status;
    }
    return result;
  };
  auto synchronize = [&](const char* caller) -> absl::Status {
    const cudaError_t cuda_status = cudaStreamSynchronize(cuda_stream);
    if (cuda_status == cudaSuccess) return absl::OkStatus();
    return return_after_cleanup(absl::InternalError(absl::StrFormat(
        "%s failed: cuda status %d (%s)", caller, static_cast<int>(cuda_status),
        cudaGetErrorString(cuda_status))));
  };

  // Donation normally aliases A to Q; preserve A explicitly when disabled.
  if (absl::Status copy_status = CopyMatrixIfNeeded(cuda_stream, a, q);
      !copy_status.ok()) {
    return return_after_cleanup(copy_status);
  }
  ffi::AnyBuffer q_work = *q;
  if (absl::Status convert_status = ConvertRowMajorToColumnMajorInPlace(
          cuda_stream, "cusolvermp_qr/a_layout_convert", q_work, scratch.base,
          scratch.elements);
      !convert_status.ok()) {
    return return_after_cleanup(convert_status);
  }
  if (absl::Status redist_status = ExecutePadded2DNativePlanRaw(
          "cusolvermp_qr/a_forward", stream, comm_stream, cuda_stream,
          process_rows, process_cols, tile_size, tile_size, m, n,
          /*reverse=*/0, rank_map, q_work, q->device_memory(), scratch.base,
          scratch.elements, collective_params, collective_cliques);
      !redist_status.ok()) {
    return return_after_cleanup(redist_status);
  }
  if (absl::Status sync_status =
          synchronize("cusolvermp_qr forward synchronize");
      !sync_status.ok()) {
    return sync_status;
  }

  if (absl::Status solver_status = RunCusolverMpQrSolver(
          stream, cuda_stream, process_rows, process_cols, m, n, tile_size,
          grid_mapping, rank_map, q, r, status, collective_params,
          collective_cliques);
      !solver_status.ok()) {
    return return_after_cleanup(solver_status);
  }
  if (absl::Status sync_status =
          synchronize("cusolvermp_qr solver synchronize");
      !sync_status.ok()) {
    return sync_status;
  }

  // Restore Q and R sequentially so both reuse the same redistribution scratch.
  if (absl::Status redist_status = ExecutePadded2DNativePlanRaw(
          "cusolvermp_qr/q_reverse", stream, comm_stream, cuda_stream,
          process_rows, process_cols, tile_size, tile_size, m, n,
          /*reverse=*/1, rank_map, *q, q->device_memory(), scratch.base,
          scratch.elements, collective_params, collective_cliques);
      !redist_status.ok()) {
    return return_after_cleanup(redist_status);
  }
  if (absl::Status convert_status = ConvertColumnMajorToRowMajorInPlace(
          cuda_stream, "cusolvermp_qr/q_layout_restore", *q, scratch.base,
          scratch.elements);
      !convert_status.ok()) {
    return return_after_cleanup(convert_status);
  }
  if (absl::Status sync_status = synchronize("cusolvermp_qr Q restore");
      !sync_status.ok()) {
    return sync_status;
  }
  if (absl::Status redist_status = ExecutePadded2DNativePlanRaw(
          "cusolvermp_qr/r_reverse", stream, comm_stream, cuda_stream,
          process_rows, process_cols, tile_size, tile_size, n, n,
          /*reverse=*/1, rank_map, *r, r->device_memory(), scratch.base,
          scratch.elements, collective_params, collective_cliques);
      !redist_status.ok()) {
    return return_after_cleanup(redist_status);
  }
  if (absl::Status convert_status = ConvertColumnMajorToRowMajorInPlace(
          cuda_stream, "cusolvermp_qr/r_layout_restore", *r, scratch.base,
          scratch.elements);
      !convert_status.ok()) {
    return return_after_cleanup(convert_status);
  }
  return return_after_cleanup(absl::OkStatus());
}

}  // namespace

absl::Status XlaCusolverMpQrPrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests) {
  return RequestAllAssignedP2PCommunicator(collective_params, clique_requests,
                                           "cusolvermp_qr");
}

absl::Status XlaCusolverMpQrDispatch(
    se::Stream* stream, cudaStream_t cuda_stream, int64_t process_rows,
    int64_t process_cols, int64_t m, int64_t n, int64_t tile_size,
    absl::Span<const int64_t> partition_slots, ffi::AnyBuffer a,
    ffi::Result<ffi::AnyBuffer> q, ffi::Result<ffi::AnyBuffer> r,
    ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  return RunCusolverMpQrDispatch(
      stream, /*comm_stream=*/nullptr, cuda_stream, process_rows, process_cols,
      m, n, tile_size, partition_slots, a, q, r, status,
      collective_params, collective_cliques);
}

}  // namespace xla::gpu
