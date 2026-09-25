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
// cuSOLVERMp polar-decomposition FFI handlers.
//
// This file implements the fused polar workflow for tall or square matrices.
// The two exported handlers compute Up alone or both factors in A = Up * H,
// without asking XLA to allocate the optional H output when it is not needed.
//
// File workflow:
//   1. Validate the input and optional H output-buffer contracts.
//   2. Convert A to local column-major storage and redistribute it into
//      cuSOLVERMp's 2D block-cyclic layout.
//   3. Run cusolverMpPolar, which overwrites A with Up and optionally writes H.
//   4. Reverse-redistribute the requested factors and restore row-major JAX
//      storage.

#include <array>
#include <cstdlib>
#include <vector>

#include "cusolvermp_common.h"
#include "cusolvermp_routines.h"

namespace xla::gpu {
namespace {

// Owns matrix descriptors and solver workspace for one distributed polar
// decomposition. A has already been redistributed into 2D block-cyclic
// column-major storage and is overwritten directly by Up.
template <typename DataType>
absl::Status RunCusolverMpPolar(
    const CusolverMpApi& api, cusolverMpHandle_t handle, cusolverMpGrid_t grid,
    cudaStream_t cuda_stream, int64_t m, int64_t n, int64_t tile_size,
    int32_t process_row, int32_t process_col,
    ffi::Result<ffi::AnyBuffer> up_out, ffi::AnyBuffer* h_out, bool compute_h,
    std::array<int32_t, kPolarStatusSize>* status_words) {
  const int debug_rank = (*status_words)[2];
  const int32_t process_rows = (*status_words)[4];
  const int32_t process_cols = (*status_words)[5];

  // cuSOLVERMp descriptors describe global matrices while each rank stores only
  // its NUMROC-owned rows and columns. Confirm that the padded JAX buffers have
  // enough local capacity before descriptor creation.
  const int64_t a_numroc_rows =
      LocalNumroc(m, tile_size, process_row, process_rows);
  const int64_t a_numroc_cols =
      LocalNumroc(n, tile_size, process_col, process_cols);
  (*status_words)[18] = static_cast<int32_t>(a_numroc_rows);
  (*status_words)[19] = static_cast<int32_t>(a_numroc_cols);
  if (up_out->dimensions()[0] < a_numroc_rows ||
      up_out->dimensions()[1] < a_numroc_cols) {
    (*status_words)[0] = kOutputShapeMismatch;
    return absl::OkStatus();
  }

  if (compute_h) {
    const int64_t h_numroc_rows =
        LocalNumroc(n, tile_size, process_row, process_rows);
    const int64_t h_numroc_cols =
        LocalNumroc(n, tile_size, process_col, process_cols);
    (*status_words)[22] = static_cast<int32_t>(h_numroc_rows);
    (*status_words)[23] = static_cast<int32_t>(h_numroc_cols);
    if (h_out == nullptr || h_out->dimensions()[0] < h_numroc_rows ||
        h_out->dimensions()[1] < h_numroc_cols) {
      (*status_words)[0] = kOutputShapeMismatch;
      return absl::OkStatus();
    }
  }

  CusolverMpDebug(debug_rank,
                  "polar enter m=%lld n=%lld tile=%lld process_coord=(%d,%d) "
                  "compute_h=%d dtype=%d",
                  static_cast<long long>(m), static_cast<long long>(n),
                  static_cast<long long>(tile_size), process_row, process_col,
                  compute_h ? 1 : 0,
                  static_cast<int>(SolverTraits<DataType>::cuda_data_type));

  cusolverMpMatrixDescriptor_t desc_a = nullptr;
  cusolverMpMatrixDescriptor_t desc_h = nullptr;
  void* d_work = nullptr;
  void* h_work = nullptr;
  int* d_info = nullptr;

  auto cleanup = [&]() {
    if (d_work != nullptr) cudaFree(d_work);
    if (d_info != nullptr) cudaFree(d_info);
    if (h_work != nullptr) std::free(h_work);
    auto record_destroy_failure = [&](cusolverStatus_t destroy_status) {
      if (destroy_status != CUSOLVER_STATUS_SUCCESS &&
          (*status_words)[0] == kStatusOk) {
        (*status_words)[0] = kDestroyMatrixDescFailed;
        (*status_words)[11] = static_cast<int32_t>(destroy_status);
      }
    };
    if (desc_h != nullptr)
      record_destroy_failure(api.destroy_matrix_desc(desc_h));
    if (desc_a != nullptr)
      record_destroy_failure(api.destroy_matrix_desc(desc_a));
  };
  auto cuda_error_after_cleanup = [&](cudaError_t cuda_status,
                                      const char* caller) -> absl::Status {
    cleanup();
    return absl::InternalError(absl::StrFormat(
        "%s failed: cuda status %d (%s)", caller, static_cast<int>(cuda_status),
        cudaGetErrorString(cuda_status)));
  };

  cusolverStatus_t solver_status = api.create_matrix_desc(
      &desc_a, grid, SolverTraits<DataType>::cuda_data_type, m, n, tile_size,
      tile_size, /*RSRC_A=*/0, /*CSRC_A=*/0, up_out->dimensions()[0]);
  if (solver_status != CUSOLVER_STATUS_SUCCESS || desc_a == nullptr) {
    (*status_words)[0] = kCreateMatrixDescFailed;
    (*status_words)[11] = static_cast<int32_t>(solver_status);
    cleanup();
    return absl::OkStatus();
  }
  (*status_words)[10] = 1;

  if (compute_h) {
    solver_status = api.create_matrix_desc(
        &desc_h, grid, SolverTraits<DataType>::cuda_data_type, n, n, tile_size,
        tile_size, /*RSRC_H=*/0, /*CSRC_H=*/0, h_out->dimensions()[0]);
    if (solver_status != CUSOLVER_STATUS_SUCCESS || desc_h == nullptr) {
      (*status_words)[0] = kCreateMatrixDescFailed;
      (*status_words)[11] = static_cast<int32_t>(solver_status);
      cleanup();
      return absl::OkStatus();
    }
    (*status_words)[21] = 1;
  }

  void* h_data = compute_h ? h_out->untyped_data() : nullptr;
  size_t workspace_device = 0;
  size_t workspace_host = 0;
  solver_status = api.polar_buffer_size(
      handle, /*polarDesc=*/nullptr, CUBLAS_FILL_MODE_FULL, m, n,
      up_out->untyped_data(), /*ia=*/1, /*ja=*/1, desc_a, h_data,
      /*ih=*/1, /*jh=*/1, desc_h, SolverTraits<DataType>::cuda_data_type,
      &workspace_device, &workspace_host);
  if (solver_status != CUSOLVER_STATUS_SUCCESS) {
    (*status_words)[0] = kPolarWorkspaceFailed;
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
  cuda_status = cudaMemsetAsync(d_info, 0, sizeof(int), cuda_stream);
  if (cuda_status != cudaSuccess) {
    return cuda_error_after_cleanup(cuda_status,
                                    "cusolvermp_polar info initialization");
  }
  cuda_status = cudaStreamSynchronize(cuda_stream);
  if (cuda_status != cudaSuccess) {
    return cuda_error_after_cleanup(cuda_status,
                                    "cusolvermp_polar info synchronize");
  }

  // cuSOLVERMp overwrites A with Up and writes H only when descH is present.
  CusolverMpDebug(debug_rank, "polar begin");
  solver_status = api.polar(
      handle, /*polarDesc=*/nullptr, CUBLAS_FILL_MODE_FULL, m, n,
      up_out->untyped_data(), /*ia=*/1, /*ja=*/1, desc_a, h_data,
      /*ih=*/1, /*jh=*/1, desc_h, SolverTraits<DataType>::cuda_data_type,
      d_work, workspace_device, h_work, workspace_host, d_info);
  CusolverMpDebug(debug_rank, "polar end status=%d",
                  static_cast<int>(solver_status));
  if (solver_status != CUSOLVER_STATUS_SUCCESS) {
    (*status_words)[0] = kPolarFailed;
    (*status_words)[11] = static_cast<int32_t>(solver_status);
    cleanup();
    return absl::OkStatus();
  }
  (*status_words)[26] = 1;
  cuda_status = cudaStreamSynchronize(cuda_stream);
  if (cuda_status != cudaSuccess) {
    return cuda_error_after_cleanup(cuda_status,
                                    "cusolvermp_polar solver synchronize");
  }

  int h_info = -1;
  cuda_status = cudaMemcpyAsync(&h_info, d_info, sizeof(int),
                                cudaMemcpyDeviceToHost, cuda_stream);
  if (cuda_status != cudaSuccess) {
    return cuda_error_after_cleanup(cuda_status, "cusolvermp_polar info copy");
  }
  cuda_status = cudaStreamSynchronize(cuda_stream);
  if (cuda_status != cudaSuccess) {
    return cuda_error_after_cleanup(cuda_status,
                                    "cusolvermp_polar info copy synchronize");
  }
  (*status_words)[27] = h_info;
  if (h_info != 0) (*status_words)[0] = kPolarInfoNonzero;

  cleanup();
  return absl::OkStatus();
}

// Binds the polar operation to this rank's CUDA device and XLA-owned NCCL
// communicator, then dispatches the matrix scalar type.
absl::Status RunCusolverMpPolarSolver(
    se::Stream* stream, cudaStream_t cuda_stream, int64_t process_rows,
    int64_t process_cols, int64_t m, int64_t n, int64_t tile_size,
    int64_t grid_mapping, absl::Span<const int64_t> rank_map,
    ffi::Result<ffi::AnyBuffer> up, ffi::AnyBuffer* h, bool compute_h,
    ffi::Result<ffi::BufferR1<S32>> status_out,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  if (stream == nullptr || cuda_stream == nullptr) {
    return absl::InvalidArgumentError(
        "cusolvermp_polar requires XLA and CUDA streams");
  }
  if (up->dimensions().size() != 2 ||
      (compute_h && (h == nullptr || h->dimensions().size() != 2))) {
    return absl::InvalidArgumentError(
        "cusolvermp_polar expects rank-2 matrix outputs");
  }
  if (compute_h && up->element_type() != h->element_type()) {
    return absl::InvalidArgumentError(
        "cusolvermp_polar requires matching output dtypes");
  }
  if (status_out->dimensions().size() != 1 ||
      status_out->dimensions()[0] != kPolarStatusSize) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "cusolvermp_polar expects status shape (%d,)", kPolarStatusSize));
  }

  std::array<int32_t, kPolarStatusSize> status_words = {
      kStatusOk,
      -1,  // CUDA device selected for this FFI invocation.
      -1,  // NCCL rank reported by the borrowed communicator.
      -1,  // NCCL communicator size.
      static_cast<int32_t>(process_rows),
      static_cast<int32_t>(process_cols),
      -1,  // cuSOLVERMp version.
      0,   // linked cuSOLVERMp runtime available.
      0,   // cusolverMpHandle_t created.
      0,   // cusolverMpGrid_t created.
      0,   // matrix descriptor for A/Up created.
      0,   // raw cuSOLVER status from the failing call, if any.
      static_cast<int32_t>(up->size_bytes()),
      static_cast<int32_t>(m),
      static_cast<int32_t>(n),
      static_cast<int32_t>(tile_size),
      static_cast<int32_t>(up->dimensions()[0]),
      static_cast<int32_t>(up->dimensions()[1]),
      -1,  // local NUMROC rows for A/Up.
      -1,  // local NUMROC cols for A/Up.
      static_cast<int32_t>(compute_h),
      0,   // matrix descriptor for H created.
      -1,  // local NUMROC rows for H.
      -1,  // local NUMROC cols for H.
      -1,  // polar device workspace, KiB.
      -1,  // polar host workspace, KiB.
      0,   // cusolverMpPolar called.
      -1,  // polar info value copied from device.
      -1,  // dtype code.
      static_cast<int32_t>(grid_mapping),
      0,  // reserved.
      0,  // reserved.
  };

  absl::StatusOr<int> buffer_device = DeviceForCudaPointer(up->untyped_data());
  if (!buffer_device.ok() || cudaSetDevice(*buffer_device) != cudaSuccess) {
    status_words[0] = kCudaDeviceFailed;
    return CopyPolarStatusToDevice(stream, status_words, status_out);
  }
  const int cuda_device = *buffer_device;
  status_words[1] = cuda_device;

  if (collective_params == nullptr || collective_cliques == nullptr) {
    status_words[0] = kCollectiveContextMissing;
    return CopyPolarStatusToDevice(stream, status_words, status_out);
  }
  absl::StatusOr<GpuCliqueKey> clique_key =
      AllAssignedDevicesP2PCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    status_words[0] = kCliqueKeyFailed;
    return CopyPolarStatusToDevice(stream, status_words, status_out);
  }
  absl::StatusOr<GpuCommunicator*> gpu_comm = collective_cliques->GetComm(
      *clique_key, collective_params->global_device_id);
  if (!gpu_comm.ok() || *gpu_comm == nullptr) {
    status_words[0] = kCommunicatorMissing;
    return CopyPolarStatusToDevice(stream, status_words, status_out);
  }
  void* platform_handle = (*gpu_comm)->platform_comm().handle;
  if (platform_handle == nullptr) {
    status_words[0] = kNcclHandleMissing;
    return CopyPolarStatusToDevice(stream, status_words, status_out);
  }
  ncclComm_t nccl_comm = reinterpret_cast<ncclComm_t>(platform_handle);
  int nccl_rank = -1;
  int nccl_count = -1;
  if (ncclCommUserRank(nccl_comm, &nccl_rank) != ncclSuccess ||
      ncclCommCount(nccl_comm, &nccl_count) != ncclSuccess) {
    status_words[0] = kNcclRankMismatch;
    return CopyPolarStatusToDevice(stream, status_words, status_out);
  }
  status_words[2] = nccl_rank;
  status_words[3] = nccl_count;

  if (process_rows <= 0 || process_cols <= 0 ||
      process_rows * process_cols != nccl_count || m < n || n <= 0 ||
      tile_size <= 0) {
    status_words[0] = kGridShapeMismatch;
    return CopyPolarStatusToDevice(stream, status_words, status_out);
  }
  JAXMG_RETURN_IF_ERROR(
      ValidateCusolverMpGridMapping("cusolvermp_polar", grid_mapping));
  if (!rank_map.empty()) {
    JAXMG_RETURN_IF_ERROR(ValidateStandardRankMapForGridMapping(
        "cusolvermp_polar", rank_map, process_rows, process_cols,
        grid_mapping));
  }

  CusolverMpApi api = LinkedCusolverMpApi(&status_words);
  cusolverMpHandle_t handle = nullptr;
  cusolverStatus_t solver_status =
      api.create(&handle, cuda_device, cuda_stream);
  if (solver_status != CUSOLVER_STATUS_SUCCESS || handle == nullptr) {
    status_words[0] = kCreateHandleFailed;
    status_words[11] = static_cast<int32_t>(solver_status);
    return CopyPolarStatusToDevice(stream, status_words, status_out);
  }
  status_words[8] = 1;

  int version = -1;
  solver_status = api.get_version(handle, &version);
  if (solver_status != CUSOLVER_STATUS_SUCCESS) {
    status_words[0] = kGetVersionFailed;
    status_words[11] = static_cast<int32_t>(solver_status);
    api.destroy(handle);
    return CopyPolarStatusToDevice(stream, status_words, status_out);
  }
  status_words[6] = version;

  cusolverMpGrid_t grid = nullptr;
  solver_status = api.create_grid(handle, &grid, nccl_comm,
                                  static_cast<int32_t>(process_rows),
                                  static_cast<int32_t>(process_cols),
                                  ToCusolverMpGridMapping(grid_mapping));
  if (solver_status != CUSOLVER_STATUS_SUCCESS || grid == nullptr) {
    status_words[0] = kCreateGridFailed;
    status_words[11] = static_cast<int32_t>(solver_status);
    api.destroy(handle);
    return CopyPolarStatusToDevice(stream, status_words, status_out);
  }
  status_words[9] = 1;

  const auto [process_row, process_col] =
      ProcessCoordFromRank(nccl_rank, process_rows, process_cols, grid_mapping);
  absl::Status polar_status;
  switch (up->element_type()) {
    case F32:
      status_words[28] = 1;
      polar_status = RunCusolverMpPolar<float>(
          api, handle, grid, cuda_stream, m, n, tile_size, process_row,
          process_col, up, h, compute_h, &status_words);
      break;
    case F64:
      status_words[28] = 2;
      polar_status = RunCusolverMpPolar<double>(
          api, handle, grid, cuda_stream, m, n, tile_size, process_row,
          process_col, up, h, compute_h, &status_words);
      break;
    case C64:
      status_words[28] = 3;
      polar_status = RunCusolverMpPolar<cuFloatComplex>(
          api, handle, grid, cuda_stream, m, n, tile_size, process_row,
          process_col, up, h, compute_h, &status_words);
      break;
    case C128:
      status_words[28] = 4;
      polar_status = RunCusolverMpPolar<cuDoubleComplex>(
          api, handle, grid, cuda_stream, m, n, tile_size, process_row,
          process_col, up, h, compute_h, &status_words);
      break;
    default:
      status_words[0] = kUnsupportedDtype;
      polar_status = absl::OkStatus();
      break;
  }
  if (!polar_status.ok()) {
    api.destroy_grid(grid);
    api.destroy(handle);
    return polar_status;
  }

  if (status_words[0] == kStatusOk) {
    solver_status = api.destroy_grid(grid);
    if (solver_status != CUSOLVER_STATUS_SUCCESS) {
      status_words[0] = kDestroyGridFailed;
      status_words[11] = static_cast<int32_t>(solver_status);
    }
  } else {
    api.destroy_grid(grid);
  }
  if (status_words[0] == kStatusOk) {
    solver_status = api.destroy(handle);
    if (solver_status != CUSOLVER_STATUS_SUCCESS) {
      status_words[0] = kDestroyHandleFailed;
      status_words[11] = static_cast<int32_t>(solver_status);
    }
  } else {
    api.destroy(handle);
  }
  return CopyPolarStatusToDevice(stream, status_words, status_out);
}

// Executes the complete fused workflow. The same scratch allocation is reused
// for A/Up and optional H because the factors are restored sequentially.
absl::Status RunCusolverMpPolarDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    int64_t process_rows, int64_t process_cols, int64_t m, int64_t n,
    int64_t tile_size, int64_t grid_mapping, absl::Span<const int64_t> rank_map,
    ffi::AnyBuffer a, ffi::Result<ffi::AnyBuffer> up, ffi::AnyBuffer* h,
    bool compute_h, ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  if (a.dimensions().size() != 2 || up->dimensions().size() != 2 ||
      (compute_h && (h == nullptr || h->dimensions().size() != 2))) {
    return absl::InvalidArgumentError(
        "cusolvermp_polar expects rank-2 matrix buffers");
  }
  if (a.element_type() != up->element_type() ||
      (compute_h && a.element_type() != h->element_type())) {
    return absl::InvalidArgumentError(
        "cusolvermp_polar requires matching matrix dtypes");
  }
  if (a.dimensions()[0] != up->dimensions()[0] ||
      a.dimensions()[1] != up->dimensions()[1]) {
    return absl::InvalidArgumentError(
        "cusolvermp_polar input and Up shapes must match");
  }

  const size_t element_bytes =
      a.size_bytes() / static_cast<size_t>(a.element_count());
  std::vector<Padded2DRedistScratchRequest> scratch_requests;
  scratch_requests.reserve(compute_h ? 2 : 1);
  scratch_requests.push_back(Padded2DRedistScratchRequest{
      process_rows, process_cols, tile_size, tile_size, m, n, a.dimensions()[0],
      a.dimensions()[1], rank_map});
  if (compute_h) {
    scratch_requests.push_back(Padded2DRedistScratchRequest{
        process_rows, process_cols, tile_size, tile_size, n, n,
        h->dimensions()[0], h->dimensions()[1], rank_map});
  }

  absl::StatusOr<Padded2DRedistScratch> scratch_status =
      AllocatePadded2DRedistScratch(cuda_stream, element_bytes,
                                    absl::MakeConstSpan(scratch_requests),
                                    "cusolvermp_polar_redistribution");
  if (!scratch_status.ok()) return scratch_status.status();
  Padded2DRedistScratch scratch = *scratch_status;
  bool scratch_freed = false;
  auto return_after_cleanup = [&](absl::Status result) -> absl::Status {
    if (!scratch_freed) {
      absl::Status free_status = FreePadded2DRedistScratch(
          cuda_stream, scratch, "cusolvermp_polar_redistribution");
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

  // Donation normally aliases A to Up. With donation disabled, copy A into the
  // Up output before applying the destructive native layout transformation.
  if (absl::Status copy_status = CopyMatrixIfNeeded(cuda_stream, a, up);
      !copy_status.ok()) {
    return return_after_cleanup(copy_status);
  }
  ffi::AnyBuffer up_work = *up;
  if (absl::Status convert_status = ConvertRowMajorToColumnMajorInPlace(
          cuda_stream, "cusolvermp_polar/a_layout_convert", up_work,
          scratch.base, scratch.elements);
      !convert_status.ok()) {
    return return_after_cleanup(convert_status);
  }
  if (absl::Status redist_status = ExecutePadded2DNativePlanRaw(
          "cusolvermp_polar/a_forward", stream, comm_stream, cuda_stream,
          process_rows, process_cols, tile_size, tile_size, m, n,
          /*reverse=*/0, rank_map, up_work, up->device_memory(), scratch.base,
          scratch.elements, collective_params, collective_cliques);
      !redist_status.ok()) {
    return return_after_cleanup(redist_status);
  }
  if (absl::Status sync_status =
          synchronize("cusolvermp_polar forward stream synchronize");
      !sync_status.ok()) {
    return sync_status;
  }

  if (absl::Status solver_status = RunCusolverMpPolarSolver(
          stream, cuda_stream, process_rows, process_cols, m, n, tile_size,
          grid_mapping, rank_map, up, h, compute_h, status, collective_params,
          collective_cliques);
      !solver_status.ok()) {
    return return_after_cleanup(solver_status);
  }
  if (absl::Status sync_status =
          synchronize("cusolvermp_polar solver stream synchronize");
      !sync_status.ok()) {
    return sync_status;
  }

  // Up occupies the overwritten A buffer and therefore follows the inverse of
  // A's redistribution and local layout conversion.
  if (absl::Status redist_status = ExecutePadded2DNativePlanRaw(
          "cusolvermp_polar/up_reverse", stream, comm_stream, cuda_stream,
          process_rows, process_cols, tile_size, tile_size, m, n,
          /*reverse=*/1, rank_map, *up, up->device_memory(), scratch.base,
          scratch.elements, collective_params, collective_cliques);
      !redist_status.ok()) {
    return return_after_cleanup(redist_status);
  }
  if (absl::Status convert_status = ConvertColumnMajorToRowMajorInPlace(
          cuda_stream, "cusolvermp_polar/up_layout_restore", *up, scratch.base,
          scratch.elements);
      !convert_status.ok()) {
    return return_after_cleanup(convert_status);
  }

  if (compute_h) {
    if (absl::Status sync_status =
            synchronize("cusolvermp_polar Up restore synchronize");
        !sync_status.ok()) {
      return sync_status;
    }
    if (absl::Status redist_status = ExecutePadded2DNativePlanRaw(
            "cusolvermp_polar/h_reverse", stream, comm_stream, cuda_stream,
            process_rows, process_cols, tile_size, tile_size, n, n,
            /*reverse=*/1, rank_map, *h, h->device_memory(), scratch.base,
            scratch.elements, collective_params, collective_cliques);
        !redist_status.ok()) {
      return return_after_cleanup(redist_status);
    }
    if (absl::Status convert_status = ConvertColumnMajorToRowMajorInPlace(
            cuda_stream, "cusolvermp_polar/h_layout_restore", *h, scratch.base,
            scratch.elements);
        !convert_status.ok()) {
      return return_after_cleanup(convert_status);
    }
  }
  return return_after_cleanup(absl::OkStatus());
}

}  // namespace

absl::Status XlaCusolverMpPolarPrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests) {
  return RequestAllAssignedP2PCommunicator(collective_params, clique_requests,
                                           "cusolvermp_polar");
}

absl::Status XlaCusolverMpPolarUhDispatch(
    se::Stream* stream, cudaStream_t cuda_stream, int64_t process_rows,
    int64_t process_cols, int64_t m, int64_t n, int64_t tile_size,
    int64_t grid_mapping, absl::Span<const int64_t> rank_map, ffi::AnyBuffer a,
    ffi::Result<ffi::AnyBuffer> up, ffi::Result<ffi::AnyBuffer> h,
    ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  ffi::AnyBuffer h_buffer = *h;
  return RunCusolverMpPolarDispatch(
      stream, /*comm_stream=*/nullptr, cuda_stream, process_rows, process_cols,
      m, n, tile_size, grid_mapping, rank_map, a, up, &h_buffer,
      /*compute_h=*/true, status, collective_params, collective_cliques);
}

absl::Status XlaCusolverMpPolarUDispatch(
    se::Stream* stream, cudaStream_t cuda_stream, int64_t process_rows,
    int64_t process_cols, int64_t m, int64_t n, int64_t tile_size,
    int64_t grid_mapping, absl::Span<const int64_t> rank_map, ffi::AnyBuffer a,
    ffi::Result<ffi::AnyBuffer> up, ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  return RunCusolverMpPolarDispatch(
      stream, /*comm_stream=*/nullptr, cuda_stream, process_rows, process_cols,
      m, n, tile_size, grid_mapping, rank_map, a, up, /*h=*/nullptr,
      /*compute_h=*/false, status, collective_params, collective_cliques);
}

}  // namespace xla::gpu
