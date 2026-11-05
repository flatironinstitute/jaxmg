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
// cuSOLVERMp symmetric/Hermitian eigensolver FFI handler.
//
// This file implements the fused SYEVD paths. Both modes convert a padded JAX
// shard to column-major storage, redistribute it into cuSOLVERMp's 2D
// block-cyclic layout, and run SYEVD through the XLA-owned NCCL communicator.
// The vector-producing path additionally restores the distributed
// eigenvectors to their original JAX layout.
//
// File workflow:
//   1. Validate the padded matrix and output-buffer contracts.
//   2. Allocate one scratch buffer shared by every redistribution stage.
//   3. Convert the local matrix from row-major to column-major.
//   4. Compact edge padding and redistribute the matrix into 2D block-cyclic
//      cuSOLVERMp storage.
//   5. Borrow the XLA communicator's NCCL handle and run eigenvalues-only or
//      vector-producing SYEVD.
//   6. For the vector-producing path, reverse the eigenvector redistribution.
//   7. Restore eigenvectors to local row-major JAX storage when requested.

#include <array>
#include <cstdlib>

#include "absl/cleanup/cleanup.h"

#include "cusolvermp_common.h"
#include "cusolvermp_routines.h"

namespace xla::gpu {
namespace {

// Executes cuSOLVERMp SYEVD on redistributed matrix storage.
//
// memory_redist has already populated work_out in local column-major 2D
// block-cyclic storage. cuSOLVERMp overwrites this work buffer as d_A, with
// vectors_out providing the separate d_Q output when eigenvectors are requested.
// For eigenvalues-only execution the unused
// Q arguments reuse d_A and descA, avoiding an additional matrix allocation
// without relying on undocumented null-pointer support. This helper owns
// descriptors, workspace allocation, the jobz selector, and info reporting.
template <typename DataType>
absl::Status RunCusolverMpSyevd(
    const CusolverMpApi& api, cusolverMpHandle_t handle,
    cusolverMpGrid_t grid, cudaStream_t cuda_stream, int64_t n,
    int64_t tile_size, int64_t local_physical_rows,
    int64_t local_physical_cols, int32_t process_row, int32_t process_col,
    ffi::Result<ffi::AnyBuffer> eigenvalues_out,
    ffi::Result<ffi::AnyBuffer> work_out,
    ffi::AnyBuffer* vectors_out, bool compute_eigenvectors,
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
  void* d_work = nullptr;
  void* h_work = nullptr;
  int* d_info = nullptr;
  // Release partial allocations and descriptors even when a CUDA check returns.
  absl::Cleanup cleanup = [&]() {
    if (d_work != nullptr) cudaFree(d_work);
    if (d_info != nullptr) cudaFree(d_info);
    if (h_work != nullptr) std::free(h_work);
    auto record_destroy_failure = [&](cusolverStatus_t result) {
      if (result != CUSOLVER_STATUS_SUCCESS && (*status_words)[0] == kStatusOk) {
        (*status_words)[0] = kDestroyMatrixDescFailed;
        (*status_words)[11] = static_cast<int32_t>(result);
      }
    };
    if (desc_q != nullptr && desc_q != desc_a) {
      record_destroy_failure(api.destroy_matrix_desc(desc_q));
    }
    if (desc_a != nullptr)
      record_destroy_failure(api.destroy_matrix_desc(desc_a));
  };
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
  desc_q = desc_a;

  if (compute_eigenvectors) {
    if (vectors_out == nullptr) {
      (*status_words)[0] = kOutputShapeMismatch;
      return absl::OkStatus();
    }
    CusolverMpDebug(debug_rank, "create desc_q begin");
    desc_q = nullptr;
    status = api.create_matrix_desc(
        &desc_q, grid, SolverTraits<DataType>::cuda_data_type, n, n, tile_size,
        tile_size, /*RSRC_Q=*/0, /*CSRC_Q=*/0, local_physical_rows);
    CusolverMpDebug(debug_rank, "create desc_q end status=%d desc=%p",
                    static_cast<int>(status), desc_q);
    if (status != CUSOLVER_STATUS_SUCCESS || desc_q == nullptr) {
      (*status_words)[0] = kCreateMatrixDescFailed;
      (*status_words)[11] = static_cast<int32_t>(status);
      return absl::OkStatus();
    }
    (*status_words)[27] = 1;
  }

  // cuSOLVERMp SYEVD overwrites d_A and writes eigenvectors to d_Q when
  // requested. Keep those buffers distinct in vector mode: the donated JAX
  // input aliases the work output used as d_A, while the public eigenvector
  // result is d_Q.
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));

  size_t workspace_device = 0;
  size_t workspace_host = 0;
  char jobz[] = {compute_eigenvectors ? 'V' : 'N', '\0'};
  void* q_data = compute_eigenvectors ? vectors_out->untyped_data()
                                      : work_out->untyped_data();
  CusolverMpDebug(
      debug_rank, "syevd_buffer_size begin jobz=%c a=%p evals=%p q=%p "
                  "desc_a=%p desc_q=%p",
      jobz[0], work_out->untyped_data(), eigenvalues_out->untyped_data(),
      q_data, desc_a, desc_q);
  status = api.syevd_buffer_size(
      handle, jobz, CUBLAS_FILL_MODE_LOWER, n, work_out->untyped_data(),
      /*IA=*/1, /*JA=*/1, desc_a, eigenvalues_out->untyped_data(),
      q_data, /*IQ=*/1, /*JQ=*/1, desc_q,
      SolverTraits<DataType>::cuda_data_type, &workspace_device,
      &workspace_host);
  CusolverMpDebug(debug_rank,
                  "syevd_buffer_size end status=%d workspace_device=%zu "
                  "workspace_host=%zu",
                  static_cast<int>(status), workspace_device, workspace_host);
  if (status != CUSOLVER_STATUS_SUCCESS) {
    (*status_words)[0] = kSyevdWorkspaceFailed;
    (*status_words)[11] = static_cast<int32_t>(status);
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
      return absl::OkStatus();
    }
  }
  if (workspace_host > 0) {
    CusolverMpDebug(debug_rank, "malloc h_work begin bytes=%zu", workspace_host);
    h_work = std::malloc(workspace_host);
    CusolverMpDebug(debug_rank, "malloc h_work end ptr=%p", h_work);
    if (h_work == nullptr) {
      (*status_words)[0] = kHostAllocFailed;
      return absl::OkStatus();
    }
  }
  CusolverMpDebug(debug_rank, "cudaMalloc d_info begin");
  cuda_status = cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int));
  CusolverMpDebug(debug_rank, "cudaMalloc d_info end status=%d ptr=%p",
                  static_cast<int>(cuda_status), d_info);
  if (cuda_status != cudaSuccess) {
    (*status_words)[0] = kDeviceAllocFailed;
    return absl::OkStatus();
  }
  JAXMG_RETURN_IF_CUDA_ERROR(
      cudaMemsetAsync(d_info, 0, sizeof(int), cuda_stream));
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));

  CusolverMpDebug(debug_rank, "syevd begin");
  status = api.syevd(handle, jobz, CUBLAS_FILL_MODE_LOWER, n,
                     work_out->untyped_data(), /*IA=*/1, /*JA=*/1, desc_a,
                     eigenvalues_out->untyped_data(), q_data, /*IQ=*/1,
                     /*JQ=*/1, desc_q,
                     SolverTraits<DataType>::cuda_data_type, d_work,
                     workspace_device, h_work, workspace_host, d_info);
  CusolverMpDebug(debug_rank, "syevd end status=%d", static_cast<int>(status));
  if (status != CUSOLVER_STATUS_SUCCESS) {
    (*status_words)[0] = kSyevdFailed;
    (*status_words)[11] = static_cast<int32_t>(status);
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
  return absl::OkStatus();
}

// Creates the borrowed-XLA-communicator cuSOLVERMp handle/grid and dispatches
// the dtype-specific SYEVD helper for the selected output mode.
absl::Status RunCusolverMpSyevdSolver(
    se::Stream* stream, cudaStream_t cuda_stream,
    int64_t process_rows, int64_t process_cols, int64_t n,
    int64_t tile_size, int64_t grid_mapping,
    absl::Span<const int64_t> rank_map,
    ffi::Result<ffi::AnyBuffer> eigenvalues_out,
    ffi::Result<ffi::AnyBuffer> work_out, ffi::AnyBuffer* vectors_out,
    bool compute_eigenvectors,
    ffi::Result<ffi::BufferR1<S32>> status_out,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  // Stage 1: validate the local FFI buffers. Both modes use the redistributed
  // work_out as cuSOLVERMp's overwritten d_A. Vector mode also
  // receives vectors_out as the separate d_Q eigenvector output.
  if (stream == nullptr || cuda_stream == nullptr) {
    return absl::InvalidArgumentError(
        "cusolvermp_syevd requires XLA and CUDA streams");
  }
  if (work_out->dimensions().size() != 2 ||
      eigenvalues_out->dimensions().size() != 1 ||
      (compute_eigenvectors &&
       (vectors_out == nullptr || vectors_out->dimensions().size() != 2))) {
    return absl::InvalidArgumentError(
        "cusolvermp_syevd expects rank-2 matrix buffers and rank-1 "
        "eigenvalue output");
  }
  if (compute_eigenvectors &&
      work_out->element_type() != vectors_out->element_type()) {
    return absl::InvalidArgumentError(
        "cusolvermp_syevd requires matching matrix/work/vector dtypes");
  }
  const PrimitiveType expected_eigen_type =
      (work_out->element_type() == F32 || work_out->element_type() == C64)
          ? F32 : F64;
  if (eigenvalues_out->element_type() != expected_eigen_type) {
    return absl::InvalidArgumentError(
        "cusolvermp_syevd received an eigenvalue output with the wrong dtype");
  }
  if (compute_eigenvectors &&
      (work_out->dimensions()[0] != vectors_out->dimensions()[0] ||
       work_out->dimensions()[1] != vectors_out->dimensions()[1])) {
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
      static_cast<int32_t>(work_out->size_bytes()),
      static_cast<int32_t>(n),
      static_cast<int32_t>(tile_size),
      static_cast<int32_t>(work_out->dimensions()[0]),
      static_cast<int32_t>(work_out->dimensions()[1]),
      -1,  // local NUMROC rows.
      -1,  // local NUMROC cols.
      static_cast<int32_t>(eigenvalues_out->size_bytes()),
      static_cast<int32_t>(compute_eigenvectors),  // Eigenvectors requested.
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

  // Publishes this rank's status. A failed solve leaves stale or partial
  // data in the results, so they are replaced with NaN first.
  auto publish_status = [&]() -> absl::Status {
    std::vector<ffi::AnyBuffer> results = {*eigenvalues_out};
    if (vectors_out != nullptr) results.push_back(*vectors_out);
    JAXMG_RETURN_IF_ERROR(
        InvalidateResultsOnFailure(cuda_stream, status_words[0], results));
    return CopyStatusToDevice(stream, status_words, status_out);
  };

  // Stage 2: bind the matrix's device and create the handle/grid using XLA's
  // borrowed communicator. Mathematical and output contracts stay routine-local.
  if (n <= 0 || tile_size <= 0) {
    status_words[0] = kGridShapeMismatch;
    return publish_status();
  }
  CusolverMpExecution execution(absl::MakeSpan(status_words));
  JAXMG_RETURN_IF_ERROR(execution.Initialize(
      "cusolvermp_syevd", cuda_stream, work_out->untyped_data(),
      process_rows, process_cols, grid_mapping, rank_map,
      collective_params, collective_cliques));
  if (status_words[0] != kStatusOk) {
    execution.Close();
    return publish_status();
  }
  const auto& api = execution.api;
  const auto handle = execution.handle;
  const auto grid = execution.grid;
  const auto nccl_comm = execution.nccl_comm;
  const int nccl_rank = execution.nccl_rank;

  const auto [process_row, process_col] = ProcessCoordFromRank(
      nccl_rank, process_rows, process_cols, grid_mapping);
  CusolverMpDebug(nccl_rank, "dispatch process_coord=(%d,%d) element_type=%d",
                  process_row, process_col,
                  static_cast<int>(work_out->element_type()));

  // Stage 3: dispatch on dtype and run the shared SYEVD helper. Vector mode
  // keeps d_A and d_Q separate; values-only mode does not allocate d_Q.
  absl::Status syevd_status;
  switch (work_out->element_type()) {
    case F32:
      status_words[25] = 1;
      syevd_status = RunCusolverMpSyevd<float>(
          api, handle, grid, cuda_stream, n, tile_size, work_out->dimensions()[0],
          work_out->dimensions()[1], process_row, process_col, eigenvalues_out,
          work_out, vectors_out, compute_eigenvectors, &status_words);
      break;
    case F64:
      status_words[25] = 2;
      syevd_status = RunCusolverMpSyevd<double>(
          api, handle, grid, cuda_stream, n, tile_size, work_out->dimensions()[0],
          work_out->dimensions()[1], process_row, process_col, eigenvalues_out,
          work_out, vectors_out, compute_eigenvectors, &status_words);
      break;
    case C64:
      status_words[25] = 3;
      syevd_status = RunCusolverMpSyevd<cuFloatComplex>(
          api, handle, grid, cuda_stream, n, tile_size, work_out->dimensions()[0],
          work_out->dimensions()[1], process_row, process_col, eigenvalues_out,
          work_out, vectors_out, compute_eigenvectors, &status_words);
      break;
    case C128:
      status_words[25] = 4;
      syevd_status = RunCusolverMpSyevd<cuDoubleComplex>(
          api, handle, grid, cuda_stream, n, tile_size, work_out->dimensions()[0],
          work_out->dimensions()[1], process_row, process_col, eigenvalues_out,
          work_out, vectors_out, compute_eigenvectors, &status_words);
      break;
    default:
      status_words[0] = kUnsupportedDtype;
      break;
  }
  if (!syevd_status.ok()) {
    return syevd_status;
  }

  // Stage 4: include cleanup failures in the collective result status.
  execution.Close();
  JAXMG_RETURN_IF_ERROR(SynchronizeSolverStatus(
      cuda_stream, nccl_comm, status_out, &status_words[0]));
  return publish_status();
}

}  // namespace

namespace {

// Executes the shared native SYEVD workflow for either output mode.
//
// Both paths prepare A in 2D block-cyclic form and invoke the same
// cuSOLVERMp/NCCL setup. Vector mode additionally receives a matrix-sized Q
// output and restores it to the original JAX layout. Values-only mode omits
// that output and ends after the solver has produced the replicated spectrum.
absl::Status RunCusolverMpSyevdDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    int64_t process_rows, int64_t process_cols, int64_t n, int64_t tile_size,
    absl::Span<const int64_t> partition_slots, ffi::AnyBuffer a,
    ffi::Result<ffi::AnyBuffer> eigenvalues, ffi::Result<ffi::AnyBuffer> work,
    ffi::AnyBuffer* vectors, bool compute_eigenvectors,
    ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  absl::StatusOr<ResolvedProcessGrid> process_grid =
      ResolveProcessGrid("cusolvermp_syevd", collective_params, partition_slots,
                         process_rows, process_cols);
  if (!process_grid.ok()) {
    return process_grid.status();
  }
  const int64_t grid_mapping = process_grid->grid_mapping;
  const absl::Span<const int64_t> rank_map = process_grid->rank_map;
  // Stage 1: validate the local FFI buffers. The Python wrapper has already
  // padded A to tile-aligned capacity. Work has the same local matrix capacity,
  // and vector mode supplies a matching Q output.
  if (a.dimensions().size() != 2 || work->dimensions().size() != 2 ||
      (compute_eigenvectors &&
       (vectors == nullptr || vectors->dimensions().size() != 2))) {
    return absl::InvalidArgumentError(
        "cusolvermp_syevd expects rank-2 matrix buffers");
  }
  if (a.element_type() != work->element_type() ||
      (compute_eigenvectors && a.element_type() != vectors->element_type())) {
    return absl::InvalidArgumentError(
        "cusolvermp_syevd requires matching matrix dtypes");
  }
  if (a.dimensions()[0] != work->dimensions()[0] ||
      a.dimensions()[1] != work->dimensions()[1] ||
      (compute_eigenvectors &&
       (a.dimensions()[0] != vectors->dimensions()[0] ||
        a.dimensions()[1] != vectors->dimensions()[1]))) {
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
  absl::StatusOr<Padded2DRedistScratch> redistribution_scratch_status =
      AllocatePadded2DRedistScratch(
          cuda_stream, element_bytes, absl::MakeConstSpan(scratch_requests),
          "cusolvermp_syevd_redistribution");
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
        cuda_stream, redistribution_scratch, "cusolvermp_syevd_redistribution");
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
  // Stage 3: copy/alias the input into work and convert the local storage from
  // JAX row-major to cuSOLVERMp column-major in place.  SYEVD overwrites d_A, so
  // the public input and the solver work buffer must stay conceptually
  // separate even when XLA can alias storage. If XLA cannot alias, `work`
  // remains the single full-size storage slot used by the fused call.
  if (absl::Status status = CopyMatrixIfNeeded(cuda_stream, a, work);
      !status.ok()) {
    return return_after_cleanup(status);
  }
  a_forward_input = *work;
  if (absl::Status status = ConvertRowMajorToColumnMajorInPlace(
          cuda_stream, "cusolvermp_syevd/a_layout_convert", a_forward_input,
          scratch_base, scratch_elements);
      !status.ok()) {
    return return_after_cleanup(status);
  }

  // Stage 4: redistribute A into cuSOLVERMp's 2D block-cyclic layout.
  if (absl::Status status = ExecutePadded2DNativePlanRaw(
          "cusolvermp_syevd/a_forward", stream, comm_stream, cuda_stream,
          process_rows, process_cols, tile_size, tile_size, n, n,
          /*reverse=*/0, rank_map, a_forward_input, work->device_memory(),
          scratch_base, scratch_elements, collective_params,
          collective_cliques);
      !status.ok()) {
    return return_after_cleanup(status);
  }

  // Complete pack, transfer, and unpack work before the host-side cuSOLVERMp
  // call reads the redistributed matrix.
  if (absl::Status status = cuda_status_after_cleanup(
          cudaStreamSynchronize(cuda_stream),
          "cusolvermp_syevd forward stream synchronize");
      !status.ok()) {
    return status;
  }

  // Stage 5: run SYEVD in the statically selected mode. The shared helper
  // creates a Q descriptor only when eigenvectors are requested.
  if (absl::Status solver_call_status = RunCusolverMpSyevdSolver(
          stream, cuda_stream, process_rows, process_cols, n,
          tile_size, grid_mapping, rank_map, eigenvalues, work,
          vectors, compute_eigenvectors, status, collective_params,
          collective_cliques);
      !solver_call_status.ok()) {
    return return_after_cleanup(solver_call_status);
  }

  // Values-only SYEVD has no matrix result to redistribute or restore.
  if (!compute_eigenvectors) {
    return return_after_cleanup(absl::OkStatus());
  }

  // Complete the solver call before reverse redistribution reads eigenvectors.
  if (absl::Status status = cuda_status_after_cleanup(
          cudaStreamSynchronize(cuda_stream),
          "cusolvermp_syevd solver stream synchronize");
      !status.ok()) {
    return status;
  }

  ffi::AnyBuffer vectors_cyclic = *vectors;
  // Stage 6: reverse-redistribute eigenvectors to the original JAX-facing shard
  // placement.  Eigenvalues are already a replicated rank-1 output from
  // cuSOLVERMp and do not need matrix redistribution.
  if (absl::Status status = ExecutePadded2DNativePlanRaw(
          "cusolvermp_syevd/vectors_reverse", stream, comm_stream,
          cuda_stream, process_rows, process_cols, tile_size, tile_size, n, n,
          /*reverse=*/1, rank_map, vectors_cyclic, vectors->device_memory(),
          scratch_base, scratch_elements, collective_params,
          collective_cliques);
      !status.ok()) {
    return return_after_cleanup(status);
  }

  // Stage 7: restore JAX's row-major local storage for the user-visible
  // eigenvector shard.
  if (absl::Status status = ConvertColumnMajorToRowMajorInPlace(
          cuda_stream, "cusolvermp_syevd/vectors_layout_restore", *vectors,
          scratch_base, scratch_elements);
      !status.ok()) {
    return return_after_cleanup(status);
  }
  return return_after_cleanup(absl::OkStatus());
}

}  // namespace

// Executes the complete vector-producing SYEVD workflow in one FFI dispatch.
// The overwritten matrix work buffer remains distinct from the eigenvector
// output required by the cuSOLVERMp API.
absl::Status XlaCusolverMpSyevdDispatch(
    se::Stream* stream, cudaStream_t cuda_stream, int64_t process_rows,
    int64_t process_cols, int64_t n, int64_t tile_size,
    absl::Span<const int64_t> partition_slots, ffi::AnyBuffer a,
    ffi::Result<ffi::AnyBuffer> eigenvalues, ffi::Result<ffi::AnyBuffer> work,
    ffi::Result<ffi::AnyBuffer> vectors, ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  ffi::AnyBuffer vector_buffer = *vectors;
  return RunCusolverMpSyevdDispatch(
      stream, /*comm_stream=*/nullptr, cuda_stream, process_rows, process_cols,
      n, tile_size, partition_slots, a, eigenvalues, work, &vector_buffer,
      /*compute_eigenvectors=*/true, status, collective_params,
      collective_cliques);
}

// Executes eigenvalues-only SYEVD without allocating or restoring a
// matrix-sized eigenvector output.
absl::Status XlaCusolverMpSyevdValuesDispatch(
    se::Stream* stream, cudaStream_t cuda_stream, int64_t process_rows,
    int64_t process_cols, int64_t n, int64_t tile_size,
    absl::Span<const int64_t> partition_slots, ffi::AnyBuffer a,
    ffi::Result<ffi::AnyBuffer> eigenvalues, ffi::Result<ffi::AnyBuffer> work,
    ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  return RunCusolverMpSyevdDispatch(
      stream, /*comm_stream=*/nullptr, cuda_stream, process_rows, process_cols,
      n, tile_size, partition_slots, a, eigenvalues, work,
      /*vectors=*/nullptr, /*compute_eigenvectors=*/false, status,
      collective_params, collective_cliques);
}

}  // namespace xla::gpu
