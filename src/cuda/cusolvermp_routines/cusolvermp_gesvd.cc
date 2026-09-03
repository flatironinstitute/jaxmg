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
// cuSOLVERMp general singular-value decomposition FFI handlers.
//
// This file implements the fused GESVD workflow for rectangular matrices. The
// four exported handlers select singular values alone, U, V^H, or both vector
// matrices without asking XLA to allocate outputs that are not requested. They
// all share the same native implementation and the same XLA-owned NCCL
// communicator.
//
// File workflow:
//   1. Validate the rectangular input and requested output-buffer contracts.
//   2. Allocate one scratch buffer sized for A and all requested vector outputs.
//   3. Convert A to local column-major storage and redistribute it into
//      cuSOLVERMp's 2D block-cyclic layout.
//   4. Create the GESVD descriptor, select thin or full vectors, and run the
//      independently selected jobu/jobvt modes.
//   5. Reverse-redistribute each requested vector matrix in sequence.
//   6. Restore requested vectors to local row-major JAX storage.

#include <algorithm>
#include <array>
#include <cstdlib>
#include <vector>

#include "cusolvermp_common.h"
#include "cusolvermp_routines.h"

namespace xla::gpu {
namespace {

// Owns the matrix descriptors and solver workspace for one distributed GESVD.
//
// The input matrix has already been converted and redistributed into
// column-major 2D block-cyclic storage. U and V^H are optional, independent
// outputs; descriptors and device pointers are supplied only for the requested
// modes. cuSOLVERMp requires A, U, and V^H to occupy distinct buffers.
template <typename DataType>
absl::Status RunCusolverMpGesvd(
    const CusolverMpApi& api, cusolverMpHandle_t handle,
    cusolverMpGrid_t grid, cudaStream_t cuda_stream, int64_t m, int64_t n,
    int64_t tile_size, int32_t process_row, int32_t process_col,
    ffi::AnyBuffer a, ffi::Result<ffi::AnyBuffer> singular_values_out,
    ffi::Result<ffi::AnyBuffer> work_out, ffi::AnyBuffer* u_out,
    ffi::AnyBuffer* vh_out, bool compute_u, bool compute_vh,
    bool full_matrices,
    std::array<int32_t, kGesvdStatusSize>* status_words) {
  const int debug_rank = (*status_words)[2];
  const int32_t process_rows = (*status_words)[4];
  const int32_t process_cols = (*status_words)[5];

  // Derive the logical vector shapes once. Thin mode returns U(m, k) and
  // V^H(k, n), while full mode returns U(m, m) and V^H(n, n), where
  // k = min(m, n).
  const int64_t k = std::min(m, n);
  const int64_t u_rows = m;
  const int64_t u_cols = full_matrices ? m : k;
  const int64_t vh_rows = full_matrices ? n : k;
  const int64_t vh_cols = n;

  // cuSOLVERMp descriptors describe the global matrices, but each rank only
  // stores its NUMROC-owned rows and columns. Check that the padded local
  // buffers provide at least that capacity before creating descriptors.
  const int64_t a_numroc_rows =
      LocalNumroc(m, tile_size, process_row, process_rows);
  const int64_t a_numroc_cols =
      LocalNumroc(n, tile_size, process_col, process_cols);
  (*status_words)[18] = static_cast<int32_t>(a_numroc_rows);
  (*status_words)[19] = static_cast<int32_t>(a_numroc_cols);
  if (a.dimensions()[0] < a_numroc_rows ||
      a.dimensions()[1] < a_numroc_cols) {
    (*status_words)[0] = kOutputShapeMismatch;
    return absl::OkStatus();
  }

  if (compute_u) {
    const int64_t u_numroc_rows =
        LocalNumroc(u_rows, tile_size, process_row, process_rows);
    const int64_t u_numroc_cols =
        LocalNumroc(u_cols, tile_size, process_col, process_cols);
    (*status_words)[27] = static_cast<int32_t>(u_numroc_rows);
    (*status_words)[28] = static_cast<int32_t>(u_numroc_cols);
    if (u_out == nullptr || u_out->dimensions()[0] < u_numroc_rows ||
        u_out->dimensions()[1] < u_numroc_cols) {
      (*status_words)[0] = kOutputShapeMismatch;
      return absl::OkStatus();
    }
  }

  if (compute_vh) {
    const int64_t vh_numroc_rows =
        LocalNumroc(vh_rows, tile_size, process_row, process_rows);
    const int64_t vh_numroc_cols =
        LocalNumroc(vh_cols, tile_size, process_col, process_cols);
    (*status_words)[29] = static_cast<int32_t>(vh_numroc_rows);
    (*status_words)[30] = static_cast<int32_t>(vh_numroc_cols);
    if (vh_out == nullptr || vh_out->dimensions()[0] < vh_numroc_rows ||
        vh_out->dimensions()[1] < vh_numroc_cols) {
      (*status_words)[0] = kOutputShapeMismatch;
      return absl::OkStatus();
    }
  }

  CusolverMpDebug(
      debug_rank,
      "gesvd enter m=%lld n=%lld tile=%lld process_coord=(%d,%d) "
      "compute_u=%d compute_vh=%d full=%d dtype=%d",
      static_cast<long long>(m), static_cast<long long>(n),
      static_cast<long long>(tile_size), process_row, process_col,
      compute_u ? 1 : 0, compute_vh ? 1 : 0, full_matrices ? 1 : 0,
      static_cast<int>(SolverTraits<DataType>::cuda_data_type));

  // These resources belong only to the cuSOLVERMp call. The separate
  // redistribution scratch buffer is owned by the fused dispatch layer and is
  // not included in the solver workspace queried below.
  cusolverMpGesvdDescriptor_t gesvd_desc = nullptr;
  cusolverMpMatrixDescriptor_t desc_a = nullptr;
  cusolverMpMatrixDescriptor_t desc_u = nullptr;
  cusolverMpMatrixDescriptor_t desc_vh = nullptr;
  void* d_work = nullptr;
  void* h_work = nullptr;
  int* d_info = nullptr;

  // All exits after resource creation pass through one cleanup path. If the
  // solver path succeeded, a descriptor-destruction failure becomes the status
  // reported to Python; otherwise the earlier failure remains authoritative.
  auto cleanup = [&]() {
    if (d_work != nullptr) cudaFree(d_work);
    if (d_info != nullptr) cudaFree(d_info);
    if (h_work != nullptr) std::free(h_work);
    auto record_destroy_failure = [&](cusolverStatus_t destroy_status,
                                      int32_t status_code) {
      if (destroy_status != CUSOLVER_STATUS_SUCCESS &&
          (*status_words)[0] == kStatusOk) {
        (*status_words)[0] = status_code;
        (*status_words)[11] = static_cast<int32_t>(destroy_status);
      }
    };
    if (desc_vh != nullptr) {
      record_destroy_failure(api.destroy_matrix_desc(desc_vh),
                             kDestroyMatrixDescFailed);
    }
    if (desc_u != nullptr) {
      record_destroy_failure(api.destroy_matrix_desc(desc_u),
                             kDestroyMatrixDescFailed);
    }
    if (desc_a != nullptr) {
      record_destroy_failure(api.destroy_matrix_desc(desc_a),
                             kDestroyMatrixDescFailed);
    }
    if (gesvd_desc != nullptr) {
      record_destroy_failure(api.gesvd_descriptor_destroy(gesvd_desc),
                             kGesvdDescriptorDestroyFailed);
    }
  };
  auto cuda_error_after_cleanup = [&](cudaError_t cuda_status,
                                      const char* caller) -> absl::Status {
    cleanup();
    return absl::InternalError(absl::StrFormat(
        "%s failed: cuda status %d (%s)", caller,
        static_cast<int>(cuda_status), cudaGetErrorString(cuda_status)));
  };

  // The operation descriptor carries settings that apply to the complete SVD,
  // including whether cuSOLVERMp should form thin or full vector matrices.
  cusolverStatus_t solver_status =
      api.gesvd_descriptor_create(&gesvd_desc);
  if (solver_status != CUSOLVER_STATUS_SUCCESS || gesvd_desc == nullptr) {
    (*status_words)[0] = kGesvdDescriptorCreateFailed;
    (*status_words)[11] = static_cast<int32_t>(solver_status);
    cleanup();
    return absl::OkStatus();
  }
  (*status_words)[24] = 1;

  const cusolverMpGesvdOutputShape_t output_shape =
      full_matrices ? CUSOLVERMP_GESVD_OUTPUT_SHAPE_FULL
                    : CUSOLVERMP_GESVD_OUTPUT_SHAPE_THIN;
  solver_status = api.gesvd_descriptor_set_attribute(
      gesvd_desc, CUSOLVERMP_GESVD_DESCRIPTOR_ATTRIBUTE_SHAPE, &output_shape,
      sizeof(output_shape));
  if (solver_status != CUSOLVER_STATUS_SUCCESS) {
    (*status_words)[0] = kGesvdDescriptorAttributeFailed;
    (*status_words)[11] = static_cast<int32_t>(solver_status);
    cleanup();
    return absl::OkStatus();
  }

  // A is always present and is overwritten by GESVD. U and V^H descriptors are
  // created independently so unrequested vector outputs require no matrix-sized
  // allocation and can be passed to cuSOLVERMp as null descriptors.
  solver_status = api.create_matrix_desc(
      &desc_a, grid, SolverTraits<DataType>::cuda_data_type, m, n, tile_size,
      tile_size, /*RSRC_A=*/0, /*CSRC_A=*/0, a.dimensions()[0]);
  if (solver_status != CUSOLVER_STATUS_SUCCESS || desc_a == nullptr) {
    (*status_words)[0] = kCreateMatrixDescFailed;
    (*status_words)[11] = static_cast<int32_t>(solver_status);
    cleanup();
    return absl::OkStatus();
  }
  (*status_words)[10] = 1;

  if (compute_u) {
    solver_status = api.create_matrix_desc(
        &desc_u, grid, SolverTraits<DataType>::cuda_data_type, u_rows, u_cols,
        tile_size, tile_size, /*RSRC_U=*/0, /*CSRC_U=*/0,
        u_out->dimensions()[0]);
    if (solver_status != CUSOLVER_STATUS_SUCCESS || desc_u == nullptr) {
      (*status_words)[0] = kCreateMatrixDescFailed;
      (*status_words)[11] = static_cast<int32_t>(solver_status);
      cleanup();
      return absl::OkStatus();
    }
    (*status_words)[25] = 1;
  }

  if (compute_vh) {
    solver_status = api.create_matrix_desc(
        &desc_vh, grid, SolverTraits<DataType>::cuda_data_type, vh_rows,
        vh_cols, tile_size, tile_size, /*RSRC_VT=*/0, /*CSRC_VT=*/0,
        vh_out->dimensions()[0]);
    if (solver_status != CUSOLVER_STATUS_SUCCESS || desc_vh == nullptr) {
      (*status_words)[0] = kCreateMatrixDescFailed;
      (*status_words)[11] = static_cast<int32_t>(solver_status);
      cleanup();
      return absl::OkStatus();
    }
    (*status_words)[26] = 1;
  }

  // GESVD overwrites A. The public input is therefore aliased or copied into
  // the A-sized work result before cuSOLVERMp receives the pointer.
  if (absl::Status copy_status =
          CopyAnyBufferToOutputIfNeeded(cuda_stream, a, work_out);
      !copy_status.ok()) {
    cleanup();
    return copy_status;
  }
  cudaError_t cuda_status = cudaStreamSynchronize(cuda_stream);
  if (cuda_status != cudaSuccess) {
    return cuda_error_after_cleanup(cuda_status,
                                    "cusolvermp_gesvd input synchronize");
  }

  const cusolverEigMode_t jobu =
      compute_u ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
  const cusolverEigMode_t jobvt =
      compute_vh ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
  void* u_data = compute_u ? u_out->untyped_data() : nullptr;
  void* vh_data = compute_vh ? vh_out->untyped_data() : nullptr;
  size_t workspace_device = 0;
  size_t workspace_host = 0;

  // Query workspace with exactly the same job flags, descriptors, and pointers
  // used by execution. The returned device and host allocations are private to
  // cuSOLVERMp and are released before this helper returns.
  solver_status = api.gesvd_buffer_size(
      handle, gesvd_desc, jobu, jobvt, m, n, work_out->untyped_data(),
      /*IA=*/1, /*JA=*/1, desc_a, singular_values_out->untyped_data(), u_data,
      /*IU=*/1, /*JU=*/1, desc_u, vh_data, /*IVT=*/1, /*JVT=*/1, desc_vh,
      SolverTraits<DataType>::cuda_data_type, &workspace_device,
      &workspace_host);
  if (solver_status != CUSOLVER_STATUS_SUCCESS) {
    (*status_words)[0] = kGesvdWorkspaceFailed;
    (*status_words)[11] = static_cast<int32_t>(solver_status);
    cleanup();
    return absl::OkStatus();
  }
  (*status_words)[31] = SizeToKiBForStatus(workspace_device);
  (*status_words)[32] = SizeToKiBForStatus(workspace_host);

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
                                    "cusolvermp_gesvd info initialization");
  }
  cuda_status = cudaStreamSynchronize(cuda_stream);
  if (cuda_status != cudaSuccess) {
    return cuda_error_after_cleanup(cuda_status,
                                    "cusolvermp_gesvd info synchronize");
  }

  // cuSOLVERMp writes singular values to the replicated real output and writes
  // each requested vector matrix directly in 2D block-cyclic column-major form.
  // The fused dispatch layer restores those vector matrices for JAX afterward.
  CusolverMpDebug(debug_rank, "gesvd begin");
  solver_status = api.gesvd(
      handle, gesvd_desc, jobu, jobvt, m, n, work_out->untyped_data(),
      /*IA=*/1, /*JA=*/1, desc_a, singular_values_out->untyped_data(), u_data,
      /*IU=*/1, /*JU=*/1, desc_u, vh_data, /*IVT=*/1, /*JVT=*/1, desc_vh,
      SolverTraits<DataType>::cuda_data_type, d_work, workspace_device, h_work,
      workspace_host, d_info);
  CusolverMpDebug(debug_rank, "gesvd end status=%d",
                  static_cast<int>(solver_status));
  if (solver_status != CUSOLVER_STATUS_SUCCESS) {
    (*status_words)[0] = kGesvdFailed;
    (*status_words)[11] = static_cast<int32_t>(solver_status);
    cleanup();
    return absl::OkStatus();
  }
  (*status_words)[33] = 1;
  cuda_status = cudaStreamSynchronize(cuda_stream);
  if (cuda_status != cudaSuccess) {
    return cuda_error_after_cleanup(cuda_status,
                                    "cusolvermp_gesvd solver synchronize");
  }

  int h_info = -1;
  cuda_status = cudaMemcpyAsync(&h_info, d_info, sizeof(int),
                                cudaMemcpyDeviceToHost, cuda_stream);
  if (cuda_status != cudaSuccess) {
    return cuda_error_after_cleanup(cuda_status,
                                    "cusolvermp_gesvd info copy");
  }
  cuda_status = cudaStreamSynchronize(cuda_stream);
  if (cuda_status != cudaSuccess) {
    return cuda_error_after_cleanup(cuda_status,
                                    "cusolvermp_gesvd info copy synchronize");
  }
  (*status_words)[34] = h_info;
  if (h_info != 0) {
    (*status_words)[0] = kGesvdInfoNonzero;
  }

  // Record the algorithm metadata exposed by the GESVD descriptor. This is
  // diagnostic information only; the singular-value output shape remains k.
  int64_t singular_values_found = -1;
  size_t attribute_bytes = 0;
  solver_status = api.gesvd_descriptor_get_attribute(
      gesvd_desc, CUSOLVERMP_GESVD_DESCRIPTOR_ATTRIBUTE_NUM_SINGULAR_FOUND,
      &singular_values_found, sizeof(singular_values_found), &attribute_bytes);
  if (solver_status != CUSOLVER_STATUS_SUCCESS) {
    if ((*status_words)[0] == kStatusOk) {
      (*status_words)[0] = kGesvdDescriptorQueryFailed;
      (*status_words)[11] = static_cast<int32_t>(solver_status);
    }
  } else {
    (*status_words)[35] = static_cast<int32_t>(singular_values_found);
  }

  cleanup();
  return absl::OkStatus();
}

// Binds GESVD to the CUDA device and XLA-owned communicator for this rank.
// This layer owns the common cuSOLVERMp handle/grid lifecycle and dispatches
// the matrix scalar type to the templated solver implementation above.
absl::Status RunCusolverMpGesvdSolver(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    int64_t process_rows, int64_t process_cols, int64_t m, int64_t n,
    int64_t tile_size, int64_t grid_mapping,
    absl::Span<const int64_t> rank_map, ffi::AnyBuffer a,
    ffi::Result<ffi::AnyBuffer> singular_values,
    ffi::Result<ffi::AnyBuffer> work, ffi::AnyBuffer* u, ffi::AnyBuffer* vh,
    bool compute_u, bool compute_vh, bool full_matrices,
    ffi::Result<ffi::BufferR1<S32>> status_out,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  (void)comm_stream;
  const int64_t k = std::min(m, n);
  // Stage 1: validate the local FFI buffers for the selected output mode. U
  // and V^H are independent optional outputs, while every mode returns the
  // singular values and an A-sized work buffer overwritten by cuSOLVERMp.
  if (stream == nullptr || cuda_stream == nullptr) {
    return absl::InvalidArgumentError(
        "cusolvermp_gesvd requires XLA and CUDA streams");
  }
  if (a.dimensions().size() != 2 || work->dimensions().size() != 2 ||
      singular_values->dimensions().size() != 1 ||
      (compute_u && (u == nullptr || u->dimensions().size() != 2)) ||
      (compute_vh && (vh == nullptr || vh->dimensions().size() != 2))) {
    return absl::InvalidArgumentError(
        "cusolvermp_gesvd expects rank-2 matrix buffers and a rank-1 "
        "singular-value output");
  }
  if (a.element_type() != work->element_type() ||
      (compute_u && a.element_type() != u->element_type()) ||
      (compute_vh && a.element_type() != vh->element_type())) {
    return absl::InvalidArgumentError(
        "cusolvermp_gesvd requires matching matrix dtypes");
  }
  const PrimitiveType expected_singular_type =
      (a.element_type() == F32 || a.element_type() == C64) ? F32 : F64;
  if (singular_values->element_type() != expected_singular_type ||
      singular_values->dimensions()[0] != k) {
    return absl::InvalidArgumentError(
        "cusolvermp_gesvd received an invalid singular-value output");
  }
  if (a.dimensions()[0] != work->dimensions()[0] ||
      a.dimensions()[1] != work->dimensions()[1]) {
    return absl::InvalidArgumentError(
        "cusolvermp_gesvd input and work shapes must match");
  }
  if (status_out->dimensions().size() != 1 ||
      status_out->dimensions()[0] != kGesvdStatusSize) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "cusolvermp_gesvd expects status shape (%d,)", kGesvdStatusSize));
  }

  std::array<int32_t, kGesvdStatusSize> status_words = {
      kStatusOk,  // JAXMg status code.
      -1,  // CUDA device selected for this FFI invocation.
      -1,  // NCCL rank reported by the borrowed communicator.
      -1,  // NCCL communicator size.
      static_cast<int32_t>(process_rows),  // Process-grid rows.
      static_cast<int32_t>(process_cols),  // Process-grid columns.
      -1,  // cuSOLVERMp version, if available.
      0,   // libcusolverMp linked runtime available.
      0,   // cusolverMpHandle_t created.
      0,   // cusolverMpGrid_t created.
      0,   // matrix descriptor for A created.
      0,   // raw cuSOLVER status from the failing call, if any.
      static_cast<int32_t>(a.size_bytes()),       // Local A size, bytes.
      static_cast<int32_t>(m),                    // Global A rows.
      static_cast<int32_t>(n),                    // Global A columns.
      static_cast<int32_t>(tile_size),            // Solver tile size.
      static_cast<int32_t>(a.dimensions()[0]),    // Local A row capacity.
      static_cast<int32_t>(a.dimensions()[1]),    // Local A column capacity.
      -1,  // local NUMROC rows for A.
      -1,  // local NUMROC cols for A.
      static_cast<int32_t>(singular_values->size_bytes()),  // S size, bytes.
      static_cast<int32_t>(compute_u),      // Left vectors requested.
      static_cast<int32_t>(compute_vh),     // Right vectors requested.
      static_cast<int32_t>(full_matrices),  // Full rather than thin SVD.
      0,   // cusolverMpGesvdDescriptor_t created.
      0,   // matrix descriptor for U created.
      0,   // matrix descriptor for V^H created.
      -1,  // local NUMROC rows for U.
      -1,  // local NUMROC cols for U.
      -1,  // local NUMROC rows for V^H.
      -1,  // local NUMROC cols for V^H.
      -1,  // GESVD device workspace, KiB.
      -1,  // GESVD host workspace, KiB.
      0,   // cusolverMpGesvd called.
      -1,  // GESVD info value copied from device.
      -1,  // number of singular values reported by cuSOLVERMp.
      -1,  // dtype code.
      static_cast<int32_t>(grid_mapping),  // cuSOLVERMp grid mapping.
      0,  // reserved.
      0,  // reserved.
      0,  // reserved.
      0,  // reserved.
  };

  // Stage 2: bind cuSOLVERMp to the CUDA device that owns this rank's local A
  // shard. This avoids relying on ambient host-thread CUDA device state.
  absl::StatusOr<int> buffer_device = DeviceForCudaPointer(a.untyped_data());
  if (!buffer_device.ok() || cudaSetDevice(*buffer_device) != cudaSuccess) {
    status_words[0] = kCudaDeviceFailed;
    return CopyGesvdStatusToDevice(stream, status_words, status_out);
  }
  const int cuda_device = *buffer_device;
  status_words[1] = cuda_device;

  // Stage 3: borrow the communicator that XLA created for the compiled
  // program. The same NCCL communicator is used by native redistribution and
  // passed to cuSOLVERMp; GESVD does not create a second communicator.
  if (collective_params == nullptr || collective_cliques == nullptr) {
    status_words[0] = kCollectiveContextMissing;
    return CopyGesvdStatusToDevice(stream, status_words, status_out);
  }
  absl::StatusOr<GpuCliqueKey> clique_key =
      AllAssignedDevicesP2PCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    status_words[0] = kCliqueKeyFailed;
    return CopyGesvdStatusToDevice(stream, status_words, status_out);
  }
  absl::StatusOr<GpuCommunicator*> gpu_comm = collective_cliques->GetComm(
      *clique_key, collective_params->global_device_id);
  if (!gpu_comm.ok() || *gpu_comm == nullptr) {
    status_words[0] = kCommunicatorMissing;
    return CopyGesvdStatusToDevice(stream, status_words, status_out);
  }
  void* platform_handle = (*gpu_comm)->platform_comm().handle;
  if (platform_handle == nullptr) {
    status_words[0] = kNcclHandleMissing;
    return CopyGesvdStatusToDevice(stream, status_words, status_out);
  }
  ncclComm_t nccl_comm = reinterpret_cast<ncclComm_t>(platform_handle);
  int nccl_rank = -1;
  int nccl_count = -1;
  if (ncclCommUserRank(nccl_comm, &nccl_rank) != ncclSuccess ||
      ncclCommCount(nccl_comm, &nccl_count) != ncclSuccess) {
    status_words[0] = kNcclRankMismatch;
    return CopyGesvdStatusToDevice(stream, status_words, status_out);
  }
  status_words[2] = nccl_rank;
  status_words[3] = nccl_count;

  // Stage 4: validate the process-grid shape and rank mapping before creating
  // cuSOLVERMp resources. Both row-major and column-major dense mesh mappings
  // are accepted, but arbitrary rank permutations are rejected.
  if (process_rows <= 0 || process_cols <= 0 ||
      process_rows * process_cols != nccl_count || m <= 0 || n <= 0 ||
      tile_size <= 0) {
    status_words[0] = kGridShapeMismatch;
    return CopyGesvdStatusToDevice(stream, status_words, status_out);
  }
  JAXMG_RETURN_IF_ERROR(
      ValidateCusolverMpGridMapping("cusolvermp_gesvd", grid_mapping));
  if (!rank_map.empty()) {
    JAXMG_RETURN_IF_ERROR(ValidateStandardRankMapForGridMapping(
        "cusolvermp_gesvd", rank_map, process_rows, process_cols,
        grid_mapping));
  }

  // Stage 5: create the cuSOLVERMp handle and process grid against the
  // borrowed NCCL communicator. The grid mapping preserves the corresponding
  // row-major or column-major JAX device ordering.
  CusolverMpApi api = LinkedCusolverMpApi(&status_words);
  cusolverMpHandle_t handle = nullptr;
  cusolverStatus_t solver_status =
      api.create(&handle, cuda_device, cuda_stream);
  if (solver_status != CUSOLVER_STATUS_SUCCESS || handle == nullptr) {
    status_words[0] = kCreateHandleFailed;
    status_words[11] = static_cast<int32_t>(solver_status);
    return CopyGesvdStatusToDevice(stream, status_words, status_out);
  }
  status_words[8] = 1;

  int version = -1;
  solver_status = api.get_version(handle, &version);
  if (solver_status != CUSOLVER_STATUS_SUCCESS) {
    status_words[0] = kGetVersionFailed;
    status_words[11] = static_cast<int32_t>(solver_status);
    api.destroy(handle);
    return CopyGesvdStatusToDevice(stream, status_words, status_out);
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
    return CopyGesvdStatusToDevice(stream, status_words, status_out);
  }
  status_words[9] = 1;

  const auto [process_row, process_col] = ProcessCoordFromRank(
      nccl_rank, process_rows, process_cols, grid_mapping);

  // Stage 6: dispatch on the XLA primitive dtype and run the shared GESVD
  // helper. Its job flags and descriptors reflect the statically selected
  // combination of U, V^H, and thin/full outputs.
  absl::Status gesvd_status;
  switch (a.element_type()) {
    case F32:
      status_words[36] = 1;
      gesvd_status = RunCusolverMpGesvd<float>(
          api, handle, grid, cuda_stream, m, n, tile_size, process_row,
          process_col, a, singular_values, work, u, vh, compute_u, compute_vh,
          full_matrices, &status_words);
      break;
    case F64:
      status_words[36] = 2;
      gesvd_status = RunCusolverMpGesvd<double>(
          api, handle, grid, cuda_stream, m, n, tile_size, process_row,
          process_col, a, singular_values, work, u, vh, compute_u, compute_vh,
          full_matrices, &status_words);
      break;
    case C64:
      status_words[36] = 3;
      gesvd_status = RunCusolverMpGesvd<cuFloatComplex>(
          api, handle, grid, cuda_stream, m, n, tile_size, process_row,
          process_col, a, singular_values, work, u, vh, compute_u, compute_vh,
          full_matrices, &status_words);
      break;
    case C128:
      status_words[36] = 4;
      gesvd_status = RunCusolverMpGesvd<cuDoubleComplex>(
          api, handle, grid, cuda_stream, m, n, tile_size, process_row,
          process_col, a, singular_values, work, u, vh, compute_u, compute_vh,
          full_matrices, &status_words);
      break;
    default:
      status_words[0] = kUnsupportedDtype;
      break;
  }
  if (!gesvd_status.ok()) {
    api.destroy_grid(grid);
    api.destroy(handle);
    return gesvd_status;
  }

  // Stage 7: release the cuSOLVERMp grid and handle in reverse construction
  // order, then return the per-rank status vector to Python.
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
  return CopyGesvdStatusToDevice(stream, status_words, status_out);
}

// Executes the complete fused workflow for one statically selected output
// mode. A single scratch allocation is shared by the rectangular A, U, and V^H
// redistributions, and requested vector outputs are restored sequentially.
absl::Status RunCusolverMpGesvdDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    int64_t process_rows, int64_t process_cols, int64_t m, int64_t n,
    int64_t tile_size, int64_t grid_mapping, int64_t full_matrices,
    absl::Span<const int64_t> rank_map, ffi::AnyBuffer a,
    ffi::Result<ffi::AnyBuffer> singular_values,
    ffi::Result<ffi::AnyBuffer> work, ffi::AnyBuffer* u, ffi::AnyBuffer* vh,
    bool compute_u, bool compute_vh,
    ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  // Stage 1: validate the matrix buffers needed by the selected output mode.
  // Python has already padded each logical matrix to the local capacity needed
  // by the process grid and tile size.
  if (a.dimensions().size() != 2 || work->dimensions().size() != 2 ||
      (compute_u && (u == nullptr || u->dimensions().size() != 2)) ||
      (compute_vh && (vh == nullptr || vh->dimensions().size() != 2))) {
    return absl::InvalidArgumentError(
        "cusolvermp_gesvd expects rank-2 matrix buffers");
  }
  if (a.element_type() != work->element_type() ||
      (compute_u && a.element_type() != u->element_type()) ||
      (compute_vh && a.element_type() != vh->element_type())) {
    return absl::InvalidArgumentError(
        "cusolvermp_gesvd requires matching matrix dtypes");
  }
  if (a.dimensions()[0] != work->dimensions()[0] ||
      a.dimensions()[1] != work->dimensions()[1]) {
    return absl::InvalidArgumentError(
        "cusolvermp_gesvd input and work shapes must match");
  }

  const int64_t k = std::min(m, n);
  const int64_t u_rows = m;
  const int64_t u_cols = full_matrices != 0 ? m : k;
  const int64_t vh_rows = full_matrices != 0 ? n : k;
  const int64_t vh_cols = n;
  const size_t element_bytes =
      a.size_bytes() / static_cast<size_t>(a.element_count());

  // Stage 2: allocate one reusable redistribution scratch buffer. Its size is
  // the maximum required by A and the requested U/V^H outputs, since those
  // matrices are converted and redistributed sequentially.
  std::vector<Padded2DRedistScratchRequest> scratch_requests;
  scratch_requests.reserve(1 + static_cast<int>(compute_u) +
                           static_cast<int>(compute_vh));
  scratch_requests.push_back(Padded2DRedistScratchRequest{
      process_rows, process_cols, tile_size, tile_size, m, n,
      a.dimensions()[0], a.dimensions()[1], rank_map});
  if (compute_u) {
    scratch_requests.push_back(Padded2DRedistScratchRequest{
        process_rows, process_cols, tile_size, tile_size, u_rows, u_cols,
        u->dimensions()[0], u->dimensions()[1], rank_map});
  }
  if (compute_vh) {
    scratch_requests.push_back(Padded2DRedistScratchRequest{
        process_rows, process_cols, tile_size, tile_size, vh_rows, vh_cols,
        vh->dimensions()[0], vh->dimensions()[1], rank_map});
  }

  absl::StatusOr<Padded2DRedistScratch> scratch_status =
      AllocatePadded2DRedistScratch(cuda_stream, element_bytes,
                                    absl::MakeConstSpan(scratch_requests),
                                    "cusolvermp_gesvd_redistribution");
  if (!scratch_status.ok()) {
    return scratch_status.status();
  }
  Padded2DRedistScratch scratch = *scratch_status;
  bool scratch_freed = false;
  auto return_after_cleanup = [&](absl::Status result) -> absl::Status {
    if (!scratch_freed) {
      absl::Status free_status = FreePadded2DRedistScratch(
          cuda_stream, scratch, "cusolvermp_gesvd_redistribution");
      scratch_freed = free_status.ok();
      if (result.ok() && !free_status.ok()) return free_status;
    }
    return result;
  };
  auto synchronize = [&](const char* caller) -> absl::Status {
    const cudaError_t cuda_status = cudaStreamSynchronize(cuda_stream);
    if (cuda_status == cudaSuccess) return absl::OkStatus();
    return return_after_cleanup(absl::InternalError(absl::StrFormat(
        "%s failed: cuda status %d (%s)", caller,
        static_cast<int>(cuda_status), cudaGetErrorString(cuda_status))));
  };

  // Stage 3: alias or copy A into the work output, then convert each local
  // shard from JAX's row-major storage to the column-major storage expected by
  // cuSOLVERMp. Donation normally makes the copy a no-op.
  if (absl::Status copy_status = CopyMatrixIfNeeded(cuda_stream, a, work);
      !copy_status.ok()) {
    return return_after_cleanup(copy_status);
  }
  ffi::AnyBuffer a_work = *work;
  if (absl::Status convert_status = ConvertRowMajorToColumnMajorInPlace(
          cuda_stream, "cusolvermp_gesvd/a_layout_convert", a_work,
          scratch.base, scratch.elements);
      !convert_status.ok()) {
    return return_after_cleanup(convert_status);
  }

  // Stage 4: align edge padding and redistribute A into cuSOLVERMp's 2D
  // block-cyclic layout. The stream is synchronized before the solver consumes
  // the redistributed matrix.
  if (absl::Status redist_status = ExecutePadded2DNativePlanRaw(
          "cusolvermp_gesvd/a_forward", stream, comm_stream, cuda_stream,
          process_rows, process_cols, tile_size, tile_size, m, n,
          /*reverse=*/0, rank_map, a_work, work->device_memory(), scratch.base,
          scratch.elements, collective_params, collective_cliques);
      !redist_status.ok()) {
    return return_after_cleanup(redist_status);
  }
  if (absl::Status sync_status =
          synchronize("cusolvermp_gesvd forward stream synchronize");
      !sync_status.ok()) {
    return sync_status;
  }

  // Stage 5: run GESVD. Singular values are replicated, while requested vector
  // outputs remain in 2D block-cyclic column-major storage until restored
  // below. Values-only mode can release scratch and return immediately.
  ffi::AnyBuffer a_cyclic = *work;
  if (absl::Status solver_status = RunCusolverMpGesvdSolver(
          stream, comm_stream, cuda_stream, process_rows, process_cols, m, n,
          tile_size, grid_mapping, rank_map, a_cyclic, singular_values, work,
          u, vh, compute_u, compute_vh, full_matrices != 0, status,
          collective_params, collective_cliques);
      !solver_status.ok()) {
    return return_after_cleanup(solver_status);
  }
  if (!compute_u && !compute_vh) {
    return return_after_cleanup(absl::OkStatus());
  }
  if (absl::Status sync_status =
          synchronize("cusolvermp_gesvd solver stream synchronize");
      !sync_status.ok()) {
    return sync_status;
  }

  // Stage 6: reverse-redistribute U, when requested, and restore its local JAX
  // row-major memory layout. Thin and full modes use their respective logical
  // column counts while sharing the same rectangular redistribution path.
  if (compute_u) {
    if (absl::Status redist_status = ExecutePadded2DNativePlanRaw(
            "cusolvermp_gesvd/u_reverse", stream, comm_stream, cuda_stream,
            process_rows, process_cols, tile_size, tile_size, u_rows, u_cols,
            /*reverse=*/1, rank_map, *u, u->device_memory(), scratch.base,
            scratch.elements, collective_params, collective_cliques);
        !redist_status.ok()) {
      return return_after_cleanup(redist_status);
    }
    if (absl::Status convert_status = ConvertColumnMajorToRowMajorInPlace(
            cuda_stream, "cusolvermp_gesvd/u_layout_restore", *u,
            scratch.base, scratch.elements);
        !convert_status.ok()) {
      return return_after_cleanup(convert_status);
    }
  }

  // Stage 7: restore V^H in the same way. If U was also requested, synchronize
  // its restoration first because both outputs reuse the same scratch buffer.
  if (compute_vh) {
    if (compute_u) {
      if (absl::Status sync_status =
              synchronize("cusolvermp_gesvd U restore synchronize");
          !sync_status.ok()) {
        return sync_status;
      }
    }
    if (absl::Status redist_status = ExecutePadded2DNativePlanRaw(
            "cusolvermp_gesvd/vh_reverse", stream, comm_stream, cuda_stream,
            process_rows, process_cols, tile_size, tile_size, vh_rows, vh_cols,
            /*reverse=*/1, rank_map, *vh, vh->device_memory(), scratch.base,
            scratch.elements, collective_params, collective_cliques);
        !redist_status.ok()) {
      return return_after_cleanup(redist_status);
    }
    if (absl::Status convert_status = ConvertColumnMajorToRowMajorInPlace(
            cuda_stream, "cusolvermp_gesvd/vh_layout_restore", *vh,
            scratch.base, scratch.elements);
        !convert_status.ok()) {
      return return_after_cleanup(convert_status);
    }
  }
  return return_after_cleanup(absl::OkStatus());
}

}  // namespace

// Requests the one XLA P2P clique used by all GESVD output modes.
absl::Status XlaCusolverMpGesvdPrepare(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests) {
  return RequestAllAssignedP2PCommunicator(
      collective_params, clique_requests, "cusolvermp_gesvd");
}

// Executes GESVD with both left and right singular vectors.
absl::Status XlaCusolverMpGesvdUvDispatch(
    se::Stream* stream, cudaStream_t cuda_stream,
    int64_t process_rows, int64_t process_cols, int64_t m, int64_t n,
    int64_t tile_size, int64_t grid_mapping, int64_t full_matrices,
    absl::Span<const int64_t> rank_map, ffi::AnyBuffer a,
    ffi::Result<ffi::AnyBuffer> singular_values,
    ffi::Result<ffi::AnyBuffer> work, ffi::Result<ffi::AnyBuffer> u,
    ffi::Result<ffi::AnyBuffer> vh,
    ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  ffi::AnyBuffer u_buffer = *u;
  ffi::AnyBuffer vh_buffer = *vh;
  return RunCusolverMpGesvdDispatch(
      stream, /*comm_stream=*/nullptr, cuda_stream, process_rows, process_cols,
      m, n,
      tile_size, grid_mapping, full_matrices, rank_map, a, singular_values,
      work, &u_buffer, &vh_buffer, /*compute_u=*/true, /*compute_vh=*/true,
      status, collective_params, collective_cliques);
}

// Executes GESVD with left singular vectors only.
absl::Status XlaCusolverMpGesvdUDispatch(
    se::Stream* stream, cudaStream_t cuda_stream,
    int64_t process_rows, int64_t process_cols, int64_t m, int64_t n,
    int64_t tile_size, int64_t grid_mapping, int64_t full_matrices,
    absl::Span<const int64_t> rank_map, ffi::AnyBuffer a,
    ffi::Result<ffi::AnyBuffer> singular_values,
    ffi::Result<ffi::AnyBuffer> work, ffi::Result<ffi::AnyBuffer> u,
    ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  ffi::AnyBuffer u_buffer = *u;
  return RunCusolverMpGesvdDispatch(
      stream, /*comm_stream=*/nullptr, cuda_stream, process_rows, process_cols,
      m, n,
      tile_size, grid_mapping, full_matrices, rank_map, a, singular_values,
      work, &u_buffer, /*vh=*/nullptr, /*compute_u=*/true,
      /*compute_vh=*/false, status, collective_params, collective_cliques);
}

// Executes GESVD with right singular vectors only.
absl::Status XlaCusolverMpGesvdVhDispatch(
    se::Stream* stream, cudaStream_t cuda_stream,
    int64_t process_rows, int64_t process_cols, int64_t m, int64_t n,
    int64_t tile_size, int64_t grid_mapping, int64_t full_matrices,
    absl::Span<const int64_t> rank_map, ffi::AnyBuffer a,
    ffi::Result<ffi::AnyBuffer> singular_values,
    ffi::Result<ffi::AnyBuffer> work, ffi::Result<ffi::AnyBuffer> vh,
    ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  ffi::AnyBuffer vh_buffer = *vh;
  return RunCusolverMpGesvdDispatch(
      stream, /*comm_stream=*/nullptr, cuda_stream, process_rows, process_cols,
      m, n,
      tile_size, grid_mapping, full_matrices, rank_map, a, singular_values,
      work, /*u=*/nullptr, &vh_buffer, /*compute_u=*/false,
      /*compute_vh=*/true, status, collective_params, collective_cliques);
}

// Executes values-only GESVD without allocating singular-vector outputs.
absl::Status XlaCusolverMpGesvdValuesDispatch(
    se::Stream* stream, cudaStream_t cuda_stream,
    int64_t process_rows, int64_t process_cols, int64_t m, int64_t n,
    int64_t tile_size, int64_t grid_mapping, int64_t full_matrices,
    absl::Span<const int64_t> rank_map, ffi::AnyBuffer a,
    ffi::Result<ffi::AnyBuffer> singular_values,
    ffi::Result<ffi::AnyBuffer> work,
    ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  return RunCusolverMpGesvdDispatch(
      stream, /*comm_stream=*/nullptr, cuda_stream, process_rows, process_cols,
      m, n,
      tile_size, grid_mapping, full_matrices, rank_map, a, singular_values,
      work, /*u=*/nullptr, /*vh=*/nullptr, /*compute_u=*/false,
      /*compute_vh=*/false, status, collective_params, collective_cliques);
}

}  // namespace xla::gpu
