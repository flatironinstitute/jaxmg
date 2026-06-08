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
// Fused potri handler for the single-node XLA communicator backend.
//
// The matrix is reshuffled into cuSolverMg's 1D block-cyclic layout before
// factorization/inversion and then reshuffled back to the original JAX
// row-sharded layout before returning.

#include "include/xla_comm_backend.h"

namespace xla::gpu {

template <typename DataType>
absl::Status XlaCommPotriMgNativePlanImpl(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    se::ScratchAllocator& scratch, int64_t tile_size, ffi::AnyBuffer a,
    ffi::Result<ffi::AnyBuffer> out,
    ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  const int64_t timing_start = PotrsPhaseTimer::NowNanos();
  if (stream == nullptr || comm_stream == nullptr || cuda_stream == nullptr) {
    return absl::InvalidArgumentError(
        "xla_comm_potri_mg_native_plan requires stream contexts");
  }
  if (collective_params == nullptr || collective_cliques == nullptr) {
    return absl::InvalidArgumentError(
        "xla_comm_potri_mg_native_plan requires XLA collective contexts");
  }
  if (tile_size <= 0) {
    return absl::InvalidArgumentError(
        "xla_comm_potri_mg_native_plan requires positive tile_size");
  }
  if (a.dimensions().size() != 2 || out->dimensions().size() != 2 ||
      status->dimensions().size() != 1 || status->dimensions()[0] != 1) {
    return absl::InvalidArgumentError(
        "xla_comm_potri_mg_native_plan expects A/out rank-2 and status shape "
        "(1,)");
  }
  if (a.dimensions()[0] != out->dimensions()[0] ||
      a.dimensions()[1] != out->dimensions()[1]) {
    return absl::InvalidArgumentError(
        "xla_comm_potri_mg_native_plan input/output shapes must match");
  }
  if (a.element_type() != out->element_type()) {
    return absl::InvalidArgumentError(
        "xla_comm_potri_mg_native_plan requires matching A/out dtypes");
  }

  const int64_t local_slots = a.dimensions()[0];
  const int64_t n = a.dimensions()[1];
  // The Python wrapper supplies a row-sharded matrix shard of shape
  // (local_slots, N). local_slots may include tile padding that is not part of
  // the global mathematical matrix.
  if (n <= 0 || local_slots <= 0) {
    return absl::InvalidArgumentError(
        "xla_comm_potri_mg_native_plan received invalid matrix dimensions");
  }

  absl::StatusOr<GpuCliqueKey> clique_key =
      NodeScopedP2PCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    return clique_key.status();
  }
  const int64_t num_ranks = static_cast<int64_t>(clique_key->num_devices());
  if (num_ranks <= 0 || num_ranks > 16 || n % num_ranks != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "xla_comm_potri_mg_native_plan requires 1..16 ranks and matrix width "
        "%d divisible by ranks %d",
        n, num_ranks));
  }
  if (local_slots < n / num_ranks) {
    return absl::InvalidArgumentError(
        "xla_comm_potri_mg_native_plan local A rows are smaller than the "
        "unpadded shard size");
  }
  std::optional<RankId> rank =
      clique_key->rank(collective_params->global_device_id);
  if (!rank.has_value()) {
    return absl::InvalidArgumentError(
        "xla_comm_potri_mg_native_plan could not resolve this device rank");
  }
  absl::StatusOr<Communicator*> comm = collective_cliques->GetComm(
      *clique_key, collective_params->global_device_id);
  if (!comm.ok()) {
    return comm.status();
  }

  // The XLA clique rank must agree with CUDA's current device because the
  // process-local arrays below are indexed by CUDA device id and passed to
  // cuSolverMg as device pointer lists.
  int current_device = 0;
  JAXMG_RETURN_IF_CUDA_ERROR(cudaGetDevice(&current_device));
  if (current_device < 0 || current_device >= num_ranks) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "CUDA current device %d is outside the XLA communicator rank count %d",
        current_device, num_ranks));
  }
  if (rank->value() != current_device) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "XLA communicator rank %d does not match CUDA device %d", rank->value(),
        current_device));
  }

  PotrsPhaseTimer timer("xla_comm_potri_mg_native_plan", current_device,
                        static_cast<int>(num_ranks), n, 0, tile_size,
                        timing_start);
  timer.Mark("xla_comm_lookup_and_device_setup");

  const size_t element_bytes = sizeof(DataType);
  const uint64_t column_bytes = static_cast<uint64_t>(n) * element_bytes;
  absl::StatusOr<void*> plan_scratch =
      AllocateFfiScratch(scratch, column_bytes, "native_plan_column_scratch");
  if (!plan_scratch.ok()) {
    return plan_scratch.status();
  }
  timer.Mark("plan_scratch_alloc");

  se::DeviceAddressBase a_base = a.device_memory();
  se::DeviceAddressBase out_base = out->device_memory();
  se::DeviceAddressBase plan_scratch_base(*plan_scratch, column_bytes);
  // Convert the donated row-sharded matrix into cuSolverMg's 1D block-cyclic
  // layout before publishing pointers to the rank-0 cuSolverMg host call.
  absl::Status plan_status = ExecuteMatrixColumnNativePlanRaw(
      "xla_comm_potri_mg_native_plan_forward", stream, comm_stream, *comm, a,
      a_base, out_base, plan_scratch_base, plan_scratch_base, local_slots, n,
      column_bytes, 1, num_ranks, static_cast<int64_t>(rank->value()),
      tile_size, /*reverse=*/false);
  if (!plan_status.ok()) {
    return plan_status;
  }
  timer.Mark("cyclic_reshape");
  // cuSolverMg does not expose a per-call stream setter in this path, so keep a
  // conservative synchronization before invoking the host-side cuSolverMg API.
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));
  timer.Mark("cyclic_reshape_stream_sync");

  // Each FFI invocation runs on one local device. The rank-0 invocation owns the
  // cuSolverMg handle and consumes the per-device pointers published here.
  fused_potri_state.a[current_device] = out->untyped_data();
  fused_potri_barrier.ArriveAndWait(static_cast<int>(num_ranks));
  timer.Mark("pointer_share_barrier");

  const int ia = 1;
  const int ja = 1;
  const int t_a = static_cast<int>(std::min(tile_size, local_slots));
  const cudaDataType compute_type = SolverTraits<DataType>::cuda_data_type;
  std::array<int, 16> device_list{};
  std::array<void*, 16> a_ptrs{};
  std::array<void*, 16> work_ptrs{};

  cusolverMgHandle_t cusolver = nullptr;
  cudaLibMgGrid_t grid_a = nullptr;
  cudaLibMgMatrixDesc_t descr_a = nullptr;
  int info = 0;
  cusolverStatus_t solver_status = CUSOLVER_STATUS_SUCCESS;

  if (current_device == 0) {
    // cuSolverMg is driven from one host invocation. Rank 0 creates the handle,
    // builds descriptor objects over all local device pointers, and publishes
    // the workspace size required on every participating device.
    for (int dev = 0; dev < num_ranks; ++dev) {
      device_list[dev] = dev;
      a_ptrs[dev] = fused_potri_state.a[dev];
    }

    JAXMG_RETURN_IF_CUSOLVER_ERROR(cusolverMgCreate(&cusolver));
    JAXMG_RETURN_IF_CUSOLVER_ERROR(
        cusolverMgDeviceSelect(cusolver, static_cast<int>(num_ranks),
                               device_list.data()));
    JAXMG_RETURN_IF_CUSOLVER_ERROR(cusolverMgCreateDeviceGrid(
        &grid_a, 1, static_cast<int>(num_ranks), device_list.data(),
        CUDALIBMG_GRID_MAPPING_COL_MAJOR));
    JAXMG_RETURN_IF_CUSOLVER_ERROR(cusolverMgCreateMatrixDesc(
        &descr_a, n, n, n, t_a, compute_type, grid_a));

    int64_t lwork_potrf = 0;
    int64_t lwork_potri = 0;
    JAXMG_RETURN_IF_CUSOLVER_ERROR(cusolverMgPotrf_bufferSize(
        cusolver, CUBLAS_FILL_MODE_LOWER, n, a_ptrs.data(), ia, ja, descr_a,
        compute_type, &lwork_potrf));
    JAXMG_RETURN_IF_CUSOLVER_ERROR(cusolverMgPotri_bufferSize(
        cusolver, CUBLAS_FILL_MODE_LOWER, n, a_ptrs.data(), ia, ja, descr_a,
        compute_type, &lwork_potri));
    const int64_t lwork = std::max(lwork_potrf, lwork_potri);
    for (int dev = 0; dev < num_ranks; ++dev) {
      fused_potri_state.lwork[dev] = lwork;
    }
    timer.Mark("cusolver_setup_and_workspace_query");
  }

  fused_potri_barrier.ArriveAndWait(static_cast<int>(num_ranks));
  timer.Mark("lwork_barrier");
  // Scratch allocation is per FFI invocation, hence per local device. The
  // pointer is then shared back to rank 0 before the cuSolverMg call.
  absl::StatusOr<void*> solver_work = AllocateFfiScratch(
      scratch, sizeof(DataType) * fused_potri_state.lwork[current_device],
      "cusolvermg_potri_workspace");
  if (!solver_work.ok()) {
    return solver_work.status();
  }
  fused_potri_state.work[current_device] = *solver_work;
  timer.Mark("workspace_alloc");
  fused_potri_barrier.ArriveAndWait(static_cast<int>(num_ranks));
  timer.Mark("workspace_barrier");

  if (current_device == 0) {
    // This preserves the old JAXMg potri sequence: factor A in-place with
    // potrf, then invert the factorized matrix with potri.
    for (int dev = 0; dev < num_ranks; ++dev) {
      work_ptrs[dev] = fused_potri_state.work[dev];
    }
    solver_status = cusolverMgPotrf(
        cusolver, CUBLAS_FILL_MODE_LOWER, n, a_ptrs.data(), ia, ja, descr_a,
        compute_type, work_ptrs.data(), fused_potri_state.lwork[0], &info);
    JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));
    timer.Mark("potrf");
    for (int dev = 0; dev < num_ranks; ++dev) {
      fused_potri_state.solver_status[dev] =
          static_cast<int32_t>(solver_status);
    }
    if (info < 0) {
      return absl::InternalError(absl::StrFormat(
          "unexpected error in cusolverMgPotrf, %d-th input parameter is wrong",
          -info));
    }

    if (solver_status == CUSOLVER_STATUS_SUCCESS) {
      solver_status = cusolverMgPotri(
          cusolver, CUBLAS_FILL_MODE_LOWER, n, a_ptrs.data(), ia, ja, descr_a,
          compute_type, work_ptrs.data(), fused_potri_state.lwork[0], &info);
      JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));
      timer.Mark("potri");
      for (int dev = 0; dev < num_ranks; ++dev) {
        fused_potri_state.solver_status[dev] =
            static_cast<int32_t>(solver_status);
      }
      if (info < 0) {
        return absl::InternalError(absl::StrFormat(
            "unexpected error in cusolverMgPotri, %d-th input parameter is "
            "wrong",
            -info));
      }
    }

    if (descr_a != nullptr) {
      JAXMG_RETURN_IF_CUSOLVER_ERROR(cusolverMgDestroyMatrixDesc(descr_a));
    }
    if (grid_a != nullptr) {
      JAXMG_RETURN_IF_CUSOLVER_ERROR(cusolverMgDestroyGrid(grid_a));
    }
    if (cusolver != nullptr) {
      JAXMG_RETURN_IF_CUSOLVER_ERROR(cusolverMgDestroy(cusolver));
    }
    timer.Mark("cleanup");
  }

  fused_potri_barrier.ArriveAndWait(static_cast<int>(num_ranks));
  timer.Mark("solver_barrier");
  const int32_t status_value = fused_potri_state.solver_status[current_device];
  if (status_value == 0) {
    // Convert the cyclic solver output back to JAX's row-sharded layout.
    plan_status = ExecuteMatrixColumnNativePlanRaw(
        "xla_comm_potri_mg_native_plan_reverse", stream, comm_stream, *comm, a,
        out_base, out_base, plan_scratch_base, plan_scratch_base, local_slots,
        n, column_bytes, 1, num_ranks, static_cast<int64_t>(rank->value()),
        tile_size, /*reverse=*/true);
    if (!plan_status.ok()) {
      return plan_status;
    }
    timer.Mark("reverse_cyclic_reshape");
  } else {
    // Match the previous public behavior: failed solver status is returned and
    // the numerical output is poisoned so downstream checks do not silently use
    // stale buffer contents.
    std::vector<typename SolverTraits<DataType>::HostNanType> host_nan(
        static_cast<size_t>(local_slots) * static_cast<size_t>(n),
        SolverTraits<DataType>::Nan());
    JAXMG_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
        out->untyped_data(), host_nan.data(),
        sizeof(DataType) * static_cast<size_t>(local_slots) *
            static_cast<size_t>(n),
        cudaMemcpyHostToDevice, cuda_stream));
  }

  JAXMG_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
      status->typed_data(), &status_value, sizeof(status_value),
      cudaMemcpyHostToDevice, cuda_stream));
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));
  timer.Mark("status_write");
  fused_potri_barrier.ArriveAndWait(static_cast<int>(num_ranks));
  timer.Mark("final_barrier");
  return absl::OkStatus();
}

absl::Status XlaCommPotriMgNativePlanDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    se::OwningScratchAllocator<> scratch, int64_t tile_size, ffi::AnyBuffer a,
    ffi::Result<ffi::AnyBuffer> out,
    ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  switch (a.element_type()) {
    case F32:
      return XlaCommPotriMgNativePlanImpl<float>(
          stream, comm_stream, cuda_stream, scratch, tile_size, a, out, status,
          collective_params, collective_cliques);
    case F64:
      return XlaCommPotriMgNativePlanImpl<double>(
          stream, comm_stream, cuda_stream, scratch, tile_size, a, out, status,
          collective_params, collective_cliques);
    case C64:
      return XlaCommPotriMgNativePlanImpl<cuFloatComplex>(
          stream, comm_stream, cuda_stream, scratch, tile_size, a, out, status,
          collective_params, collective_cliques);
    case C128:
      return XlaCommPotriMgNativePlanImpl<cuDoubleComplex>(
          stream, comm_stream, cuda_stream, scratch, tile_size, a, out, status,
          collective_params, collective_cliques);
    default:
      return absl::InvalidArgumentError(
          "Unsupported dtype for xla_comm_potri_mg_native_plan");
  }
}

}  // namespace xla::gpu
