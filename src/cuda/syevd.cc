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
// Fused syevd handlers for the single-node XLA communicator backend.
//
// The vector-returning path reuses the eigenvector output as cyclic matrix
// storage. The no-vector path allocates a full local cyclic matrix scratch
// because there is no vector output buffer to donate for that purpose.

#include "include/xla_comm_backend.h"
#include "include/mpmd_ipc.h"

namespace xla::gpu {

template <typename DataType>
absl::Status XlaCommSyevdMgNativePlanImpl(
    const char* caller, bool compute_vectors, ReusableHostBarrier& barrier,
    FusedSyevdState& state, se::Stream* stream, se::Stream* comm_stream,
    cudaStream_t cuda_stream, se::ScratchAllocator& scratch, int64_t tile_size,
    ffi::AnyBuffer a, ffi::Result<ffi::AnyBuffer> eigenvalues,
    ffi::Result<ffi::AnyBuffer>* vectors,
    ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  const int64_t timing_start = PotrsPhaseTimer::NowNanos();
  if (stream == nullptr || comm_stream == nullptr || cuda_stream == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s requires stream contexts", caller));
  }
  if (collective_params == nullptr || collective_cliques == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s requires XLA collective contexts", caller));
  }
  if (tile_size <= 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s requires positive tile_size", caller));
  }
  if (a.dimensions().size() != 2 || eigenvalues->dimensions().size() != 1 ||
      status->dimensions().size() != 1 || status->dimensions()[0] != 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s expects A rank-2, eigenvalues rank-1, and status shape (1,)",
        caller));
  }
  if (compute_vectors) {
    if (vectors == nullptr || (*vectors)->dimensions().size() != 2 ||
        a.dimensions()[0] != (*vectors)->dimensions()[0] ||
        a.dimensions()[1] != (*vectors)->dimensions()[1] ||
        a.element_type() != (*vectors)->element_type()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "%s expects vector output to match A shape and dtype", caller));
    }
  }

  const int64_t local_slots = a.dimensions()[0];
  const int64_t n = a.dimensions()[1];
  // The Python wrapper supplies a row-sharded matrix shard of shape
  // (local_slots, N). local_slots may include tile padding that is not part of
  // the global mathematical matrix.
  if (n <= 0 || local_slots <= 0 || eigenvalues->dimensions()[0] != n) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s received invalid matrix/eigenvalue dimensions",
                        caller));
  }

  absl::StatusOr<GpuCliqueKey> clique_key =
      NodeScopedP2PCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    return clique_key.status();
  }
  const int64_t num_ranks = static_cast<int64_t>(clique_key->num_devices());
  if (num_ranks <= 0 || num_ranks > 16 || n % num_ranks != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s requires 1..16 ranks and matrix width %d divisible by ranks %d",
        caller, n, num_ranks));
  }
  if (local_slots < n / num_ranks) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s local A rows are smaller than the unpadded shard size", caller));
  }
  std::optional<RankId> rank =
      clique_key->rank(collective_params->global_device_id);
  if (!rank.has_value()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s could not resolve this device rank", caller));
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
  const int rank_value = static_cast<int>(rank->value());
  const bool mpmd_mode =
      collective_params->local_device_count > 0 &&
      collective_params->local_device_count < num_ranks;

  PotrsPhaseTimer timer(caller, current_device, static_cast<int>(num_ranks), n,
                        0, tile_size, timing_start);
  timer.Mark("xla_comm_lookup_and_device_setup");

  const size_t element_bytes = sizeof(DataType);
  const uint64_t column_bytes = static_cast<uint64_t>(n) * element_bytes;
  absl::StatusOr<void*> plan_scratch =
      AllocateFfiScratch(scratch, column_bytes, "native_plan_column_scratch");
  if (!plan_scratch.ok()) {
    return plan_scratch.status();
  }

  const size_t matrix_bytes =
      sizeof(DataType) * static_cast<size_t>(local_slots) *
      static_cast<size_t>(n);
  // If eigenvectors are requested, the output vector buffer can temporarily
  // hold the cyclic matrix and is later reshuffled back. If only eigenvalues
  // are requested, there is no donated matrix output, so allocate scratch.
  void* cyclic_matrix_ptr =
      compute_vectors ? (*vectors)->untyped_data() : nullptr;
  if (!compute_vectors) {
    absl::StatusOr<void*> cyclic_matrix_scratch =
        AllocateFfiScratch(scratch, matrix_bytes, "syevd_cyclic_matrix");
    if (!cyclic_matrix_scratch.ok()) {
      return cyclic_matrix_scratch.status();
    }
    cyclic_matrix_ptr = *cyclic_matrix_scratch;
  }
  timer.Mark("scratch_alloc");

  se::DeviceAddressBase a_base = a.device_memory();
  se::DeviceAddressBase cyclic_matrix_base(cyclic_matrix_ptr, matrix_bytes);
  se::DeviceAddressBase plan_scratch_base(*plan_scratch, column_bytes);
  // Convert the donated row-sharded matrix into cuSolverMg's 1D block-cyclic
  // layout before publishing pointers to the rank-0 cuSolverMg host call.
  absl::Status plan_status = ExecuteMatrixColumnNativePlanRaw(
      caller, stream, comm_stream, *comm, a, a_base, cyclic_matrix_base,
      plan_scratch_base, plan_scratch_base, local_slots, n, column_bytes, 1,
      num_ranks, static_cast<int64_t>(rank->value()), tile_size,
      /*reverse=*/false);
  if (!plan_status.ok()) {
    return plan_status;
  }
  timer.Mark("cyclic_reshape");
  // cuSolverMg does not expose a per-call stream setter in this path, so keep a
  // conservative synchronization before invoking the host-side cuSolverMg API.
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));
  timer.Mark("cyclic_reshape_stream_sync");

  std::optional<MpmdSolverExchange> mpmd_exchange;
  auto shared_barrier = [&]() -> absl::Status {
    if (mpmd_mode) {
      return mpmd_exchange->ArriveAndWait();
    }
    barrier.ArriveAndWait(static_cast<int>(num_ranks));
    return absl::OkStatus();
  };

  if (mpmd_mode) {
    absl::StatusOr<int> node_group =
        NodeScopedGroupOrdinal(*collective_params);
    if (!node_group.ok()) {
      return node_group.status();
    }
    const int64_t run_id = collective_params->run_id.ToInt();
    const char* exchange_prefix = compute_vectors ? "syevd" : "syevd_no_v";
    mpmd_exchange.emplace(MpmdSolverExchangeConfig{
        rank_value, static_cast<int>(num_ranks), run_id, *node_group,
        exchange_prefix, /*include_b=*/false});
    if (!mpmd_exchange->ok()) {
      return mpmd_exchange->status();
    }
    JAXMG_RETURN_IF_ERROR(mpmd_exchange->PublishA(cyclic_matrix_ptr));
  } else {
    // Each FFI invocation runs on one local device. The rank-0 invocation owns
    // the cuSolverMg handle and consumes the per-device pointers published
    // here.
    state.a[current_device] = cyclic_matrix_ptr;
    state.eigenvalues[current_device] = eigenvalues->untyped_data();
  }
  JAXMG_RETURN_IF_ERROR(shared_barrier());
  timer.Mark("pointer_share_barrier");

  using EigenvalueType = typename SolverTraits<DataType>::EigenvalueType;
  const int ia = 1;
  const int ja = 1;
  const int t_a = static_cast<int>(std::min(tile_size, local_slots));
  const cudaDataType compute_type = SolverTraits<DataType>::cuda_data_type;
  const cudaDataType eigenvalue_type =
      SolverTraits<DataType>::eigenvalue_cuda_data_type;
  const cusolverEigMode_t jobz =
      compute_vectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;

  std::array<int, 16> device_list{};
  std::array<void*, 16> a_ptrs{};
  std::array<void*, 16> work_ptrs{};
  std::array<OpenedIpcPointer, 16> opened_a{};
  std::array<OpenedIpcPointer, 16> opened_work{};
  std::vector<EigenvalueType> eigenvalues_host(
      static_cast<size_t>(n), static_cast<EigenvalueType>(0));

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
      if (mpmd_mode && dev != current_device) {
        absl::StatusOr<void*> remote_a =
            mpmd_exchange->OpenAOnDevice(dev, &opened_a[dev]);
        if (!remote_a.ok()) return remote_a.status();
        a_ptrs[dev] = *remote_a;
      } else {
        a_ptrs[dev] = mpmd_mode ? cyclic_matrix_ptr : state.a[dev];
      }
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

    int64_t lwork_syevd = 0;
    JAXMG_RETURN_IF_CUSOLVER_ERROR(cusolverMgSyevd_bufferSize(
        cusolver, jobz, CUBLAS_FILL_MODE_LOWER, n, a_ptrs.data(), ia, ja,
        descr_a, reinterpret_cast<void*>(eigenvalues_host.data()),
        eigenvalue_type, compute_type, &lwork_syevd));
    for (int dev = 0; dev < num_ranks; ++dev) {
      if (mpmd_mode) {
        mpmd_exchange->SetLwork(dev, lwork_syevd);
      } else {
        state.lwork[dev] = lwork_syevd;
      }
    }
    timer.Mark("cusolver_setup_and_workspace_query");
  }

  JAXMG_RETURN_IF_ERROR(shared_barrier());
  timer.Mark("lwork_barrier");
  // Scratch allocation is per FFI invocation, hence per local device. The
  // pointer is then shared back to rank 0 before the cuSolverMg call.
  const int64_t local_lwork =
      mpmd_mode ? mpmd_exchange->Lwork(rank_value)
                : state.lwork[current_device];
  absl::StatusOr<void*> solver_work = AllocateFfiScratch(
      scratch, sizeof(DataType) * local_lwork,
      compute_vectors ? "cusolvermg_syevd_workspace"
                      : "cusolvermg_syevd_no_v_workspace");
  if (!solver_work.ok()) {
    return solver_work.status();
  }
  if (mpmd_mode) {
    JAXMG_RETURN_IF_ERROR(mpmd_exchange->PublishWork(*solver_work));
  } else {
    state.work[current_device] = *solver_work;
  }
  timer.Mark("workspace_alloc");
  JAXMG_RETURN_IF_ERROR(shared_barrier());
  timer.Mark("workspace_barrier");

  if (current_device == 0) {
    // cuSolverMg writes eigenvalues through a host pointer. Rank 0 later copies
    // them to device memory and broadcasts them to the replicated JAX result.
    for (int dev = 0; dev < num_ranks; ++dev) {
      if (mpmd_mode && dev != current_device) {
        absl::StatusOr<void*> remote_work =
            mpmd_exchange->OpenWorkOnDevice(dev, &opened_work[dev]);
        if (!remote_work.ok()) return remote_work.status();
        work_ptrs[dev] = *remote_work;
      } else {
        work_ptrs[dev] = mpmd_mode ? *solver_work : state.work[dev];
      }
    }
    solver_status = cusolverMgSyevd(
        cusolver, jobz, CUBLAS_FILL_MODE_LOWER, n, a_ptrs.data(), ia, ja,
        descr_a, reinterpret_cast<void*>(eigenvalues_host.data()),
        eigenvalue_type, compute_type, work_ptrs.data(),
        mpmd_mode ? mpmd_exchange->Lwork(0) : state.lwork[0], &info);
    JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));
    timer.Mark(compute_vectors ? "syevd" : "syevd_no_v");
    for (int dev = 0; dev < num_ranks; ++dev) {
      if (mpmd_mode) {
        mpmd_exchange->SetSolverStatus(dev,
                                       static_cast<int32_t>(solver_status));
      } else {
        state.solver_status[dev] = static_cast<int32_t>(solver_status);
      }
    }
    if (info < 0) {
      return absl::InternalError(absl::StrFormat(
          "unexpected error in cusolverMgSyevd, %d-th input parameter is wrong",
          -info));
    }
    if (solver_status == CUSOLVER_STATUS_SUCCESS) {
      JAXMG_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
          eigenvalues->untyped_data(), eigenvalues_host.data(),
          sizeof(EigenvalueType) * static_cast<size_t>(n),
          cudaMemcpyHostToDevice, cuda_stream));
      timer.Mark("eigenvalues_host_to_device");
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
    if (mpmd_mode) {
      JAXMG_RETURN_IF_ERROR(CloseOpenedRemoteIpcPointers(
          static_cast<int>(num_ranks), current_device,
          {opened_work.data(), opened_a.data()}));
    }
    timer.Mark("cleanup");
  }

  JAXMG_RETURN_IF_ERROR(shared_barrier());
  timer.Mark("solver_barrier");
  const int32_t status_value =
      mpmd_mode ? mpmd_exchange->SolverStatus(rank_value)
                : state.solver_status[current_device];
  if (status_value == 0) {
    // Eigenvalues are replicated at the Python level, so every local device
    // receives the rank-0 result through the XLA communicator.
    absl::Status broadcast_status = BroadcastBufferFromRank0(
        caller, stream, comm_stream, *comm, eigenvalues->device_memory(),
        eigenvalues->element_type(), static_cast<size_t>(eigenvalues->element_count()),
        num_ranks, static_cast<int64_t>(rank->value()));
    if (!broadcast_status.ok()) {
      return broadcast_status;
    }
    timer.Mark("eigenvalues_broadcast");

    if (compute_vectors) {
      // Convert cyclic eigenvectors back to JAX's row-sharded layout.
      plan_status = ExecuteMatrixColumnNativePlanRaw(
          caller, stream, comm_stream, *comm, a, cyclic_matrix_base,
          cyclic_matrix_base, plan_scratch_base, plan_scratch_base, local_slots,
          n, column_bytes, 1, num_ranks, static_cast<int64_t>(rank->value()),
          tile_size, /*reverse=*/true);
      if (!plan_status.ok()) {
        return plan_status;
      }
      timer.Mark("reverse_cyclic_reshape");
    }
  } else {
    // Match the previous public behavior: failed solver status is returned and
    // numerical outputs are poisoned so downstream checks do not silently use
    // stale buffer contents.
    std::vector<EigenvalueType> host_ev_nan(
        static_cast<size_t>(n), SolverTraits<DataType>::EigenvalueNan());
    JAXMG_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
        eigenvalues->untyped_data(), host_ev_nan.data(),
        sizeof(EigenvalueType) * static_cast<size_t>(n),
        cudaMemcpyHostToDevice, cuda_stream));
    if (compute_vectors) {
      std::vector<typename SolverTraits<DataType>::HostNanType> host_v_nan(
          static_cast<size_t>(local_slots) * static_cast<size_t>(n),
          SolverTraits<DataType>::Nan());
      JAXMG_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
          (*vectors)->untyped_data(), host_v_nan.data(),
          sizeof(DataType) * static_cast<size_t>(local_slots) *
              static_cast<size_t>(n),
          cudaMemcpyHostToDevice, cuda_stream));
    }
  }

  JAXMG_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
      status->typed_data(), &status_value, sizeof(status_value),
      cudaMemcpyHostToDevice, cuda_stream));
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));
  timer.Mark("status_write");
  JAXMG_RETURN_IF_ERROR(shared_barrier());
  timer.Mark("final_barrier");
  if (mpmd_mode) {
    JAXMG_RETURN_IF_ERROR(mpmd_exchange->CloseAndUnlink());
  }
  return absl::OkStatus();
}

absl::Status XlaCommSyevdMgNativePlanDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    se::OwningScratchAllocator<> scratch, int64_t tile_size, ffi::AnyBuffer a,
    ffi::Result<ffi::AnyBuffer> eigenvalues, ffi::Result<ffi::AnyBuffer> vectors,
    ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  switch (a.element_type()) {
    case F32:
      return XlaCommSyevdMgNativePlanImpl<float>(
          "xla_comm_syevd_mg_native_plan", true, fused_syevd_barrier,
          fused_syevd_state, stream, comm_stream, cuda_stream, scratch,
          tile_size, a, eigenvalues, &vectors, status, collective_params,
          collective_cliques);
    case F64:
      return XlaCommSyevdMgNativePlanImpl<double>(
          "xla_comm_syevd_mg_native_plan", true, fused_syevd_barrier,
          fused_syevd_state, stream, comm_stream, cuda_stream, scratch,
          tile_size, a, eigenvalues, &vectors, status, collective_params,
          collective_cliques);
    case C64:
      return XlaCommSyevdMgNativePlanImpl<cuFloatComplex>(
          "xla_comm_syevd_mg_native_plan", true, fused_syevd_barrier,
          fused_syevd_state, stream, comm_stream, cuda_stream, scratch,
          tile_size, a, eigenvalues, &vectors, status, collective_params,
          collective_cliques);
    case C128:
      return XlaCommSyevdMgNativePlanImpl<cuDoubleComplex>(
          "xla_comm_syevd_mg_native_plan", true, fused_syevd_barrier,
          fused_syevd_state, stream, comm_stream, cuda_stream, scratch,
          tile_size, a, eigenvalues, &vectors, status, collective_params,
          collective_cliques);
    default:
      return absl::InvalidArgumentError(
          "Unsupported dtype for xla_comm_syevd_mg_native_plan");
  }
}

absl::Status XlaCommSyevdNoVMgNativePlanDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    se::OwningScratchAllocator<> scratch, int64_t tile_size, ffi::AnyBuffer a,
    ffi::Result<ffi::AnyBuffer> eigenvalues,
    ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  switch (a.element_type()) {
    case F32:
      return XlaCommSyevdMgNativePlanImpl<float>(
          "xla_comm_syevd_no_v_mg_native_plan", false,
          fused_syevd_no_v_barrier, fused_syevd_no_v_state, stream,
          comm_stream, cuda_stream, scratch, tile_size, a, eigenvalues,
          nullptr, status, collective_params, collective_cliques);
    case F64:
      return XlaCommSyevdMgNativePlanImpl<double>(
          "xla_comm_syevd_no_v_mg_native_plan", false,
          fused_syevd_no_v_barrier, fused_syevd_no_v_state, stream,
          comm_stream, cuda_stream, scratch, tile_size, a, eigenvalues,
          nullptr, status, collective_params, collective_cliques);
    case C64:
      return XlaCommSyevdMgNativePlanImpl<cuFloatComplex>(
          "xla_comm_syevd_no_v_mg_native_plan", false,
          fused_syevd_no_v_barrier, fused_syevd_no_v_state, stream,
          comm_stream, cuda_stream, scratch, tile_size, a, eigenvalues,
          nullptr, status, collective_params, collective_cliques);
    case C128:
      return XlaCommSyevdMgNativePlanImpl<cuDoubleComplex>(
          "xla_comm_syevd_no_v_mg_native_plan", false,
          fused_syevd_no_v_barrier, fused_syevd_no_v_state, stream,
          comm_stream, cuda_stream, scratch, tile_size, a, eigenvalues,
          nullptr, status, collective_params, collective_cliques);
    default:
      return absl::InvalidArgumentError(
          "Unsupported dtype for xla_comm_syevd_no_v_mg_native_plan");
  }
}

}  // namespace xla::gpu
