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
// Fused potrs handler for the single-node XLA communicator backend.
//
// The handler receives donated JAX buffers, converts the row-sharded matrix
// into cuSolverMg's 1D block-cyclic layout, synchronizes with the cuSolverMg
// host call, and broadcasts the replicated RHS result through the same
// XLA-owned communicator path.

#include "include/xla_comm_backend.h"
#include "include/mpmd_ipc.h"

namespace xla::gpu {

template <typename DataType>
absl::Status XlaCommPotrsMgNativePlanImpl(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    se::ScratchAllocator& scratch, int64_t tile_size, ffi::AnyBuffer a,
    ffi::AnyBuffer b, ffi::Result<ffi::AnyBuffer> out_a,
    ffi::Result<ffi::AnyBuffer> out_b,
    ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  const int64_t timing_start = PotrsPhaseTimer::NowNanos();
  if (stream == nullptr || comm_stream == nullptr || cuda_stream == nullptr) {
    return absl::InvalidArgumentError(
        "xla_comm_potrs_mg_native_plan requires stream contexts");
  }
  if (collective_params == nullptr || collective_cliques == nullptr) {
    return absl::InvalidArgumentError(
        "xla_comm_potrs_mg_native_plan requires XLA collective contexts");
  }
  if (tile_size <= 0) {
    return absl::InvalidArgumentError(
        "xla_comm_potrs_mg_native_plan requires positive tile_size");
  }
  if (a.dimensions().size() != 2 || b.dimensions().size() != 2 ||
      out_a->dimensions().size() != 2 || out_b->dimensions().size() != 2 ||
      status->dimensions().size() != 1 || status->dimensions()[0] != 1) {
    return absl::InvalidArgumentError(
        "xla_comm_potrs_mg_native_plan expects A/B rank-2 outputs and status "
        "shape (1,)");
  }
  if (a.dimensions()[0] != out_a->dimensions()[0] ||
      a.dimensions()[1] != out_a->dimensions()[1] ||
      b.dimensions()[0] != out_b->dimensions()[0] ||
      b.dimensions()[1] != out_b->dimensions()[1]) {
    return absl::InvalidArgumentError(
        "xla_comm_potrs_mg_native_plan input/output shapes must match");
  }
  if (a.element_type() != out_a->element_type() ||
      a.element_type() != b.element_type() ||
      b.element_type() != out_b->element_type()) {
    return absl::InvalidArgumentError(
        "xla_comm_potrs_mg_native_plan requires matching A/B dtypes");
  }

  const int64_t local_slots = a.dimensions()[0];
  const int64_t n = a.dimensions()[1];
  const int64_t n_b = b.dimensions()[0];
  const int64_t nrhs = b.dimensions()[1];
  // The Python wrapper supplies a row-sharded A shard of shape
  // (local_slots, N) and a replicated B of shape (N, nrhs). local_slots may be
  // larger than N / num_ranks when JAX-side padding was needed for tile
  // alignment.
  if (n <= 0 || local_slots <= 0 || nrhs <= 0 || n_b != n) {
    return absl::InvalidArgumentError(
        "xla_comm_potrs_mg_native_plan received invalid matrix dimensions");
  }
  if (nrhs > 256) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Number of right hand sides must be <=256, got %d",
                        nrhs));
  }

  absl::StatusOr<GpuCliqueKey> clique_key =
      NodeScopedP2PCliqueKey(*collective_params);
  if (!clique_key.ok()) {
    return clique_key.status();
  }
  const int64_t num_ranks = static_cast<int64_t>(clique_key->num_devices());
  if (num_ranks <= 0 || num_ranks > 16 || n % num_ranks != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "xla_comm_potrs_mg_native_plan requires 1..16 ranks and matrix width "
        "%d divisible by ranks %d",
        n, num_ranks));
  }
  if (local_slots < n / num_ranks) {
    return absl::InvalidArgumentError(
        "xla_comm_potrs_mg_native_plan local A rows are smaller than the "
        "unpadded shard size");
  }
  std::optional<RankId> rank =
      clique_key->rank(collective_params->global_device_id);
  if (!rank.has_value()) {
    return absl::InvalidArgumentError(
        "xla_comm_potrs_mg_native_plan could not resolve this device rank");
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
        "XLA communicator rank %d does not match CUDA device %d",
        rank->value(), current_device));
  }
  const int rank_value = static_cast<int>(rank->value());
  const bool mpmd_mode =
      collective_params->local_device_count > 0 &&
      collective_params->local_device_count < num_ranks;
  PotrsPhaseTimer timer("xla_comm_potrs_mg_native_plan", current_device,
                        static_cast<int>(num_ranks), n, nrhs, tile_size,
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
  se::DeviceAddressBase out_a_base = out_a->device_memory();
  se::DeviceAddressBase plan_scratch_base(*plan_scratch, column_bytes);
  // Convert the donated row-sharded matrix into cuSolverMg's 1D block-cyclic
  // layout before publishing pointers to the rank-0 cuSolverMg host call.
  absl::Status plan_status = ExecuteMatrixColumnNativePlanRaw(
      "xla_comm_potrs_mg_native_plan", stream, comm_stream, *comm, a, a_base,
      out_a_base, plan_scratch_base, plan_scratch_base, local_slots, n,
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

  std::optional<MpmdProcessBarrier> mpmd_barrier;
  std::optional<SharedMemoryArray<IpcHandleWithOffset>> mpmd_a_ipc;
  std::optional<SharedMemoryArray<IpcHandleWithOffset>> mpmd_b_ipc;
  std::optional<SharedMemoryArray<IpcHandleWithOffset>> mpmd_work_ipc;
  std::optional<SharedMemoryArray<int64_t>> mpmd_lwork;
  std::optional<SharedMemoryArray<int32_t>> mpmd_solver_status;
  auto shared_barrier = [&]() -> absl::Status {
    if (mpmd_mode) {
      return mpmd_barrier->ArriveAndWait();
    }
    fused_potrs_barrier.ArriveAndWait(static_cast<int>(num_ranks));
    return absl::OkStatus();
  };

  if (mpmd_mode) {
    absl::StatusOr<int> node_group =
        NodeScopedGroupOrdinal(*collective_params);
    if (!node_group.ok()) {
      return node_group.status();
    }
    const int64_t run_id = collective_params->run_id.ToInt();
    mpmd_barrier.emplace(static_cast<int>(num_ranks),
                         MpmdSharedName("potrs_barrier", run_id, *node_group));
    if (!mpmd_barrier->ok()) {
      return mpmd_barrier->status();
    }
    mpmd_a_ipc.emplace(rank_value, static_cast<int>(num_ranks),
                       MpmdSharedName("potrs_a", run_id, *node_group));
    mpmd_b_ipc.emplace(rank_value, static_cast<int>(num_ranks),
                       MpmdSharedName("potrs_b", run_id, *node_group));
    mpmd_work_ipc.emplace(rank_value, static_cast<int>(num_ranks),
                          MpmdSharedName("potrs_work", run_id, *node_group));
    mpmd_lwork.emplace(rank_value, static_cast<int>(num_ranks),
                       MpmdSharedName("potrs_lwork", run_id, *node_group));
    mpmd_solver_status.emplace(
        rank_value, static_cast<int>(num_ranks),
        MpmdSharedName("potrs_status", run_id, *node_group));
    if (!mpmd_a_ipc->ok()) return mpmd_a_ipc->status();
    if (!mpmd_b_ipc->ok()) return mpmd_b_ipc->status();
    if (!mpmd_work_ipc->ok()) return mpmd_work_ipc->status();
    if (!mpmd_lwork->ok()) return mpmd_lwork->status();
    if (!mpmd_solver_status->ok()) return mpmd_solver_status->status();

    absl::StatusOr<IpcHandleWithOffset> a_handle =
        ExportIpcHandleWithOffset(out_a->untyped_data());
    if (!a_handle.ok()) return a_handle.status();
    absl::StatusOr<IpcHandleWithOffset> b_handle =
        ExportIpcHandleWithOffset(b.untyped_data());
    if (!b_handle.ok()) return b_handle.status();
    (*mpmd_a_ipc)[rank_value] = *a_handle;
    (*mpmd_b_ipc)[rank_value] = *b_handle;
  } else {
    // Each FFI invocation runs on one local device. The rank-0 invocation owns
    // the cuSolverMg handle and consumes the per-device pointers published
    // here.
    fused_potrs_state.a[current_device] = out_a->untyped_data();
    fused_potrs_state.b[current_device] = b.untyped_data();
  }
  JAXMG_RETURN_IF_ERROR(shared_barrier());
  timer.Mark("pointer_share_barrier");

  const int ia = 1;
  const int ja = 1;
  const int ib = 1;
  const int jb = 1;
  const int t_a = static_cast<int>(std::min(tile_size, local_slots));
  const int t_b = static_cast<int>(nrhs);
  const cudaDataType compute_type = SolverTraits<DataType>::cuda_data_type;

  std::array<int, 16> device_list{};
  std::array<void*, 16> a_ptrs{};
  std::array<void*, 16> b_ptrs{};
  std::array<void*, 16> work_ptrs{};
  std::array<OpenedIpcPointer, 16> opened_a{};
  std::array<OpenedIpcPointer, 16> opened_b{};
  std::array<OpenedIpcPointer, 16> opened_work{};

  cusolverMgHandle_t cusolver = nullptr;
  cudaLibMgGrid_t grid_a = nullptr;
  cudaLibMgGrid_t grid_b = nullptr;
  cudaLibMgMatrixDesc_t descr_a = nullptr;
  cudaLibMgMatrixDesc_t descr_b = nullptr;
  int info = 0;
  cusolverStatus_t solver_status = CUSOLVER_STATUS_SUCCESS;

  if (current_device == 0) {
    // cuSolverMg is driven from one host invocation. Rank 0 creates the handle,
    // builds descriptor objects over all local device pointers, and publishes
    // the workspace size required on every participating device.
    for (int dev = 0; dev < num_ranks; ++dev) {
      device_list[dev] = dev;
      if (mpmd_mode && dev != current_device) {
        absl::StatusOr<OpenedIpcPointer> opened =
            OpenIpcHandleWithOffsetOnDevice((*mpmd_a_ipc)[dev], dev);
        if (!opened.ok()) return opened.status();
        opened_a[dev] = *opened;
        a_ptrs[dev] = opened_a[dev].ptr;

        opened = OpenIpcHandleWithOffsetOnDevice((*mpmd_b_ipc)[dev], dev);
        if (!opened.ok()) return opened.status();
        opened_b[dev] = *opened;
        b_ptrs[dev] = opened_b[dev].ptr;
      } else {
        a_ptrs[dev] =
            mpmd_mode ? out_a->untyped_data() : fused_potrs_state.a[dev];
        b_ptrs[dev] =
            mpmd_mode ? b.untyped_data() : fused_potrs_state.b[dev];
      }
    }

    JAXMG_RETURN_IF_CUSOLVER_ERROR(cusolverMgCreate(&cusolver));
    JAXMG_RETURN_IF_CUSOLVER_ERROR(
        cusolverMgDeviceSelect(cusolver, static_cast<int>(num_ranks),
                               device_list.data()));
    JAXMG_RETURN_IF_CUSOLVER_ERROR(cusolverMgCreateDeviceGrid(
        &grid_a, 1, static_cast<int>(num_ranks), device_list.data(),
        CUDALIBMG_GRID_MAPPING_COL_MAJOR));
    JAXMG_RETURN_IF_CUSOLVER_ERROR(cusolverMgCreateDeviceGrid(
        &grid_b, 1, static_cast<int>(num_ranks), device_list.data(),
        CUDALIBMG_GRID_MAPPING_COL_MAJOR));
    JAXMG_RETURN_IF_CUSOLVER_ERROR(cusolverMgCreateMatrixDesc(
        &descr_a, n, n, n, t_a, compute_type, grid_a));
    JAXMG_RETURN_IF_CUSOLVER_ERROR(cusolverMgCreateMatrixDesc(
        &descr_b, n, nrhs, n, t_b, compute_type, grid_b));

    int64_t lwork_potrf = 0;
    int64_t lwork_potrs = 0;
    JAXMG_RETURN_IF_CUSOLVER_ERROR(cusolverMgPotrf_bufferSize(
        cusolver, CUBLAS_FILL_MODE_LOWER, n, a_ptrs.data(), ia, ja, descr_a,
        compute_type, &lwork_potrf));
    JAXMG_RETURN_IF_CUSOLVER_ERROR(cusolverMgPotrs_bufferSize(
        cusolver, CUBLAS_FILL_MODE_LOWER, n, nrhs, a_ptrs.data(), ia, ja,
        descr_a, b_ptrs.data(), ib, jb, descr_b, compute_type, &lwork_potrs));
    const int64_t lwork = std::max(lwork_potrf, lwork_potrs);
    for (int dev = 0; dev < num_ranks; ++dev) {
      if (mpmd_mode) {
        (*mpmd_lwork)[dev] = lwork;
      } else {
        fused_potrs_state.lwork[dev] = lwork;
      }
    }
    timer.Mark("cusolver_setup_and_workspace_query");
  }

  JAXMG_RETURN_IF_ERROR(shared_barrier());
  timer.Mark("lwork_barrier");
  // Scratch allocation is per FFI invocation, hence per local device. The
  // pointer is then shared back to rank 0 before the cuSolverMg call.
  const int64_t local_lwork =
      mpmd_mode ? (*mpmd_lwork)[rank_value]
                : fused_potrs_state.lwork[current_device];
  absl::StatusOr<void*> solver_work = AllocateFfiScratch(
      scratch, sizeof(DataType) * local_lwork,
      "cusolvermg_potrs_workspace");
  if (!solver_work.ok()) {
    return solver_work.status();
  }
  if (mpmd_mode) {
    absl::StatusOr<IpcHandleWithOffset> work_handle =
        ExportIpcHandleWithOffset(*solver_work);
    if (!work_handle.ok()) return work_handle.status();
    (*mpmd_work_ipc)[rank_value] = *work_handle;
  } else {
    fused_potrs_state.work[current_device] = *solver_work;
  }
  timer.Mark("workspace_alloc");
  JAXMG_RETURN_IF_ERROR(shared_barrier());
  timer.Mark("workspace_barrier");

  if (current_device == 0) {
    // This preserves the old JAXMg potrs sequence: factor A in-place with
    // potrf, then solve using potrs and the replicated RHS pointers.
    for (int dev = 0; dev < num_ranks; ++dev) {
      if (mpmd_mode && dev != current_device) {
        absl::StatusOr<OpenedIpcPointer> opened =
            OpenIpcHandleWithOffsetOnDevice((*mpmd_work_ipc)[dev], dev);
        if (!opened.ok()) return opened.status();
        opened_work[dev] = *opened;
        work_ptrs[dev] = opened_work[dev].ptr;
      } else {
        work_ptrs[dev] =
            mpmd_mode ? *solver_work : fused_potrs_state.work[dev];
      }
    }
    solver_status = cusolverMgPotrf(
        cusolver, CUBLAS_FILL_MODE_LOWER, n, a_ptrs.data(), ia, ja, descr_a,
        compute_type, work_ptrs.data(),
        mpmd_mode ? (*mpmd_lwork)[0] : fused_potrs_state.lwork[0], &info);
    JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));
    timer.Mark("potrf");
    for (int dev = 0; dev < num_ranks; ++dev) {
      if (mpmd_mode) {
        (*mpmd_solver_status)[dev] = static_cast<int32_t>(solver_status);
      } else {
        fused_potrs_state.solver_status[dev] =
            static_cast<int32_t>(solver_status);
      }
    }
    if (info < 0) {
      return absl::InternalError(absl::StrFormat(
          "unexpected error in cusolverMgPotrf, %d-th input parameter is wrong",
          -info));
    }

    if (solver_status == CUSOLVER_STATUS_SUCCESS) {
      solver_status = cusolverMgPotrs(
          cusolver, CUBLAS_FILL_MODE_LOWER, n, nrhs, a_ptrs.data(), ia, ja,
          descr_a, b_ptrs.data(), ib, jb, descr_b, compute_type,
          work_ptrs.data(),
          mpmd_mode ? (*mpmd_lwork)[0] : fused_potrs_state.lwork[0], &info);
      JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));
      timer.Mark("potrs");
      for (int dev = 0; dev < num_ranks; ++dev) {
        if (mpmd_mode) {
          (*mpmd_solver_status)[dev] = static_cast<int32_t>(solver_status);
        } else {
          fused_potrs_state.solver_status[dev] =
              static_cast<int32_t>(solver_status);
        }
      }
      if (info < 0) {
        return absl::InternalError(absl::StrFormat(
            "unexpected error in cusolverMgPotrs, %d-th input parameter is "
            "wrong",
            -info));
      }
    }

    if (descr_a != nullptr) {
      JAXMG_RETURN_IF_CUSOLVER_ERROR(cusolverMgDestroyMatrixDesc(descr_a));
    }
    if (descr_b != nullptr) {
      JAXMG_RETURN_IF_CUSOLVER_ERROR(cusolverMgDestroyMatrixDesc(descr_b));
    }
    if (grid_a != nullptr) {
      JAXMG_RETURN_IF_CUSOLVER_ERROR(cusolverMgDestroyGrid(grid_a));
    }
    if (grid_b != nullptr) {
      JAXMG_RETURN_IF_CUSOLVER_ERROR(cusolverMgDestroyGrid(grid_b));
    }
    if (cusolver != nullptr) {
      JAXMG_RETURN_IF_CUSOLVER_ERROR(cusolverMgDestroy(cusolver));
    }
    if (mpmd_mode) {
      for (int dev = 0; dev < num_ranks; ++dev) {
        if (dev == current_device) {
          continue;
        }
        JAXMG_RETURN_IF_ERROR(CloseIpcPointer(opened_work[dev]));
        JAXMG_RETURN_IF_ERROR(CloseIpcPointer(opened_b[dev]));
        JAXMG_RETURN_IF_ERROR(CloseIpcPointer(opened_a[dev]));
      }
    }
    timer.Mark("cleanup");
  }

  JAXMG_RETURN_IF_ERROR(shared_barrier());
  timer.Mark("solver_barrier");
  const int32_t status_value =
      mpmd_mode ? (*mpmd_solver_status)[rank_value]
                : fused_potrs_state.solver_status[current_device];
  if (status_value == 0) {
    // B is replicated at the Python level. cuSolverMg updates the rank-0 host
    // path's B pointer, so the result is redistributed with the same XLA
    // communicator rather than a CUDA peer broadcast.
    absl::Status broadcast_status = BroadcastBufferFromRank0(
        "xla_comm_potrs_mg_native_plan_b_broadcast", stream, comm_stream, *comm,
        out_b->device_memory(), out_b->element_type(),
        static_cast<size_t>(out_b->element_count()), num_ranks,
        static_cast<int64_t>(rank->value()));
    if (!broadcast_status.ok()) {
      return broadcast_status;
    }
    timer.Mark("b_broadcast");
  }
  JAXMG_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
      status->typed_data(), &status_value, sizeof(status_value),
      cudaMemcpyHostToDevice, cuda_stream));
  if (status_value != 0) {
    // Match the previous public behavior: failed solver status is returned and
    // the numerical output is poisoned so downstream checks do not silently use
    // stale buffer contents.
    std::vector<typename SolverTraits<DataType>::HostNanType> host_nan(
        n * nrhs, SolverTraits<DataType>::Nan());
    JAXMG_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
        out_b->untyped_data(), host_nan.data(), sizeof(DataType) * n * nrhs,
        cudaMemcpyHostToDevice, cuda_stream));
  }
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));
  timer.Mark("status_write");
  JAXMG_RETURN_IF_ERROR(shared_barrier());
  timer.Mark("final_barrier");
  if (mpmd_mode) {
    JAXMG_RETURN_IF_ERROR(mpmd_a_ipc->CloseAndUnlink());
    JAXMG_RETURN_IF_ERROR(mpmd_b_ipc->CloseAndUnlink());
    JAXMG_RETURN_IF_ERROR(mpmd_work_ipc->CloseAndUnlink());
    JAXMG_RETURN_IF_ERROR(mpmd_lwork->CloseAndUnlink());
    JAXMG_RETURN_IF_ERROR(mpmd_solver_status->CloseAndUnlink());
    JAXMG_RETURN_IF_ERROR(mpmd_barrier->CloseAndUnlink());
  }
  return absl::OkStatus();
}

absl::Status XlaCommPotrsMgNativePlanDispatch(
    se::Stream* stream, se::Stream* comm_stream, cudaStream_t cuda_stream,
    se::OwningScratchAllocator<> scratch, int64_t tile_size, ffi::AnyBuffer a,
    ffi::AnyBuffer b, ffi::Result<ffi::AnyBuffer> out_a,
    ffi::Result<ffi::AnyBuffer> out_b,
    ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  switch (a.element_type()) {
    case F32:
      return XlaCommPotrsMgNativePlanImpl<float>(
          stream, comm_stream, cuda_stream, scratch, tile_size, a, b, out_a,
          out_b, status, collective_params, collective_cliques);
    case F64:
      return XlaCommPotrsMgNativePlanImpl<double>(
          stream, comm_stream, cuda_stream, scratch, tile_size, a, b, out_a,
          out_b, status, collective_params, collective_cliques);
    case C64:
      return XlaCommPotrsMgNativePlanImpl<cuFloatComplex>(
          stream, comm_stream, cuda_stream, scratch, tile_size, a, b, out_a,
          out_b, status, collective_params, collective_cliques);
    case C128:
      return XlaCommPotrsMgNativePlanImpl<cuDoubleComplex>(
          stream, comm_stream, cuda_stream, scratch, tile_size, a, b, out_a,
          out_b, status, collective_params, collective_cliques);
    default:
      return absl::InvalidArgumentError(
          "Unsupported dtype for xla_comm_potrs_mg_native_plan");
  }
}

}  // namespace xla::gpu
