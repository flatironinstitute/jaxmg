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
// cuSOLVERMp LU factorization FFI handler.
//
// This file implements the factor-only GETRF path. GETRF overwrites the
// donated A buffer with the packed unit-lower L and upper U factors and writes
// its row interchanges into a distributed pivot array. The pivots are gathered
// into one replicated, 0-based vector before the packed factor is restored to
// JAX.
//
// File workflow:
//   1. Convert A to local column-major storage and redistribute it into the
//      cuSOLVERMp 2D block-cyclic layout.
//   2. Run GETRF.
//   3. Scatter each rank's local pivots into a global vector and replicate it
//      with an NCCL max all-reduce.
//   4. Reverse-redistribute LU and restore row-major JAX storage.

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "cusolvermp_common.h"
#include "cusolvermp_routines.h"
#include "factor_extract.h"
#include "factorization.h"

namespace xla::gpu {
namespace {

// cuSOLVERMp follows ScaLAPACK: pivot entries are 1-based global row indices.
constexpr int64_t kGetrfPivotIndexBase = 1;

struct CudaFreeDeleter {
  void operator()(void* ptr) const {
    if (ptr != nullptr) cudaFree(ptr);
  }
};

template <typename DataType>
absl::Status RunCusolverMpGetrf(
    const CusolverMpApi& api, cusolverMpHandle_t handle, cusolverMpGrid_t grid,
    cudaStream_t cuda_stream, int64_t n, int64_t tile_size,
    int32_t process_row, int32_t process_col,
    ffi::Result<ffi::AnyBuffer> lu_out, int64_t* d_ipiv, size_t ipiv_bytes,
    std::array<int32_t, kGetrfStatusSize>* status_words) {
  CusolverMpDebug((*status_words)[2],
                  "getrf enter n=%lld tile=%lld process_coord=(%d,%d) "
                  "dtype=%d",
                  static_cast<long long>(n), static_cast<long long>(tile_size),
                  process_row, process_col,
                  static_cast<int>(SolverTraits<DataType>::cuda_data_type));

  cusolverMpMatrixDescriptor_t desc_a = nullptr;
  const cusolverStatus_t solver_status = api.create_matrix_desc(
      &desc_a, grid, SolverTraits<DataType>::cuda_data_type, n, n, tile_size,
      tile_size, /*RSRC_A=*/0, /*CSRC_A=*/0, lu_out->dimensions()[0]);
  if (solver_status != CUSOLVER_STATUS_SUCCESS || desc_a == nullptr) {
    (*status_words)[0] = kCreateMatrixDescFailed;
    (*status_words)[11] = static_cast<int32_t>(solver_status);
    if (desc_a != nullptr) api.destroy_matrix_desc(desc_a);
    return absl::OkStatus();
  }
  (*status_words)[10] = 1;

  // The caller retains the native pivots for gathering after factorization.
  const absl::Status result = RunLuFactorization(
      api, handle, desc_a, cuda_stream, n, lu_out->untyped_data(),
      SolverTraits<DataType>::cuda_data_type, d_ipiv, ipiv_bytes,
      {(*status_words)[0], (*status_words)[11], (*status_words)[19],
       (*status_words)[20], (*status_words)[21], (*status_words)[22]});
  const cusolverStatus_t destroy_status = api.destroy_matrix_desc(desc_a);
  if (destroy_status != CUSOLVER_STATUS_SUCCESS &&
      (*status_words)[0] == kStatusOk) {
    (*status_words)[0] = kDestroyMatrixDescFailed;
    (*status_words)[11] = static_cast<int32_t>(destroy_status);
  }
  return result;
}

// Prints this rank's raw GETRF pivots when JAXMG_CUSOLVERMP_DEBUG is set, so
// the pivot layout cuSOLVERMp uses on a grid can be inspected directly.
void DebugDumpLocalPivots(cudaStream_t cuda_stream, int rank,
                          const int64_t* d_ipiv, int64_t ipiv_len,
                          int32_t process_row, int32_t process_col,
                          int64_t numroc_rows, int64_t numroc_cols) {
  if (!CusolverMpDebugEnabled()) return;
  const int64_t dump_len = std::min<int64_t>(ipiv_len, 64);
  std::vector<int64_t> host(static_cast<size_t>(dump_len));
  if (cudaMemcpyAsync(host.data(), d_ipiv, dump_len * sizeof(int64_t),
                      cudaMemcpyDeviceToHost, cuda_stream) != cudaSuccess ||
      cudaStreamSynchronize(cuda_stream) != cudaSuccess) {
    CusolverMpDebug(rank, "getrf local ipiv copy failed");
    return;
  }
  std::string values;
  for (int64_t value : host) {
    values += absl::StrFormat(" %lld", static_cast<long long>(value));
  }
  CusolverMpDebug(rank,
                  "getrf local ipiv process_coord=(%d,%d) numroc_rows=%lld "
                  "numroc_cols=%lld ipiv_len=%lld values:%s",
                  process_row, process_col,
                  static_cast<long long>(numroc_rows),
                  static_cast<long long>(numroc_cols),
                  static_cast<long long>(ipiv_len), values.c_str());
}

absl::Status RunCusolverMpGetrfSolver(
    se::Stream* stream, cudaStream_t cuda_stream, int64_t process_rows,
    int64_t process_cols, int64_t n, int64_t tile_size, int64_t grid_mapping,
    absl::Span<const int64_t> rank_map, ffi::Result<ffi::AnyBuffer> lu_out,
    ffi::Result<ffi::BufferR1<S32>> piv_out,
    ffi::Result<ffi::BufferR1<S32>> status_out,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  // Stage 1: validate the streams and fixed diagnostic-buffer contract.
  if (stream == nullptr || cuda_stream == nullptr) {
    return absl::InvalidArgumentError(
        "cusolvermp_getrf requires XLA and CUDA streams");
  }
  if (status_out->dimensions().size() != 1 ||
      status_out->dimensions()[0] != kGetrfStatusSize) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "cusolvermp_getrf expects status shape (%d,)", kGetrfStatusSize));
  }

  std::array<int32_t, kGetrfStatusSize> status_words = {
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
      static_cast<int32_t>(lu_out->size_bytes()),
      static_cast<int32_t>(n),
      static_cast<int32_t>(tile_size),
      static_cast<int32_t>(lu_out->dimensions()[0]),
      static_cast<int32_t>(lu_out->dimensions()[1]),
      -1,  // A NUMROC rows.
      -1,  // A NUMROC cols.
      -1,  // GETRF device workspace, KiB.
      -1,  // GETRF host workspace, KiB.
      0,   // GETRF called.
      -1,  // GETRF info.
      -1,  // local pivot vector length.
      -1,  // local pivot vector allocation, KiB.
      0,   // pivots gathered and replicated.
      -1,  // dtype code.
      static_cast<int32_t>(grid_mapping),
  };
  int32_t* piv = piv_out->typed_data();
  const size_t piv_bytes = static_cast<size_t>(n) * sizeof(int32_t);

  // Publishes this rank's status. A failed factorization leaves stale or
  // partial data in the results, so LU is replaced with NaN and the integer
  // pivots with -1 first.
  auto publish_status = [&]() -> absl::Status {
    JAXMG_RETURN_IF_ERROR(InvalidateResultsOnFailure(
        cuda_stream, status_words[0], {*lu_out}));
    if (status_words[0] != kStatusOk) {
      JAXMG_RETURN_IF_CUDA_ERROR(
          cudaMemsetAsync(piv, 0xFF, piv_bytes, cuda_stream));
    }
    return CopyStatusToDevice(stream, status_words, status_out);
  };

  // Stage 2: bind the matrix's device, borrow XLA's communicator, and create
  // the matching cuSOLVERMp handle/grid. Matrix contracts stay routine-local.
  if (n <= 0 || tile_size <= 0) {
    status_words[0] = kGridShapeMismatch;
    return publish_status();
  }
  CusolverMpExecution execution(absl::MakeSpan(status_words));
  JAXMG_RETURN_IF_ERROR(execution.Initialize(
      "cusolvermp_getrf", cuda_stream, lu_out->untyped_data(),
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

  const auto [process_row, process_col] =
      ProcessCoordFromRank(nccl_rank, process_rows, process_cols, grid_mapping);
  const int64_t numroc_rows = LocalNumroc(
      n, tile_size, process_row, static_cast<int32_t>(process_rows));
  const int64_t numroc_cols = LocalNumroc(
      n, tile_size, process_col, static_cast<int32_t>(process_cols));
  status_words[17] = static_cast<int32_t>(numroc_rows);
  status_words[18] = static_cast<int32_t>(numroc_cols);

  // Share the pivot capacity policy with LU solve. Local failures only set
  // the status so every rank reaches the shared status agreement below.
  const int64_t ipiv_len =
      LuPivotAllocationLength(numroc_cols, process_row);
  const size_t ipiv_bytes = static_cast<size_t>(ipiv_len) * sizeof(int64_t);
  status_words[23] = static_cast<int32_t>(ipiv_len);
  status_words[24] = SizeToKiBForStatus(ipiv_bytes);
  std::unique_ptr<int64_t, CudaFreeDeleter> d_ipiv;
  if (lu_out->dimensions()[0] < numroc_rows ||
      lu_out->dimensions()[1] < numroc_cols) {
    status_words[0] = kOutputShapeMismatch;
  } else {
    int64_t* raw_ipiv = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&raw_ipiv), ipiv_bytes) ==
        cudaSuccess) {
      d_ipiv.reset(raw_ipiv);
    } else {
      status_words[0] = kDeviceAllocFailed;
    }
  }

  // Stage 3: factor A, then gather the native pivots into the public vector.
  absl::Status getrf_status = absl::OkStatus();
  if (status_words[0] == kStatusOk) {
    switch (lu_out->element_type()) {
      case F32:
        status_words[26] = 1;
        getrf_status = RunCusolverMpGetrf<float>(
            api, handle, grid, cuda_stream, n, tile_size, process_row,
            process_col, lu_out, d_ipiv.get(), ipiv_bytes, &status_words);
        break;
      case F64:
        status_words[26] = 2;
        getrf_status = RunCusolverMpGetrf<double>(
            api, handle, grid, cuda_stream, n, tile_size, process_row,
            process_col, lu_out, d_ipiv.get(), ipiv_bytes, &status_words);
        break;
      case C64:
        status_words[26] = 3;
        getrf_status = RunCusolverMpGetrf<cuFloatComplex>(
            api, handle, grid, cuda_stream, n, tile_size, process_row,
            process_col, lu_out, d_ipiv.get(), ipiv_bytes, &status_words);
        break;
      case C128:
        status_words[26] = 4;
        getrf_status = RunCusolverMpGetrf<cuDoubleComplex>(
            api, handle, grid, cuda_stream, n, tile_size, process_row,
            process_col, lu_out, d_ipiv.get(), ipiv_bytes, &status_words);
        break;
      default:
        status_words[0] = kUnsupportedDtype;
        break;
    }
  }
  if (!getrf_status.ok()) {
    return getrf_status;
  }
  if (d_ipiv != nullptr) {
    DebugDumpLocalPivots(cuda_stream, nccl_rank, d_ipiv.get(), ipiv_len,
                         process_row, process_col, numroc_rows, numroc_cols);
  }

  // Every rank must agree on success before entering the pivot all-reduce.
  JAXMG_RETURN_IF_ERROR(SynchronizeSolverStatus(
      cuda_stream, nccl_comm, status_out, &status_words[0]));
  if (status_words[0] == kStatusOk) {
    // Only process row zero owns pivots. All other ranks contribute -1;
    // an elementwise max replicates the gathered vector across the grid.
    // A rank whose gather failed still joins the all-reduce so no peer hangs;
    // the final status agreement reports its failure.
    cudaError_t cuda_status =
        cudaMemsetAsync(piv, 0xFF, piv_bytes, cuda_stream);
    if (cuda_status == cudaSuccess && process_row == 0) {
      cuda_status = GatherDistributedPivots(
          cuda_stream, d_ipiv.get(), piv, numroc_cols, n, tile_size,
          process_cols, process_col, kGetrfPivotIndexBase);
    }
    if (cuda_status != cudaSuccess) status_words[0] = kPivotGatherFailed;
    JAXMG_RETURN_IF_NCCL_ERROR(ncclAllReduce(piv, piv, n, ncclInt32, ncclMax,
                                             nccl_comm, cuda_stream));
    JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));
    if (status_words[0] == kStatusOk) status_words[25] = 1;
  }

  // Stage 4: include cleanup failures in the shared output-invalidating status.
  execution.Close();
  JAXMG_RETURN_IF_ERROR(SynchronizeSolverStatus(
      cuda_stream, nccl_comm, status_out, &status_words[0]));
  return publish_status();
}

absl::Status RunCusolverMpGetrfDispatch(
    se::Stream* stream, cudaStream_t cuda_stream,
    int64_t process_rows, int64_t process_cols, int64_t n, int64_t tile_size,
    absl::Span<const int64_t> partition_slots, ffi::AnyBuffer a,
    ffi::Result<ffi::AnyBuffer> lu, ffi::Result<ffi::BufferR1<S32>> piv,
    ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  if (piv->dimensions().size() != 1 || piv->dimensions()[0] != n) {
    return absl::InvalidArgumentError("cusolvermp_getrf expects pivot shape (n,)");
  }
  return DispatchSquareMatrix(
      "cusolvermp_getrf", stream, cuda_stream, process_rows, process_cols,
      n, tile_size, partition_slots, a, lu, collective_params,
      collective_cliques,
      [&](const ResolvedProcessGrid& grid, const Padded2DRedistScratch&) {
        return RunCusolverMpGetrfSolver(
            stream, cuda_stream, process_rows, process_cols, n, tile_size,
            grid.grid_mapping, grid.rank_map, lu, piv, status,
            collective_params, collective_cliques);
      });
}

}  // namespace

// Executes the complete factor-only GETRF workflow in one FFI dispatch. The
// packed LU output aliases A, and the pivots are replicated on every rank.
absl::Status XlaCusolverMpGetrfDispatch(
    se::Stream* stream, cudaStream_t cuda_stream, int64_t process_rows,
    int64_t process_cols, int64_t n, int64_t tile_size,
    absl::Span<const int64_t> partition_slots, ffi::AnyBuffer a,
    ffi::Result<ffi::AnyBuffer> lu, ffi::Result<ffi::BufferR1<S32>> piv,
    ffi::Result<ffi::BufferR1<S32>> status,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques) {
  return RunCusolverMpGetrfDispatch(
      stream, cuda_stream, process_rows, process_cols,
      n, tile_size, partition_slots, a, lu, piv, status, collective_params,
      collective_cliques);
}

}  // namespace xla::gpu
