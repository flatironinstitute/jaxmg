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
// Fused Cholesky/LU inversion. The input is A, not an existing factor.
//
// File workflow:
//   1. Reuse the square-matrix layout pipeline and shared factorization.
//   2. Overwrite the factor with POTRI or GETRI; no identity RHS is formed.
//   3. Complete POTRI's Hermitian upper triangle using one tile of scratch.
//   4. Restore the inverse to JAX and report failures consistently on all ranks.

#include <memory>
#include <type_traits>

#include "cusolvermp_common.h"
#include "cusolvermp_routines.h"
#include "factor_extract.h"
#include "factorization.h"

namespace jaxmg {
namespace {

// POTRI leaves the opposite triangle untouched. Transfer tiles in global
// order so rectangular process grids need neither a full transpose buffer
// nor a gather of the inverse onto one device.
absl::Status CompleteHermitianInverse(
    cudaStream_t stream, ncclComm_t comm, int rank, cudaDataType_t dtype,
    int64_t n, int64_t tile, int64_t pr, int64_t pc,
    absl::Span<const int64_t> rank_map, ffi::AnyBuffer a,
    const Padded2DRedistScratch& scratch) {
  if (scratch.elements < tile * tile) {
    return absl::InternalError("Hermitian completion requires one tile of scratch");
  }
  const size_t bytes = a.size_bytes() / a.element_count();
  const int64_t ld = a.dimensions()[0];
  char* data = static_cast<char*>(a.untyped_data());
  const int64_t tiles = (n + tile - 1) / tile;
  for (int64_t j = 0; j < tiles; ++j) {
    for (int64_t i = j; i < tiles; ++i) {
      const int src = rank_map[(i % pr) * pc + j % pc];
      const int dst = rank_map[(j % pr) * pc + i % pc];
      const int64_t rows = std::min(tile, n - i * tile);
      const int64_t cols = std::min(tile, n - j * tile);
      if (rank == src) {
        const int64_t offset = (i / pr) * tile + (j / pc) * tile * ld;
        JAXMG_RETURN_IF_CUDA_ERROR(PackHermitianTile(
            stream, dtype, data + offset * bytes, ld, rows, cols, i == j,
            scratch.pointer));
      }
      if (src != dst) {
        if (rank == src) {
          JAXMG_RETURN_IF_NCCL_ERROR(ncclSend(
              scratch.pointer, rows * cols * bytes, ncclUint8, dst, comm, stream));
        } else if (rank == dst) {
          JAXMG_RETURN_IF_NCCL_ERROR(ncclRecv(
              scratch.pointer, rows * cols * bytes, ncclUint8, src, comm, stream));
        }
      }
      if (rank == dst) {
        const int64_t offset = (j / pr) * tile + (i / pc) * tile * ld;
        JAXMG_RETURN_IF_CUDA_ERROR(cudaMemcpy2DAsync(
            data + offset * bytes, ld * bytes, scratch.pointer, cols * bytes,
            cols * bytes, rows, cudaMemcpyDeviceToDevice, stream));
      }
    }
  }
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(stream));
  return absl::OkStatus();
}

absl::Status RunInverse(
    cudaStream_t cuda_stream, bool cholesky,
    int64_t pr, int64_t pc, int64_t n, int64_t tile,
    const ResolvedProcessGrid& process_grid,
    const Padded2DRedistScratch& scratch, ffi::Result<ffi::AnyBuffer> inverse,
    ffi::Result<ffi::BufferR1<ffi::S32>> status_out,
    CollectiveContext collectives) {
  std::array<int32_t, kInverseStatusSize> status = {
      kStatusOk,
      -1,  // CUDA device.
      -1,  // NCCL rank.
      -1,  // NCCL rank count.
      static_cast<int32_t>(pr),
      static_cast<int32_t>(pc),
      -1,  // cuSOLVERMp version.
      0,   // Linked runtime available.
      0,   // Handle created.
      0,   // Grid created.
      0,   // A descriptor created.
      0,   // Raw cuSOLVER status.
      static_cast<int32_t>(inverse->size_bytes()),
      static_cast<int32_t>(n),
      static_cast<int32_t>(tile),
      static_cast<int32_t>(inverse->dimensions()[0]),
      static_cast<int32_t>(inverse->dimensions()[1]),
      -1,  // A NUMROC rows.
      -1,  // A NUMROC columns.
      -1,  // Factorization device workspace, KiB.
      -1,  // Factorization host workspace, KiB.
      0,   // Factorization called.
      -1,  // Factorization info.
      -1,  // Inverse device workspace, KiB.
      -1,  // Inverse host workspace, KiB.
      0,   // Inverse called.
      -1,  // Inverse info.
      0,   // Native pivot allocation, KiB; zero for Cholesky.
      -1,  // Dtype code.
      static_cast<int32_t>(process_grid.grid_mapping),
      static_cast<int32_t>(cholesky),
      0,   // Full inverse completed.
  };
  auto publish = [&]() -> absl::Status {
    JAXMG_RETURN_IF_ERROR(InvalidateResultsOnFailure(
        cuda_stream, status[0], {*inverse}));
    return CopyStatusToDevice(cuda_stream, status, status_out);
  };

  // Stage 1: borrow XLA's communicator and create the common execution state.
  CusolverMpExecution execution(absl::MakeSpan(status));
  JAXMG_RETURN_IF_ERROR(execution.Initialize(
      cholesky ? "cusolvermp_potri" : "cusolvermp_getri", cuda_stream,
      inverse->untyped_data(), pr, pc, process_grid.grid_mapping,
      process_grid.rank_map, collectives));
  if (status[0] != kStatusOk) {
    execution.Close();
    return publish();
  }
  const auto& api = execution.api;
  const auto [row, col] = ProcessCoordFromRank(
      execution.nccl_rank, pr, pc, process_grid.grid_mapping);
  status[17] = LocalNumroc(n, tile, row, pr);
  status[18] = LocalNumroc(n, tile, col, pc);
  cudaDataType_t dtype = CUDA_R_32F;
  switch (inverse->element_type()) {
    case ffi::F32: dtype = CUDA_R_32F; status[28] = 1; break;
    case ffi::F64: dtype = CUDA_R_64F; status[28] = 2; break;
    case ffi::C64: dtype = CUDA_C_32F; status[28] = 3; break;
    case ffi::C128: dtype = CUDA_C_64F; status[28] = 4; break;
    default: status[0] = kUnsupportedDtype; break;
  }
  if (inverse->dimensions()[0] < status[17] ||
      inverse->dimensions()[1] < status[18]) status[0] = kOutputShapeMismatch;

  cusolverMpMatrixDescriptor_t desc = nullptr;
  auto destroy_desc = [&](auto* value) { api.destroy_matrix_desc(value); };
  std::unique_ptr<std::remove_pointer_t<cusolverMpMatrixDescriptor_t>,
                  decltype(destroy_desc)> descriptor(nullptr, destroy_desc);
  auto free_pivots = [](int64_t* value) { cudaFree(value); };
  std::unique_ptr<int64_t, decltype(free_pivots)> pivots(nullptr, free_pivots);
  size_t pivot_bytes = 0;
  if (status[0] == kStatusOk) {
    const auto result = api.create_matrix_desc(
        &desc, execution.grid, dtype, n, n, tile, tile, 0, 0,
        inverse->dimensions()[0]);
    descriptor.reset(desc);
    if (result != CUSOLVER_STATUS_SUCCESS || desc == nullptr) {
      status[0] = kCreateMatrixDescFailed;
      status[11] = static_cast<int32_t>(result);
    } else {
      status[10] = 1;
    }
  }
  if (!cholesky && status[0] == kStatusOk) {
    pivot_bytes = LuPivotAllocationLength(status[18], row) * sizeof(int64_t);
    status[27] = SizeToKiBForStatus(pivot_bytes);
    int64_t* ptr = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&ptr), pivot_bytes) != cudaSuccess) {
      status[0] = kDeviceAllocFailed;
    }
    pivots.reset(ptr);
  }
  JAXMG_RETURN_IF_ERROR(SynchronizeSolverStatus(
      cuda_stream, execution.nccl_comm, status_out, &status[0]));

  // Stage 2: factor the original A; retain LU pivots through GETRI.
  if (status[0] == kStatusOk) {
    const FactorizationStatusFields fields{
        status[0], status[11], status[19], status[20], status[21], status[22]};
    if (cholesky) {
      JAXMG_RETURN_IF_ERROR(RunCholeskyFactorization(
          api, execution.handle, desc, cuda_stream, n, inverse->untyped_data(),
          dtype, execution.nccl_comm, status_out, fields));
    } else {
      JAXMG_RETURN_IF_ERROR(RunLuFactorization(
          api, execution.handle, desc, cuda_stream, n, inverse->untyped_data(),
          dtype, pivots.get(), pivot_bytes, execution.nccl_comm, status_out,
          fields));
    }
  }
  JAXMG_RETURN_IF_ERROR(SynchronizeSolverStatus(
      cuda_stream, execution.nccl_comm, status_out, &status[0]));

  // Stage 3: invert the factor using the same matrix allocation and descriptor.
  if (status[0] == kStatusOk) {
    JAXMG_RETURN_IF_ERROR(RunFactorInverse(
        api, execution.handle, desc, cuda_stream, n, inverse->untyped_data(),
        dtype, cholesky, pivots.get(), execution.nccl_comm, status_out,
        {status[0], status[11], status[23], status[24], status[25], status[26]}));
  }
  JAXMG_RETURN_IF_ERROR(SynchronizeSolverStatus(
      cuda_stream, execution.nccl_comm, status_out, &status[0]));

  // Stage 4: POTRI supplies one triangle; GETRI already supplies the full inverse.
  if (status[0] == kStatusOk) {
    if (cholesky) {
      JAXMG_RETURN_IF_ERROR(CompleteHermitianInverse(
          cuda_stream, execution.nccl_comm, execution.nccl_rank, dtype,
          n, tile, pr, pc, process_grid.rank_map, *inverse, scratch));
    }
    status[31] = 1;
  }
  if (descriptor) {
    const auto result = api.destroy_matrix_desc(descriptor.release());
    if (result != CUSOLVER_STATUS_SUCCESS && status[0] == kStatusOk) {
      status[0] = kDestroyMatrixDescFailed;
      status[11] = static_cast<int32_t>(result);
    }
  }
  execution.Close();
  JAXMG_RETURN_IF_ERROR(SynchronizeSolverStatus(
      cuda_stream, execution.nccl_comm, status_out, &status[0]));
  return publish();
}

}  // namespace

absl::Status XlaCusolverMpInverseDispatch(
    cudaStream_t cuda_stream, int64_t process_rows,
    int64_t process_cols, int64_t n, int64_t tile_size, bool cholesky,
    absl::Span<const int64_t> partition_slots, ffi::AnyBuffer a,
    ffi::Result<ffi::AnyBuffer> inverse,
    ffi::Result<ffi::BufferR1<ffi::S32>> status,
    CollectiveContext collectives) {
  if (cuda_stream == nullptr ||
      status->dimensions().size() != 1 ||
      status->dimensions()[0] != kInverseStatusSize) {
    return absl::InvalidArgumentError("inverse requires streams and status shape (32,)");
  }
  return DispatchSquareMatrix(
      cholesky ? "cusolvermp_potri" : "cusolvermp_getri", cuda_stream,
      process_rows, process_cols, n, tile_size, partition_slots, a, inverse,
      collectives,
      [&](const ResolvedProcessGrid& grid, const Padded2DRedistScratch& scratch) {
        return RunInverse(cuda_stream, cholesky, process_rows,
                          process_cols, n, tile_size, grid, scratch, inverse,
                          status, collectives);
      });
}

}  // namespace jaxmg
