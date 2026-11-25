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
// Shared FFI status conversion and borrowed NCCL communicator access.
//
// File workflow:
//   1. Request the ordered SPMD communicator during FFI preparation.
//   2. Borrow its NCCL handle through the public C extension at execution.
//   3. Run communication on the supplied CUDA stream, alongside solver work.

#include "runtime.h"

#include <limits>
#include <numeric>

namespace jaxmg {
namespace {

std::vector<std::vector<int64_t>> PartitionGroup(int64_t num_ranks) {
  std::vector<std::vector<int64_t>> groups(1, std::vector<int64_t>(num_ranks));
  std::iota(groups[0].begin(), groups[0].end(), 0);
  return groups;
}

}  // namespace

absl::Status CollectiveErrorPolicy::TakeError(const XLA_FFI_Api* api,
                                              XLA_FFI_Error* error) {
  const auto result = ffi::internal::ErrorPolicy::TakeError(api, error);
  return absl::Status(static_cast<absl::StatusCode>(*result.errc()),
                      result.message());
}

absl::Status CudaToStatus(cudaError_t err, const char* file, int line) {
  if (err == cudaSuccess) return absl::OkStatus();
  return absl::InternalError(
      absl::StrFormat("CUDA error %d (%s) at %s:%d", static_cast<int>(err),
                      cudaGetErrorString(err), file, line));
}

absl::Status NcclToStatus(ncclResult_t err, const char* file, int line) {
  if (err == ncclSuccess) return absl::OkStatus();
  return absl::InternalError(
      absl::StrFormat("NCCL error %d (%s) at %s:%d", static_cast<int>(err),
                      ncclGetErrorString(err), file, line));
}

absl::Status XlaCusolverMpPrepare(ffi::Dictionary attributes,
                                  CollectiveContext collectives) {
  // Each solver adds different attributes; preparation needs only the grid.
  auto rows = attributes.get<int64_t>("process_rows");
  auto cols = attributes.get<int64_t>("process_cols");
  if (!rows || !cols) {
    return absl::InvalidArgumentError(
        "cuSOLVERMp requires process_rows and process_cols");
  }
  const int64_t process_rows = *rows, process_cols = *cols;
  if (process_rows <= 0 || process_cols <= 0 ||
      process_rows > std::numeric_limits<int>::max() / process_cols) {
    return absl::InvalidArgumentError(
        "cuSOLVERMp requires a positive process grid");
  }
  // Communication ID 1 keeps these P2P transfers separate from XLA collectives.
  return collectives.RequestCommunicator(
      ffi::GroupMode::kFlattenedId, PartitionGroup(process_rows * process_cols),
      1);
}

absl::StatusOr<ncclComm_t> BorrowNcclComm(const char* caller,
                                          CollectiveContext collectives,
                                          int64_t num_ranks) {
  if (num_ranks <= 0 || num_ranks > std::numeric_limits<int>::max()) {
    return absl::InvalidArgumentError("invalid cuSOLVERMp communicator size");
  }
  auto borrowed = collectives.GetCommunicator(ffi::GroupMode::kFlattenedId,
                                              PartitionGroup(num_ranks), 1);
  if (!borrowed.ok()) return borrowed.status();
  if (*borrowed == nullptr) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "%s requires an XLA communicator with a platform NCCL handle", caller));
  }
  return reinterpret_cast<ncclComm_t>(*borrowed);
}

absl::Status ValidateBorrowedNcclComm(const char* caller, ncclComm_t comm,
                                      int64_t expected_rank,
                                      int64_t expected_count) {
  int rank = -1, count = -1;
  JAXMG_RETURN_IF_NCCL_ERROR(ncclCommUserRank(comm, &rank));
  JAXMG_RETURN_IF_NCCL_ERROR(ncclCommCount(comm, &count));
  if (rank != expected_rank || count != expected_count) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "%s borrowed NCCL rank/count=(%d, %d), expected (%d, %d)", caller, rank,
        count, expected_rank, expected_count));
  }
  return absl::OkStatus();
}

absl::Status RunRawNcclAllReduceReal(const char* caller,
                                     cudaStream_t cuda_stream, ncclComm_t comm,
                                     ncclDataType_t dtype, void* value) {
  if (comm == nullptr || value == nullptr ||
      (dtype != ncclFloat && dtype != ncclDouble)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s requires an NCCL communicator and a real device scalar", caller));
  }
  JAXMG_RETURN_IF_NCCL_ERROR(
      ncclAllReduce(value, value, 1, dtype, ncclSum, comm, cuda_stream));
  return absl::OkStatus();
}

}  // namespace jaxmg
