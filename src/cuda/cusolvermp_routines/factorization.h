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

#ifndef JAXMG_FACTORIZATION_H_
#define JAXMG_FACTORIZATION_H_

#include <algorithm>

#include "cusolvermp_common.h"

namespace xla::gpu {

// References into a caller's existing diagnostic vector. Factor-only and
// solve interfaces retain their own field ordering and status-buffer sizes.
struct FactorizationStatusFields {
  int32_t& code;
  int32_t& raw_status;
  int32_t& device_workspace_kib;
  int32_t& host_workspace_kib;
  int32_t& called;
  int32_t& info;
};

// Overwrites A with its lower Cholesky factor in native block-cyclic storage.
// Owns only the factorization workspace and info buffer; the caller retains
// the handle, descriptor, and matrix for POTRS or factor extraction. Solver
// failures populate the status fields; CUDA execution errors return a status.
absl::Status RunCholeskyFactorization(
    const CusolverMpApi& api, cusolverMpHandle_t handle,
    cusolverMpMatrixDescriptor_t desc_a, cudaStream_t cuda_stream, int64_t n,
    void* a, cudaDataType_t dtype, FactorizationStatusFields status);

// Overwrites A with packed LU and writes the native 1-based pivot indices.
// The caller owns the pivot buffer through GETRS or pivot gathering; temporary
// factorization workspace is released before this function returns.
absl::Status RunLuFactorization(
    const CusolverMpApi& api, cusolverMpHandle_t handle,
    cusolverMpMatrixDescriptor_t desc_a, cudaStream_t cuda_stream, int64_t n,
    void* a, cudaDataType_t dtype, int64_t* ipiv, size_t ipiv_bytes,
    FactorizationStatusFields status);

// Pivots are distributed over process row zero. Other rows must still pass a
// non-null pointer so every rank selects pivoted factorization and solves.
inline int64_t LuPivotAllocationLength(int64_t local_cols, int32_t process_row) {
  return process_row == 0 ? std::max<int64_t>(local_cols, 1) : 1;
}

}  // namespace xla::gpu

#endif  // JAXMG_FACTORIZATION_H_
