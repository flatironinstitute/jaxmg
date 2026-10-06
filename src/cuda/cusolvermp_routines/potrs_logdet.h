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
// CUDA entry points for extracting a Cholesky log determinant.
//
// cuSOLVERMp overwrites the distributed input matrix with its lower Cholesky
// factor. This header exposes the small CUDA operation needed by the fused
// POTRS handler to accumulate the diagonal entries owned by one rank.
// Cross-rank reduction remains in the XLA/NCCL runtime layer because
// communicator ownership is independent of the solver. A failed solve skips
// the accumulation, and the dispatch fills the scalar with NaN instead.

#ifndef JAXMG_POTRS_LOGDET_H_
#define JAXMG_POTRS_LOGDET_H_

#include <cstdint>

#include <cuda_runtime_api.h>
#include <library_types.h>

namespace xla::gpu {

// Accumulates 2 * log(abs(L_ii)) for diagonal entries owned by this rank.
// `factor` uses cuSOLVERMp's local column-major 2D block-cyclic layout and
// `local_physical_rows` is the descriptor leading dimension.
cudaError_t AccumulateLocalCholeskyLogdet(
    cudaStream_t cuda_stream, cudaDataType_t dtype, const void* factor,
    int64_t n, int64_t tile_size, int64_t process_rows,
    int64_t process_cols, int32_t process_row, int32_t process_col,
    int64_t local_physical_rows, void* logdet_out);

}  // namespace xla::gpu

#endif  // JAXMG_POTRS_LOGDET_H_
