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

#ifndef JAXMG_FACTOR_EXTRACT_H_
#define JAXMG_FACTOR_EXTRACT_H_

#include <cstdint>

#include <cuda_runtime_api.h>
#include <library_types.h>

namespace xla::gpu {

// Zeros the strict upper triangle of a distributed square matrix in place.
// The buffer holds one rank's local block in 2D block-cyclic, column-major
// storage, so POTRF's lower factor becomes L without a reverse redistribution.
cudaError_t ZeroDistributedStrictUpper(
    cudaStream_t cuda_stream, cudaDataType_t dtype, void* a, int64_t tile_size,
    int64_t process_rows, int64_t process_cols, int32_t process_row,
    int32_t process_col, int64_t leading_dimension, int64_t numroc_rows,
    int64_t numroc_cols);

// Scatters one rank's block-cyclically distributed GETRF pivots into a global
// length-n vector. Local entry j belongs to global index
// ((j / tile_size) * process_count + process) * tile_size + j % tile_size along
// the grid dimension the pivots are tied to. Values are shifted by
// `index_base` to 0-based row indices; entries below `index_base` were never
// written by cuSOLVERMp and are skipped.
cudaError_t GatherDistributedPivots(cudaStream_t cuda_stream,
                                    const int64_t* local_ipiv, int32_t* piv,
                                    int64_t local_len, int64_t n,
                                    int64_t tile_size, int64_t process_count,
                                    int32_t process, int64_t index_base);

}  // namespace xla::gpu

#endif  // JAXMG_FACTOR_EXTRACT_H_
