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

#ifndef JAXMG_QR_EXTRACT_H_
#define JAXMG_QR_EXTRACT_H_

#include <cstdint>

#include <cuda_runtime_api.h>
#include <library_types.h>

namespace xla::gpu {

// Copies the upper triangle produced by GEQRF from a distributed M-by-N
// buffer into an N-by-N buffer with the same 2D block-cyclic ownership.
cudaError_t ExtractDistributedQrR(
    cudaStream_t cuda_stream, cudaDataType_t dtype, const void* packed_qr,
    void* r, int64_t n, int64_t tile_size, int64_t process_rows,
    int64_t process_cols, int32_t process_row, int32_t process_col,
    int64_t qr_leading_dimension, int64_t r_leading_dimension,
    int64_t r_numroc_rows, int64_t r_numroc_cols);

}  // namespace xla::gpu

#endif  // JAXMG_QR_EXTRACT_H_
