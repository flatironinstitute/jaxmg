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
// CUDA entry point for overwriting a solver result buffer with one value.
//
// The fused solver handlers use it to replace every element of a failed rank's
// result buffers with the dtype's NaN, so a failure cannot be mistaken for a
// solution. The fill value is chosen by the caller from SolverTraits, which
// keeps this header free of XLA dependencies.

#ifndef JAXMG_RESULT_FILL_H_
#define JAXMG_RESULT_FILL_H_

#include <cstdint>

#include <cuComplex.h>
#include <cuda_runtime_api.h>

namespace xla::gpu {

// Writes `value` to each of the `count` contiguous elements of `data`.
// Instantiated for float, double, cuFloatComplex, and cuDoubleComplex.
template <typename DataType>
cudaError_t FillDeviceArray(cudaStream_t cuda_stream, DataType* data,
                            int64_t count, DataType value);

}  // namespace xla::gpu

#endif  // JAXMG_RESULT_FILL_H_
