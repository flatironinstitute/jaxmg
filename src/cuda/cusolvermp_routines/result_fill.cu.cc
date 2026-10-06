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
// Device-side constant fill for solver result buffers.
//
// The fill ignores buffer layout: every element of the local allocation,
// padding included, receives the same value. Python discards padding after
// the FFI call, so a whole-buffer fill is correct for any local layout.

#include <algorithm>

#include "result_fill.h"

namespace xla::gpu {
namespace {

constexpr int kFillBlockSize = 256;
constexpr int64_t kMaxFillBlocks = 4096;

// Grid-stride loop so a capped launch covers buffers of any size.
template <typename DataType>
__global__ void FillDeviceArrayKernel(DataType* data, int64_t count,
                                      DataType value) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += stride) {
    data[index] = value;
  }
}

}  // namespace

template <typename DataType>
cudaError_t FillDeviceArray(cudaStream_t cuda_stream, DataType* data,
                            int64_t count, DataType value) {
  if (count == 0) return cudaSuccess;
  if (cuda_stream == nullptr || data == nullptr || count < 0) {
    return cudaErrorInvalidValue;
  }
  const int block_count = static_cast<int>(std::min<int64_t>(
      kMaxFillBlocks, (count + kFillBlockSize - 1) / kFillBlockSize));
  FillDeviceArrayKernel<DataType>
      <<<block_count, kFillBlockSize, 0, cuda_stream>>>(data, count, value);
  return cudaGetLastError();
}

template cudaError_t FillDeviceArray<float>(cudaStream_t, float*, int64_t,
                                            float);
template cudaError_t FillDeviceArray<double>(cudaStream_t, double*, int64_t,
                                             double);
template cudaError_t FillDeviceArray<cuFloatComplex>(cudaStream_t,
                                                     cuFloatComplex*, int64_t,
                                                     cuFloatComplex);
template cudaError_t FillDeviceArray<cuDoubleComplex>(cudaStream_t,
                                                      cuDoubleComplex*,
                                                      int64_t,
                                                      cuDoubleComplex);

}  // namespace xla::gpu
