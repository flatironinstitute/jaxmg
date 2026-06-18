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
// CUDA launcher for local row-major <-> column-major layout conversion.
//
// The implementation lives in layout_convert.cu.cc and is linked directly into
// libjaxmg_xla_comm_backend.so by Bazel.  Keeping this declaration in a small
// C-compatible header lets ordinary C++ redistribution code call the launcher
// without loading a second helper shared library at runtime.

#ifndef JAXMG_MEMORY_REDIST_LAYOUT_CONVERT_H_
#define JAXMG_MEMORY_REDIST_LAYOUT_CONVERT_H_

#include <cstdint>

#include <cuda_runtime_api.h>

extern "C" cudaError_t JaxmgLaunchRowMajorToColumnMajorDecomposition(
    cudaStream_t stream, void* data, void* scratch, std::int64_t rows,
    std::int64_t cols, std::int64_t scratch_elements,
    std::int64_t element_bytes);

#endif  // JAXMG_MEMORY_REDIST_LAYOUT_CONVERT_H_
