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
// Diagnostic-only XLA FFI entry point for local layout conversion.
//
// This file deliberately avoids cuSOLVERMp and NCCL. It exists to distinguish
// three failure modes while debugging large SYEVD eigenvector restores:
//
//   1. the standalone CUDA decomposition kernel fails by itself;
//   2. the kernel works alone but fails when launched through XLA-managed
//      donated buffers and XLA scratch;
//   3. the kernel only fails after preceding cuSOLVERMp / redistribution work.
//
// Python does not register this target during normal package initialization.
// Diagnostic scripts load the symbol manually with ctypes and call it through
// jax.ffi.ffi_call.

#include "runtime.h"
#include "../memory_redist/memory_redist.h"

#include <algorithm>

#include "absl/strings/str_format.h"

namespace xla::gpu {

// Runtime diagnostic hook. It copies or aliases a rank-local matrix into the
// output, allocates XLA-owned scratch, and performs either the forward
// row-major -> column-major conversion or the inverse column-major -> row-major
// conversion. The production SYEVD failure currently happens in the inverse
// path, so test scripts normally pass inverse=1.
absl::Status XlaLayoutConvertProbeDispatch(
    se::Stream* stream, cudaStream_t cuda_stream,
    se::OwningScratchAllocator<> scratch, int64_t scratch_elements,
    int64_t inverse, ffi::AnyBuffer matrix,
    ffi::Result<ffi::AnyBuffer> matrix_out) {
  (void)stream;
  if (cuda_stream == nullptr) {
    return absl::InvalidArgumentError(
        "layout_convert_probe requires a CUDA stream");
  }
  if (matrix.dimensions().size() != 2 || matrix_out->dimensions().size() != 2) {
    return absl::InvalidArgumentError(
        "layout_convert_probe expects rank-2 input and output buffers");
  }
  if (matrix.element_type() != matrix_out->element_type() ||
      matrix.dimensions()[0] != matrix_out->dimensions()[0] ||
      matrix.dimensions()[1] != matrix_out->dimensions()[1]) {
    return absl::InvalidArgumentError(
        "layout_convert_probe input and output buffers must have matching "
        "dtype and shape");
  }
  if (matrix.element_count() == 0) {
    return absl::OkStatus();
  }

  const size_t element_bytes =
      matrix.size_bytes() / static_cast<size_t>(matrix.element_count());
  if (element_bytes != 4 && element_bytes != 8 && element_bytes != 16) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "layout_convert_probe supports only 4, 8, and 16 byte payloads; got "
        "%d bytes",
        element_bytes));
  }
  if (scratch_elements < 0) {
    return absl::InvalidArgumentError(
        "layout_convert_probe requires non-negative scratch_elements");
  }

  const size_t scratch_bytes =
      static_cast<size_t>(std::max<int64_t>(1, scratch_elements)) *
      element_bytes;
  absl::StatusOr<void*> scratch_pointer =
      AllocateFfiScratch(scratch, scratch_bytes, "layout_convert_probe");
  if (!scratch_pointer.ok()) {
    return scratch_pointer.status();
  }
  se::DeviceAddressBase scratch_base(*scratch_pointer, scratch_bytes);

  // When the Python ffi_call aliases input 0 to output 0 this copy is a no-op.
  // Keeping the copy makes the probe robust if XLA declines the alias or if a
  // script intentionally tests the non-aliased path.
  JAXMG_RETURN_IF_ERROR(
      CopyMatrixIfNeeded(cuda_stream, matrix, matrix_out));

  if (inverse != 0) {
    JAXMG_RETURN_IF_ERROR(ConvertColumnMajorToRowMajorInPlace(
        cuda_stream, "layout_convert_probe/inverse", *matrix_out,
        scratch_base, scratch_elements));
  } else {
    JAXMG_RETURN_IF_ERROR(ConvertRowMajorToColumnMajorInPlace(
        cuda_stream, "layout_convert_probe/forward", *matrix_out,
        scratch_base, scratch_elements));
  }
  JAXMG_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));
  return absl::OkStatus();
}

// Registers the runtime-only diagnostic target. There is no prepare stage
// because this handler does not request XLA communicator cliques.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XlaLayoutConvertProbeFFI, XlaLayoutConvertProbeDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Attr<int64_t>("scratch_elements")
        .Attr<int64_t>("inverse")
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());

}  // namespace xla::gpu
