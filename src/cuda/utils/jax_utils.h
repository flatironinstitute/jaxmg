#ifndef JAXLIB_GPU_UTILS_H_
#define JAXLIB_GPU_UTILS_H_

// C++
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string_view>
// Abseil
#include "absl/status/statusor.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
// JAXlib
#include "jaxlib/ffi_helpers.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/gpu/handle_pool.h"
// CUDA
#include "third_party/gpus/cuda/include/cusolverMg.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
// XLA
#include "xla/ffi/api/ffi.h"

namespace jax
{

  using SolverHandlePool = HandlePool<cusolverMgHandle_t, gpuStream_t>;

  template <>
  absl::StatusOr<SolverHandlePool::Handle> SolverHandlePool::Borrow(
      gpuStream_t stream);

  template <typename T>
  inline absl::StatusOr<T *> AllocateWorkspaceBytes(
      ::xla::ffi::ScratchAllocator &scratch, int64_t n_bytes,
      std::string_view name)
  {
    auto maybe_workspace = scratch.Allocate(n_bytes);
    if (!maybe_workspace.has_value())
    {
      return absl::Status(
          absl::StatusCode::kResourceExhausted,
          absl::StrFormat("Unable to allocate workspace for %s", name));
    }
    return static_cast<T *>(maybe_workspace.value());
  }
} // namespace jax

inline absl::Status CudaToStatus(cudaError_t err, const char *file, int line)
{
  if (err == cudaSuccess)
    return absl::OkStatus();
  return absl::InternalError(
      absl::StrFormat("CUDA error %d (%s) at %s:%d",
                      static_cast<int>(err),
                      cudaGetErrorString(err),
                      file, line));
}

inline absl::Status CusolverToStatus(cusolverStatus_t err, const char *file, int line)
{
  if (err == CUSOLVER_STATUS_SUCCESS)
    return absl::OkStatus();
  return absl::InternalError(
      absl::StrFormat("cuSolver error %d at %s:%d",
                      static_cast<int>(err), file, line));
}

#define CUDA_CHECK_STATUS(expr) CudaToStatus((expr), __FILE__, __LINE__)
#define CUSOLVER_CHECK_STATUS(expr) CusolverToStatus((expr), __FILE__, __LINE__)

// Compose CUDA_CHECK with FFI return.
#define CUDA_CHECK_OR_RETURN(...)        \
  do                                     \
  {                                      \
    FFI_RETURN_IF_ERROR_STATUS(          \
        CUDA_CHECK_STATUS(__VA_ARGS__)); \
  } while (0)

// Compose CUSOLVER_CHECK with FFI return.
#define CUSOLVER_CHECK_OR_RETURN(...)        \
  do                                         \
  {                                          \
    FFI_RETURN_IF_ERROR_STATUS(              \
        CUSOLVER_CHECK_STATUS(__VA_ARGS__)); \
  } while (0)

// CUDA API error checking
#define CUDA_CHECK(err)                                             \
  do                                                                \
  {                                                                 \
    cudaError_t err_ = (err);                                       \
    if (err_ != cudaSuccess)                                        \
    {                                                               \
      printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("CUDA error");                       \
    }                                                               \
  } while (0)

// cusolver API error checking
#define CUSOLVER_CHECK(err)                                             \
  do                                                                    \
  {                                                                     \
    cusolverStatus_t err_ = (err);                                      \
    if (err_ != CUSOLVER_STATUS_SUCCESS)                                \
    {                                                                   \
      printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("cusolver error");                       \
    }                                                                   \
  } while (0)

#define JAX_FFI_RETURN_IF_GPU_ERROR(...) \
  FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(__VA_ARGS__))


// Allocate per-device workspace on the listed devices. These have external linkage so
// they can be called from multiple translation units (do NOT mark `static`).
void workspaceAlloc(int num_devices, const int *deviceIdA, /* <int> dimension num_devices */
                    size_t sizeInBytes,                    /* number of bytes per device */
                    void **array_d_work                    /* <t> num_devices, host array */
                                                           /* array_d_work[j] points to device workspace of device j */
);

void workspaceFree(int num_devices, const int *deviceIdA, /* <int> dimension num_devices */
                   void **array_d_work                    /* <t> num_devices, host array */
                                                          /* array_d_work[j] points to device workspace of device j */
);

#endif // JAXLIB_GPU_UTILS_H_