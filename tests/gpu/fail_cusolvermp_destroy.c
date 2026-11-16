// Inject cleanup/readback failures without changing the production backend.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

static int info_copies;
static int injected_copy_failure;
static int destroyed_descriptors;
static int created_handles;
static int created_grids;
static int destroyed_handles;
static int destroyed_grids;

int jaxmg_test_copy_failure_injected(void) { return injected_copy_failure; }
int jaxmg_test_destroyed_descriptors(void) { return destroyed_descriptors; }
int jaxmg_test_created_handles(void) { return created_handles; }
int jaxmg_test_created_grids(void) { return created_grids; }
int jaxmg_test_destroyed_handles(void) { return destroyed_handles; }
int jaxmg_test_destroyed_grids(void) { return destroyed_grids; }

static int from_jaxmg(void* address) {
  Dl_info caller;
  return dladdr(address, &caller) != 0 && caller.dli_fname != NULL &&
         strstr(caller.dli_fname, "libjaxmg") != NULL;
}

int cusolverMpCreate(void** handle, int device, void* stream) {
  typedef int (*CreateFn)(void**, int, void*);
  CreateFn create = (CreateFn)dlsym(RTLD_NEXT, "cusolverMpCreate");
  if (create == NULL) abort();
  int status = create(handle, device, stream);
  if (status == 0 && from_jaxmg(__builtin_return_address(0))) ++created_handles;
  return status;
}

int cusolverMpCreateDeviceGrid(void* handle, void** grid, void* comm,
                             int rows, int cols, int mapping) {
  typedef int (*CreateFn)(void*, void**, void*, int, int, int);
  CreateFn create = (CreateFn)dlsym(RTLD_NEXT, "cusolverMpCreateDeviceGrid");
  if (create == NULL) abort();
  int status = create(handle, grid, comm, rows, cols, mapping);
  if (status == 0 && from_jaxmg(__builtin_return_address(0))) ++created_grids;
  return status;
}

int cusolverMpDestroyGrid(void* grid) {
  typedef int (*DestroyFn)(void*);
  DestroyFn destroy = (DestroyFn)dlsym(RTLD_NEXT, "cusolverMpDestroyGrid");
  if (destroy == NULL) abort();
  int status = destroy(grid);
  if (status == 0 && from_jaxmg(__builtin_return_address(0))) ++destroyed_grids;
  return status;
}

int cusolverMpDestroyMatrixDesc(void* descriptor) {
  typedef int (*DestroyFn)(void*);
  DestroyFn destroy = (DestroyFn)dlsym(RTLD_NEXT, "cusolverMpDestroyMatrixDesc");
  if (destroy == NULL) abort();
  int status = destroy(descriptor);
  if (status == 0 && from_jaxmg(__builtin_return_address(0))) {
    ++destroyed_descriptors;
    const char* fail = getenv("JAXMG_TEST_FAIL_DESCRIPTOR_DESTROY");
    if (fail != NULL && fail[0] == '1') return 7;
  }
  return status;
}

int cudaMemcpyAsync(void* dst, const void* src, size_t bytes, int kind,
                    void* stream) {
  typedef int (*CopyFn)(void*, const void*, size_t, int, void*);
  CopyFn copy = (CopyFn)dlsym(RTLD_NEXT, "cudaMemcpyAsync");
  if (copy == NULL) abort();
  const char* fail_after = getenv("JAXMG_TEST_FAIL_INFO_COPY");
  // Target only JAXMg's scalar info readback, not JAX or vendor-library copies.
  if (fail_after != NULL && kind == 2 && bytes == sizeof(int) &&
      from_jaxmg(__builtin_return_address(0)) &&
      ++info_copies == atoi(fail_after)) {
    injected_copy_failure = 1;
    return 1;  // cudaErrorInvalidValue, without poisoning the CUDA context.
  }
  return copy(dst, src, bytes, kind, stream);
}

int cusolverMpDestroy(void* handle) {
  typedef int (*DestroyFn)(void*);
  DestroyFn destroy = (DestroyFn)dlsym(RTLD_NEXT, "cusolverMpDestroy");
  if (destroy == NULL) abort();
  int status = destroy(handle);
  if (status == 0 && from_jaxmg(__builtin_return_address(0))) ++destroyed_handles;
  const char* fail = getenv("JAXMG_TEST_FAIL_DESTROY");
  // CUSOLVER_STATUS_INTERNAL_ERROR is 7; successful cleanup returns 0.
  return status == 0 && fail != NULL && fail[0] == '1' ? 7 : status;
}
