// Inject cleanup/readback failures without changing the production backend.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

static int info_copies;
static int injected_copy_failure;
static int destroyed_descriptors;

int jaxmg_test_copy_failure_injected(void) { return injected_copy_failure; }
int jaxmg_test_destroyed_descriptors(void) { return destroyed_descriptors; }

int cusolverMpDestroyMatrixDesc(void* descriptor) {
  typedef int (*DestroyFn)(void*);
  DestroyFn destroy = (DestroyFn)dlsym(RTLD_NEXT, "cusolverMpDestroyMatrixDesc");
  if (destroy == NULL) abort();
  int status = destroy(descriptor);
  Dl_info caller;
  if (status == 0 && dladdr(__builtin_return_address(0), &caller) != 0 &&
      caller.dli_fname != NULL && strstr(caller.dli_fname, "libjaxmg") != NULL)
    ++destroyed_descriptors;
  return status;
}

int cudaMemcpyAsync(void* dst, const void* src, size_t bytes, int kind,
                    void* stream) {
  typedef int (*CopyFn)(void*, const void*, size_t, int, void*);
  CopyFn copy = (CopyFn)dlsym(RTLD_NEXT, "cudaMemcpyAsync");
  if (copy == NULL) abort();
  const char* fail_after = getenv("JAXMG_TEST_FAIL_INFO_COPY");
  Dl_info caller;
  // Target only JAXMg's scalar info readback, not JAX or vendor-library copies.
  if (fail_after != NULL && kind == 2 && bytes == sizeof(int) &&
      dladdr(__builtin_return_address(0), &caller) != 0 &&
      caller.dli_fname != NULL && strstr(caller.dli_fname, "libjaxmg") != NULL &&
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
  const char* fail = getenv("JAXMG_TEST_FAIL_DESTROY");
  // CUSOLVER_STATUS_INTERNAL_ERROR is 7; successful cleanup returns 0.
  return status == 0 && fail != NULL && fail[0] == '1' ? 7 : status;
}
