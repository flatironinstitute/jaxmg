// Complete the real cleanup, then report a failure on the selected test rank.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>

int cusolverMpDestroy(void* handle) {
  typedef int (*DestroyFn)(void*);
  DestroyFn destroy = (DestroyFn)dlsym(RTLD_NEXT, "cusolverMpDestroy");
  if (destroy == NULL) abort();
  int status = destroy(handle);
  const char* fail = getenv("JAXMG_TEST_FAIL_DESTROY");
  // CUSOLVER_STATUS_INTERNAL_ERROR is 7; successful cleanup returns 0.
  return status == 0 && fail != NULL && fail[0] == '1' ? 7 : status;
}
