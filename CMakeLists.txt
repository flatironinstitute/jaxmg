cmake_minimum_required(VERSION 3.26)
project(jaxmg LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_CUDA_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

find_package(Python REQUIRED COMPONENTS Interpreter)
find_package(CUDAToolkit REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import jax.ffi; print(jax.ffi.include_dir())"
  OUTPUT_VARIABLE JAX_FFI_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
if(NOT EXISTS "${JAX_FFI_INCLUDE_DIR}/xla/ffi/api/collectives_api.h")
  message(FATAL_ERROR "JAX >= 0.11.2 is required for the Collectives FFI headers")
endif()
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
    "import site; print(';'.join(site.getsitepackages() + [site.getusersitepackages()]))"
  OUTPUT_VARIABLE PYTHON_SITES OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
string(REGEX MATCH "^[0-9]+" CUDA_MAJOR "${CUDAToolkit_VERSION}")
find_path(CUSOLVERMP_INCLUDE_DIR cusolverMp.h HINTS ${PYTHON_SITES}
  PATH_SUFFIXES "nvidia/cu${CUDA_MAJOR}/include" REQUIRED)
find_library(CUSOLVERMP_LIBRARY NAMES cusolverMp libcusolverMp.so.0
  HINTS ${PYTHON_SITES} PATH_SUFFIXES "nvidia/cu${CUDA_MAJOR}/lib" REQUIRED)
find_path(NCCL_INCLUDE_DIR nccl.h HINTS ${PYTHON_SITES}
  PATH_SUFFIXES nvidia/nccl/include "nvidia/cu${CUDA_MAJOR}/include" REQUIRED)
find_library(NCCL_LIBRARY NAMES nccl libnccl.so.2 HINTS ${PYTHON_SITES}
  PATH_SUFFIXES nvidia/nccl/lib "nvidia/cu${CUDA_MAJOR}/lib" REQUIRED)

# Abseil is private implementation code, never part of the FFI boundary.
include(FetchContent)
set(ABSL_PROPAGATE_CXX_STD ON)
set(ABSL_ENABLE_INSTALL OFF)
set(BUILD_TESTING OFF)
FetchContent_Declare(abseil
  GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
  GIT_TAG d9e4955c65cd4367dd6bf46f4ccb8cd3d100540b) # 20250127.1
FetchContent_MakeAvailable(abseil)

set(NATIVE src/cuda)
set(KERNELS
  ${NATIVE}/memory_redist/layout_convert.cu.cc
  ${NATIVE}/cusolvermp_routines/potrs_logdet.cu.cc
  ${NATIVE}/cusolvermp_routines/factor_extract.cu.cc
  ${NATIVE}/cusolvermp_routines/qr_extract.cu.cc
  ${NATIVE}/cusolvermp_routines/result_fill.cu.cc)
set_source_files_properties(${KERNELS} PROPERTIES LANGUAGE CUDA)
add_library(jaxmg_xla_comm_backend SHARED ${KERNELS}
  ${NATIVE}/xla_ffi/runtime.cc
  ${NATIVE}/xla_ffi/handlers.cc
  ${NATIVE}/memory_redist/block_cyclic_2d.cc
  ${NATIVE}/memory_redist/edge_padding_2d.cc
  ${NATIVE}/memory_redist/rectangle_pack.cc
  ${NATIVE}/memory_redist/scratch.cc
  ${NATIVE}/cusolvermp_routines/cusolvermp_cache.cc
  ${NATIVE}/cusolvermp_routines/cusolvermp_common.cc
  ${NATIVE}/cusolvermp_routines/factorization.cc
  ${NATIVE}/cusolvermp_routines/cusolvermp_inverse.cc
  ${NATIVE}/cusolvermp_routines/cusolvermp_potrs.cc
  ${NATIVE}/cusolvermp_routines/cusolvermp_potrf.cc
  ${NATIVE}/cusolvermp_routines/cusolvermp_lu_solve.cc
  ${NATIVE}/cusolvermp_routines/cusolvermp_getrf.cc
  ${NATIVE}/cusolvermp_routines/cusolvermp_gels.cc
  ${NATIVE}/cusolvermp_routines/cusolvermp_qr.cc
  ${NATIVE}/cusolvermp_routines/cusolvermp_syevd.cc
  ${NATIVE}/cusolvermp_routines/cusolvermp_gesvd.cc
  ${NATIVE}/cusolvermp_routines/cusolvermp_polar.cc)
target_include_directories(jaxmg_xla_comm_backend PRIVATE
  ${JAX_FFI_INCLUDE_DIR} ${CUSOLVERMP_INCLUDE_DIR} ${NCCL_INCLUDE_DIR})
target_link_libraries(jaxmg_xla_comm_backend PRIVATE
  CUDA::cudart ${CUSOLVERMP_LIBRARY} ${NCCL_LIBRARY}
  absl::statusor absl::str_format absl::cleanup absl::span absl::function_ref)
# Export only the C handlers, including when private dependencies are static.
set_source_files_properties(${NATIVE}/xla_ffi/handlers.cc PROPERTIES
  COMPILE_OPTIONS "-fvisibility=default")
target_link_options(jaxmg_xla_comm_backend PRIVATE
  "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/${NATIVE}/xla_ffi/exports.map"
  "LINKER:--no-undefined")
set_target_properties(jaxmg_xla_comm_backend PROPERTIES
  CUDA_RUNTIME_LIBRARY Shared
  BUILD_WITH_INSTALL_RPATH ON
  INSTALL_RPATH "$ORIGIN/../../nvidia/cu${CUDA_MAJOR}/lib;$ORIGIN/../../nvidia/nccl/lib;$ORIGIN/../../nvidia/cuda_runtime/lib")
install(TARGETS jaxmg_xla_comm_backend LIBRARY DESTINATION "src/jaxmg/cu${CUDA_MAJOR}")

option(JAXMG_RUN_NATIVE_TESTS "Run cache and C FFI contract tests without a GPU" OFF)
if(JAXMG_RUN_NATIVE_TESTS)
  enable_testing()
  add_executable(cache_lifetime_test tests/native/cache_lifetime_test.cc
    ${NATIVE}/cusolvermp_routines/cusolvermp_cache.cc)
  target_include_directories(cache_lifetime_test PRIVATE ${NATIVE}
    ${CUDAToolkit_INCLUDE_DIRS} ${CUSOLVERMP_INCLUDE_DIR} ${NCCL_INCLUDE_DIR})
  target_link_libraries(cache_lifetime_test PRIVATE absl::status)
  target_compile_options(cache_lifetime_test PRIVATE -UNDEBUG)
  add_test(NAME cache_lifetime COMMAND cache_lifetime_test)

  add_executable(collectives_ffi_test tests/native/collectives_ffi_test.cc
    ${NATIVE}/xla_ffi/runtime.cc)
  target_include_directories(collectives_ffi_test PRIVATE ${NATIVE}
    ${CUDAToolkit_INCLUDE_DIRS} ${JAX_FFI_INCLUDE_DIR} ${NCCL_INCLUDE_DIR})
  target_link_libraries(collectives_ffi_test PRIVATE absl::statusor absl::str_format)
  target_compile_options(collectives_ffi_test PRIVATE -UNDEBUG)
  add_test(NAME collectives_ffi COMMAND collectives_ffi_test)
endif()
