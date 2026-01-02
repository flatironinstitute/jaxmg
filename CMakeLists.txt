cmake_minimum_required(VERSION 3.25.2)
project(JaxSolve)

# Get CPM
set(CPM_DOWNLOAD_VERSION 0.42.0)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake")
    file(DOWNLOAD https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake ${CPM_DOWNLOAD_LOCATION})
endif()
include(${CPM_DOWNLOAD_LOCATION})

# Abseil requires C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" ${CMAKE_MODULE_PATH})

# Static linking against C++20 standard library
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static-libgcc -static-libstdc++")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -static-libgcc")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -static-libgcc")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libc++ -static-libgcc")
endif()

# Abseill headers
CPMAddPackage(
    NAME absl
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp
    GIT_TAG 20250512.1
)

# Import CUDAToolkit (Requires CUDA_HOME to be set)
find_package(CUDAToolkit REQUIRED)
find_library(CUSOLVERMG_LIB cusolverMg
    HINTS "${CUDAToolkit_LIBRARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "/usr/local/cuda/lib64"
    REQUIRED)
find_package(CUDNN REQUIRED)
message(STATUS "CUDA include directory: $ENV{CUDA_ROOT}")
message(STATUS "CUDNN include directory: $ENV{CUDNN_ROOT}")
# Enable CUDA 
enable_language(CUDA)
## Set backend install subdirectory based on CUDA major version
string(REGEX MATCH "^([0-9]+)\.([0-9]+)" _cuda_ver_match "${CMAKE_CUDA_COMPILER_VERSION}")
set(CUDA_MAJOR "${CMAKE_MATCH_1}")
set(BIN_DST src/jaxmg/cu${CUDA_MAJOR})
message(STATUS "Library install destination: ${CMAKE_SOURCE_DIR}/${BIN_DST}")

# Ensure CUDA uses C++20 (required by this project)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Get Jaxlib
# Allow overriding the JAX version via -DJAX_VERSION=0.x.y or env JAX_VERSION; default to 0.6.2
if(NOT DEFINED JAX_VERSION OR JAX_VERSION STREQUAL "")
    if(DEFINED ENV{JAX_VERSION} AND NOT "$ENV{JAX_VERSION}" STREQUAL "")
        set(JAX_VERSION "$ENV{JAX_VERSION}")
    else()
        set(JAX_VERSION "0.6.2")
    endif()
endif()
set(JAX_GIT_TAG "jax-v${JAX_VERSION}")
message(STATUS "Using JAX version: ${JAX_VERSION} (tag: ${JAX_GIT_TAG})")
CPMAddPackage(
    NAME jax
    GIT_REPOSITORY https://github.com/jax-ml/jax.git
    GIT_TAG ${JAX_GIT_TAG}
    DOWNLOAD_ONLY YES
)
message(STATUS "Jax source directory: ${jax_SOURCE_DIR}")
# https://github.com/openxla/xla/archive/8f3d7099bc30c2912357c770ca25172fad9ae302.zip
CPMAddPackage(
    NAME XLA
    GIT_REPOSITORY https://github.com/openxla/xla.git
    GIT_TAG 8f3d7099bc30c2912357c770ca25172fad9ae302
    DOWNLOAD_ONLY YES
)
message(STATUS "XLA source directory: ${XLA_SOURCE_DIR}")

# Since jaxlib is asking for #include "third_party/gpus/cuda/extras/CUPTI/include/cupti.h"
# we need to create a symlink to to CUDA_HOME and pretend it can be accessed from third_party
# Create the target directory
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/gpus)
execute_process(
    COMMAND ln -sfn $ENV{CUDA_ROOT} ${CMAKE_SOURCE_DIR}/third_party/gpus/cuda
    RESULT_VARIABLE ln_result
)
if(NOT ln_result EQUAL 0)
    message(FATAL_ERROR "Failed to create symlink for CUDA: $ENV{CUDA_ROOT}")
endif()

message(STATUS "Resolved cuDNN include dir: ${CUDNN_INCLUDE_PATH}")
execute_process(
    COMMAND ln -sfn ${CUDNN_INCLUDE_PATH} ${CMAKE_SOURCE_DIR}/third_party/gpus/cudnn
    RESULT_VARIABLE ln_result
)
if(NOT ln_result EQUAL 0)
    message(FATAL_ERROR "Failed to create symlink for CUDNN include: ${CUDNN_INCLUDE_PATH}")
endif()
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/xla/ffi)
execute_process(
    COMMAND ln -sfn ${XLA_SOURCE_DIR}/xla/ffi/api ${CMAKE_SOURCE_DIR}/xla/ffi
    RESULT_VARIABLE ln_result
)

if(NOT ln_result EQUAL 0)
    message(FATAL_ERROR "Failed to create symlink for XLA_DIR: ${XLA_SOURCE_DIR}")
endif()

# Set JAX_GPU_CUDA to 1 (this is required for jaxlib to compile correctly)
add_definitions(-DJAX_GPU_CUDA=1)
# Add libraries
add_library(cyclic SHARED
    src/cuda/cyclic.cu
    src/cuda/utils/jax_utils.cc
    src/cuda/utils/shm.cc
    ${jax_SOURCE_DIR}/jaxlib/gpu/gpu_kernel_helpers.cc
)

add_library(potrs SHARED
    src/cuda/potrs.cu
    src/cuda/utils/jax_utils.cc
    src/cuda/utils/shm.cc
    ${jax_SOURCE_DIR}/jaxlib/gpu/gpu_kernel_helpers.cc
)
add_library(potrs_mp SHARED
    src/cuda/potrs_mp.cu
    src/cuda/utils/jax_utils.cc
    src/cuda/utils/shm.cc
    src/cuda/utils/ipc_utils.cc
    ${jax_SOURCE_DIR}/jaxlib/gpu/gpu_kernel_helpers.cc
)
add_library(potri SHARED
    src/cuda/potri.cu
    src/cuda/utils/jax_utils.cc
    src/cuda/utils/shm.cc
    ${jax_SOURCE_DIR}/jaxlib/gpu/gpu_kernel_helpers.cc
)
add_library(syevd SHARED
    src/cuda/syevd.cu
    src/cuda/utils/jax_utils.cc
    src/cuda/utils/shm.cc
    ${jax_SOURCE_DIR}/jaxlib/gpu/gpu_kernel_helpers.cc
)
add_library(syevd_no_V SHARED
    src/cuda/syevd_no_V.cu
    src/cuda/utils/jax_utils.cc
    src/cuda/utils/shm.cc
    ${jax_SOURCE_DIR}/jaxlib/gpu/gpu_kernel_helpers.cc
)

# Make sure we can see "thirdparty/gpus" and jax
set(SRC_CUDA_INCLUDE ${CMAKE_SOURCE_DIR}/src/cuda/include)
set(SRC_CUDA_ROOT ${CMAKE_SOURCE_DIR}/src/cuda)
target_include_directories(cyclic PUBLIC ${CMAKE_SOURCE_DIR} ${jax_SOURCE_DIR} ${SRC_CUDA_INCLUDE} ${SRC_CUDA_ROOT})
target_include_directories(potrs PUBLIC ${CMAKE_SOURCE_DIR} ${jax_SOURCE_DIR} ${SRC_CUDA_INCLUDE} ${SRC_CUDA_ROOT})
target_include_directories(potrs_mp PUBLIC ${CMAKE_SOURCE_DIR} ${jax_SOURCE_DIR} ${SRC_CUDA_INCLUDE} ${SRC_CUDA_ROOT})
target_include_directories(potri PUBLIC ${CMAKE_SOURCE_DIR} ${jax_SOURCE_DIR} ${SRC_CUDA_INCLUDE} ${SRC_CUDA_ROOT})
target_include_directories(syevd PUBLIC ${CMAKE_SOURCE_DIR} ${jax_SOURCE_DIR} ${SRC_CUDA_INCLUDE} ${SRC_CUDA_ROOT})
target_include_directories(syevd_no_V PUBLIC ${CMAKE_SOURCE_DIR} ${jax_SOURCE_DIR} ${SRC_CUDA_INCLUDE} ${SRC_CUDA_ROOT})

# Set the CUDA architecture
set_target_properties(cyclic PROPERTIES
    CUDA_ARCHITECTURES "all"
)
set_target_properties(potrs PROPERTIES
    CUDA_ARCHITECTURES "all"
)
set_target_properties(potrs_mp PROPERTIES
    CUDA_ARCHITECTURES "all"
)
set_target_properties(potri PROPERTIES
    CUDA_ARCHITECTURES "all"
)
set_target_properties(syevd PROPERTIES
    CUDA_ARCHITECTURES "all"
)
set_target_properties(syevd_no_V PROPERTIES
    CUDA_ARCHITECTURES "all"
)

# Link against CUDA libraries and absl
target_link_libraries(cyclic
    absl::strings
    absl::status
    absl::statusor
    CUDA::cusolver
    CUDA::cupti
    ${CUSOLVERMG_LIB}
    cuda
)
target_link_libraries(potrs
    absl::strings
    absl::status
    absl::statusor
    CUDA::cusolver
    CUDA::cupti
    ${CUSOLVERMG_LIB}
    cuda
)
target_link_libraries(potrs_mp
    absl::strings
    absl::status
    absl::statusor
    CUDA::cusolver
    CUDA::cupti
    ${CUSOLVERMG_LIB}
    cuda
)

target_link_libraries(potri
    absl::strings
    absl::status
    absl::statusor
    CUDA::cusolver
    CUDA::cupti
    ${CUSOLVERMG_LIB}
    cuda
)
target_link_libraries(syevd
    absl::strings
    absl::status
    absl::statusor
    CUDA::cusolver
    CUDA::cupti
    ${CUSOLVERMG_LIB}
    cuda
)

target_link_libraries(syevd_no_V
    absl::strings
    absl::status
    absl::statusor
    CUDA::cusolver
    CUDA::cupti
    ${CUSOLVERMG_LIB}
    cuda
)
# Make install targets
install(TARGETS cyclic
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/${BIN_DST}
)
install(TARGETS potrs
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/${BIN_DST}
)
install(TARGETS potrs_mp
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/${BIN_DST}
)
install(TARGETS potri
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/${BIN_DST}
)
install(TARGETS syevd
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/${BIN_DST}
)
install(TARGETS syevd_no_V
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/${BIN_DST}
)

# module load gcc/11.5.0 cudnn/9.2.0.82-12 cuda/12.8.0

# cmake ..
# cmake --build . --target install -j10
# OR
# cmake --build . --target cyclic && cmake --install .
# cmake --build . --target potrs_mp && cmake --install .
# cmake --build . --target potrs && cmake --install .
# cmake --build . --target potri && cmake --install .
# cmake --build . --target syevd && cmake --install .
# cmake --build . --target syevd_no_V && cmake --install .

