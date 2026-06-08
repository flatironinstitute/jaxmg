cmake_minimum_required(VERSION 3.25.2)
project(JaxMgXlaBackendBootstrap LANGUAGES NONE)

# CMake is now only used to materialize the pinned JAX/OpenXLA sources that the
# Bazel-built XLA communicator backend links against. The production CUDA
# backend is built by tools/build_xla_collective_probe_bazel.sh.

set(CPM_DOWNLOAD_VERSION 0.42.0)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
if(NOT EXISTS "${CPM_DOWNLOAD_LOCATION}")
    message(STATUS "Downloading CPM.cmake")
    file(DOWNLOAD
        "https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake"
        "${CPM_DOWNLOAD_LOCATION}"
    )
endif()
include("${CPM_DOWNLOAD_LOCATION}")

if(NOT DEFINED JAX_VERSION OR JAX_VERSION STREQUAL "")
    if(DEFINED ENV{JAX_VERSION} AND NOT "$ENV{JAX_VERSION}" STREQUAL "")
        set(JAX_VERSION "$ENV{JAX_VERSION}")
    else()
        set(JAX_VERSION "0.10.1")
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
message(STATUS "JAX source directory: ${jax_SOURCE_DIR}")

if(NOT DEFINED XLA_GIT_TAG OR XLA_GIT_TAG STREQUAL "")
    if(DEFINED ENV{XLA_GIT_TAG} AND NOT "$ENV{XLA_GIT_TAG}" STREQUAL "")
        set(XLA_GIT_TAG "$ENV{XLA_GIT_TAG}")
    else()
        # Matching OpenXLA revision for JAX/JAXLIB 0.10.1.
        set(XLA_GIT_TAG "9b635916ecc6df6efee62d8e4b0c7ef87ef84d69")
    endif()
endif()
message(STATUS "Using XLA tag: ${XLA_GIT_TAG}")
CPMAddPackage(
    NAME XLA
    GIT_REPOSITORY https://github.com/openxla/xla.git
    GIT_TAG ${XLA_GIT_TAG}
    DOWNLOAD_ONLY YES
)
message(STATUS "XLA source directory: ${XLA_SOURCE_DIR}")

add_custom_target(xla_comm_cusolvermg_backend
    COMMAND
        "${CMAKE_COMMAND}" -E env "XLA_SRC=${XLA_SOURCE_DIR}"
        "${CMAKE_SOURCE_DIR}/tools/build_xla_collective_probe_bazel.sh"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Build the JAXMg XLA communicator/cuSolverMg backend with Bazel"
    USES_TERMINAL
)

# Preserve the historical CMake/Jenkins contract:
#
#   cmake --build <build-dir> --target install
#
# The native backend is now built by Bazel, but the maintainer CI still expects
# the CMake install target to populate src/jaxmg/cu*/ with package data.
install(CODE "
message(STATUS \"Building and installing the JAXMg XLA communicator/cuSolverMg backend\")
execute_process(
    COMMAND \"${CMAKE_COMMAND}\" -E env \"XLA_SRC=${XLA_SOURCE_DIR}\"
            \"${CMAKE_SOURCE_DIR}/tools/build_xla_collective_probe_bazel.sh\"
    WORKING_DIRECTORY \"${CMAKE_SOURCE_DIR}\"
    RESULT_VARIABLE JAXMG_XLA_BACKEND_RESULT
)
if(NOT JAXMG_XLA_BACKEND_RESULT EQUAL 0)
    message(FATAL_ERROR \"JAXMg XLA communicator/cuSolverMg backend build failed\")
endif()
")
