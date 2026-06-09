cmake_minimum_required(VERSION 3.25.2)
project(JaxMgBazelBackendCompatibility LANGUAGES NONE)

# The native JAXMg backend is built by Bazel. This CMake file exists only to
# preserve the historical maintainer command shape:
#
#   cmake --build <build-dir> --target install
#
# No native code is configured or compiled by CMake.

add_custom_target(xla_comm_cusolvermg_backend
    COMMAND "${CMAKE_SOURCE_DIR}/tools/build_xla_collective_probe_bazel.sh"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Build the JAXMg XLA communicator/cuSolverMg backend with Bazel"
    USES_TERMINAL
)

install(CODE "
message(STATUS \"Building and installing the JAXMg XLA communicator/cuSolverMg backend with Bazel\")
execute_process(
    COMMAND \"${CMAKE_SOURCE_DIR}/tools/build_xla_collective_probe_bazel.sh\"
    WORKING_DIRECTORY \"${CMAKE_SOURCE_DIR}\"
    RESULT_VARIABLE JAXMG_XLA_BACKEND_RESULT
)
if(NOT JAXMG_XLA_BACKEND_RESULT EQUAL 0)
    message(FATAL_ERROR \"JAXMg XLA communicator/cuSolverMg backend build failed\")
endif()
")
