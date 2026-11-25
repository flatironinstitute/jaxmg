#!/usr/bin/env bash
set -euo pipefail

# Build only JAXMg's CUDA backend; public FFI headers come from installed jaxlib.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
CUDA_HOME="${CUDA_HOME:-${CUDA_ROOT:-/usr/local/cuda}}"
CUDA_MAJOR="${CUDA_MAJOR:-$("${CUDA_HOME}/bin/nvcc" --version | sed -n 's/.*release \([0-9]*\).*/\1/p')}"
case "$CUDA_MAJOR" in
  12) DEFAULT_ARCHS='70-real;80-real;90-real;120-real;90-virtual' ;;
  13) DEFAULT_ARCHS='80-real;90-real;120-real;90-virtual' ;;
  *) echo "Expected CUDA 12 or 13, got ${CUDA_MAJOR}" >&2; exit 1 ;;
esac
BUILD_DIR="${JAXMG_BUILD_DIR:-${ROOT}/build/native-cu${CUDA_MAJOR}}"
cmake -S "$ROOT" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$(command -v "$PYTHON")" \
  -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES="${JAXMG_CUDA_ARCHITECTURES:-${DEFAULT_ARCHS}}" \
  -DJAXMG_RUN_NATIVE_TESTS="${JAXMG_RUN_NATIVE_TESTS:-OFF}" "$@"
cmake --build "$BUILD_DIR" --parallel "${JAXMG_BUILD_JOBS:-8}"
if [[ "${JAXMG_RUN_NATIVE_TESTS:-OFF}" != "OFF" && "${JAXMG_RUN_NATIVE_TESTS:-0}" != "0" ]]; then
  ctest --test-dir "$BUILD_DIR" --output-on-failure
fi
cmake --install "$BUILD_DIR" --prefix "$ROOT"
