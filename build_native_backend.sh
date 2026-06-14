#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XLA_GIT_REPOSITORY="${XLA_GIT_REPOSITORY:-https://github.com/openxla/xla.git}"
XLA_GIT_TAG="${XLA_GIT_TAG:-9b635916ecc6df6efee62d8e4b0c7ef87ef84d69}"
XLA_SHORT_TAG="${XLA_GIT_TAG:0:12}"
XLA_SOURCE_ROOT="${JAXMG_XLA_SOURCE_ROOT:-${ROOT}/.jaxmg-xla}"
XLA_SRC="${XLA_SRC:-${XLA_SOURCE_ROOT}/xla-${XLA_SHORT_TAG}}"
EXPECTED_JAX_VERSION="${JAXMG_EXPECTED_JAX_VERSION:-0.10.1}"
PYTHON="${PYTHON:-python}"
BAZEL="${BAZEL:-bazel}"
BAZEL_CONFIGS="${JAXMG_XLA_BAZEL_CONFIGS:-bzlmod cuda}"
BAZEL_JOBS="${JAXMG_XLA_BAZEL_JOBS:-8}"
BAZEL_CACHE_ROOT="${JAXMG_XLA_BAZEL_CACHE_ROOT:-${TMPDIR:-/tmp}/jaxmg-bazel-${USER:-user}}"
BAZEL_OUTPUT_USER_ROOT="${JAXMG_XLA_BAZEL_OUTPUT_USER_ROOT:-${BAZEL_CACHE_ROOT}/output_user_root}"
BAZEL_REPOSITORY_CACHE="${JAXMG_XLA_BAZEL_REPOSITORY_CACHE:-${BAZEL_CACHE_ROOT}/repository_cache}"
BACKEND_BUILD_TEMPLATE="${JAXMG_BACKEND_BUILD_TEMPLATE:-${ROOT}/bazel/jaxmg_backend.BUILD.bazel}"

if [[ ! -d "${XLA_SRC}/.git" ]]; then
  if [[ -e "${XLA_SRC}" ]]; then
    echo "XLA source path exists but is not a git checkout: ${XLA_SRC}" >&2
    echo "Set XLA_SRC to a valid OpenXLA checkout or remove the path." >&2
    exit 1
  fi
  mkdir -p "$(dirname "${XLA_SRC}")"
  git clone "${XLA_GIT_REPOSITORY}" "${XLA_SRC}"
fi

if ! git -C "${XLA_SRC}" cat-file -e "${XLA_GIT_TAG}^{commit}" 2>/dev/null; then
  git -C "${XLA_SRC}" fetch origin
fi
git -C "${XLA_SRC}" checkout --detach "${XLA_GIT_TAG}"

if ! command -v "${BAZEL}" >/dev/null 2>&1; then
  echo "Unable to find Bazel executable: ${BAZEL}" >&2
  echo "Set BAZEL to a bazel/bazelisk path or install Bazelisk on PATH." >&2
  exit 1
fi

if [[ "${JAXMG_SKIP_JAX_VERSION_CHECK:-0}" != "1" ]]; then
  if command -v "${PYTHON}" >/dev/null 2>&1; then
    JAX_VERSION="$("${PYTHON}" - <<'PY' 2>/dev/null || true
import importlib.metadata as metadata

try:
    print(metadata.version("jax"))
except metadata.PackageNotFoundError:
    pass
PY
)"
    if [[ -n "${JAX_VERSION}" && "${JAX_VERSION}" != "${EXPECTED_JAX_VERSION}" ]]; then
      echo "JAX version mismatch: found ${JAX_VERSION}, expected ${EXPECTED_JAX_VERSION}." >&2
      echo "This backend is pinned to a matching OpenXLA revision; install the pinned JAX version or set JAXMG_SKIP_JAX_VERSION_CHECK=1." >&2
      exit 1
    elif [[ -n "${JAX_VERSION}" ]]; then
      echo "Using Python JAX version: ${JAX_VERSION}"
    else
      echo "JAX is not installed in ${PYTHON}; skipping Python package version check." >&2
    fi
  else
    echo "Python executable not found for JAX version check: ${PYTHON}" >&2
  fi
fi

if [[ ! -f "${BACKEND_BUILD_TEMPLATE}" ]]; then
  echo "Unable to find Bazel backend target template: ${BACKEND_BUILD_TEMPLATE}" >&2
  exit 1
fi

if [[ -z "${CUDA_HOME:-}" && -n "${CUDA_ROOT:-}" ]]; then
  export CUDA_HOME="${CUDA_ROOT}"
fi

CUDA_MAJOR="${CUDA_MAJOR:-}"
if [[ -z "${CUDA_MAJOR}" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_MAJOR="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\).*/\1/p' | head -1)"
  elif [[ -n "${CUDA_HOME:-}" ]]; then
    CUDA_MAJOR="$(basename "${CUDA_HOME}" | sed -n 's/^\([0-9][0-9]*\).*/\1/p')"
  fi
fi
if [[ -z "${CUDA_MAJOR}" ]]; then
  echo "Unable to infer CUDA major version. Set CUDA_MAJOR." >&2
  exit 1
fi

echo "Using OpenXLA checkout: ${XLA_SRC}"
echo "Using OpenXLA revision: $(git -C "${XLA_SRC}" rev-parse --short HEAD)"

BACKEND_PKG="${XLA_SRC}/jaxmg_backend"
mkdir -p "${BACKEND_PKG}/include" "${BACKEND_PKG}/utils"
ln -sfn "${ROOT}/src/cuda/include/xla_comm_backend.h" \
  "${BACKEND_PKG}/include/xla_comm_backend.h"
ln -sfn "${ROOT}/src/cuda/include/mpmd_ipc.h" \
  "${BACKEND_PKG}/include/mpmd_ipc.h"
ln -sfn "${ROOT}/src/cuda/utils/xla_comm_common.cc" \
  "${BACKEND_PKG}/utils/xla_comm_common.cc"
ln -sfn "${ROOT}/src/cuda/utils/mpmd_ipc.cc" \
  "${BACKEND_PKG}/utils/mpmd_ipc.cc"
for src in \
  block_cyclic_2d.cc \
  collective_diagnostics.cc \
  cusolvermp.cc \
  cyclic_1d.cc \
  edge_padding_2d.cc \
  ffi_handlers.cc \
  potri.cc \
  potrs.cc \
  rectangle_pack.cc \
  syevd.cc; do
  ln -sfn "${ROOT}/src/cuda/${src}" "${BACKEND_PKG}/${src}"
done

cp "${BACKEND_BUILD_TEMPLATE}" "${BACKEND_PKG}/BUILD.bazel"

cd "${XLA_SRC}"
mkdir -p "${BAZEL_OUTPUT_USER_ROOT}" "${BAZEL_REPOSITORY_CACHE}"
BAZEL_CONFIG_ARGS=()
for config in ${BAZEL_CONFIGS}; do
  BAZEL_CONFIG_ARGS+=(--config="${config}")
done
"${BAZEL}" \
  --output_user_root="${BAZEL_OUTPUT_USER_ROOT}" \
  build \
  "${BAZEL_CONFIG_ARGS[@]}" \
  --repository_cache="${BAZEL_REPOSITORY_CACHE}" \
  --jobs="${BAZEL_JOBS}" \
  //jaxmg_backend:libjaxmg_xla_comm_backend.so

INSTALL_DIR="${ROOT}/src/jaxmg/cu${CUDA_MAJOR}"
mkdir -p "${INSTALL_DIR}"
install -m 755 \
  "${XLA_SRC}/bazel-bin/jaxmg_backend/libjaxmg_xla_comm_backend.so" \
  "${INSTALL_DIR}/libjaxmg_xla_comm_backend.so"

echo "Installed ${INSTALL_DIR}/libjaxmg_xla_comm_backend.so"
