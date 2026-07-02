#!/usr/bin/env bash
set -euo pipefail

# Builds libjaxmg_xla_comm_backend.so against XLA as the external Bazel repo
# `@xla`, from inside a JAX repo checkout (the JAX CI container workflow).
#
# Expected usage (see CONTRIBUTING.md):
#   1. Clone jax and start the JAX CI container:
#        git clone https://github.com/jax-ml/jax.git && cd jax
#        ./ci/utilities/run_docker_container.sh
#   2. Generate .jax_configure.bazelrc (CUDA/cuDNN/Bazel config):
#        docker exec jax ./ci/build_artifacts.sh jax-cuda-plugin
#   3. Run this script inside the container with JAX_SRC pointing at the jax
#      checkout (default /jax) and JAXMG_ROOT pointing at this repo:
#        docker exec jax bash -lc 'JAX_SRC=/jax JAXMG_ROOT=/path/to/jaxmg \
#          /path/to/jaxmg/build_native_backend.sh'
#
# The script no longer clones a pinned OpenXLA tree and no longer inspects the
# locally installed JAX version: both are determined by the container/jax repo.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JAXMG_ROOT="${JAXMG_ROOT:-${ROOT}}"
JAX_SRC="${JAX_SRC:-/jax}"
PYTHON="${PYTHON:-python}"
BAZEL="${BAZEL:-bazel}"
# Config provided by .jax_configure.bazelrc; we only add the stubs config used
# to build XLA targets in the JAX CI container.
BAZEL_CONFIGS="${JAXMG_XLA_BAZEL_CONFIGS:-cuda_libraries_from_stubs}"
BAZEL_JOBS="${JAXMG_XLA_BAZEL_JOBS:-8}"
# Build for a broad range of GPU architectures (Volta -> Hopper SASS, plus PTX
# for forward compatibility). Override for a narrower/faster build.
CUDA_COMPUTE_CAPABILITIES="${JAXMG_XLA_CUDA_COMPUTE_CAPABILITIES:-sm_70,sm_80,sm_90,compute_90}"
BACKEND_BUILD_TEMPLATE="${JAXMG_BACKEND_BUILD_TEMPLATE:-${JAXMG_ROOT}/bazel/jaxmg_backend.BUILD.bazel}"

if [[ ! -d "${JAX_SRC}" ]]; then
  echo "JAX checkout not found: ${JAX_SRC}" >&2
  echo "Set JAX_SRC to the jax repo root (the JAX CI container mounts it at /jax)." >&2
  exit 1
fi

if ! command -v "${BAZEL}" >/dev/null 2>&1; then
  echo "Unable to find Bazel executable: ${BAZEL}" >&2
  echo "Run inside the JAX CI container (tensorflow/ml-build:latest), which ships Bazel." >&2
  exit 1
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

CUSOLVERMP_INCLUDE_DIR="${JAXMG_CUSOLVERMP_INCLUDE_DIR:-}"
CUSOLVERMP_LIBRARY_DIR="${JAXMG_CUSOLVERMP_LIBRARY_DIR:-}"
CUSOLVERMP_LIBRARY_NAME="${JAXMG_CUSOLVERMP_LIBRARY_NAME:-libcusolverMp.so.0}"

if [[ -z "${CUSOLVERMP_INCLUDE_DIR}" || -z "${CUSOLVERMP_LIBRARY_DIR}" ]]; then
  if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    echo "Python executable not found for cuSOLVERMp wheel discovery: ${PYTHON}" >&2
    exit 1
  fi
  CUSOLVERMP_PATHS="$("${PYTHON}" - "${CUDA_MAJOR}" "${CUSOLVERMP_LIBRARY_NAME}" <<'PY'
import pathlib
import site
import sys

cuda_major = sys.argv[1]
library_name = sys.argv[2]

roots = [pathlib.Path(path) for path in sys.path if path]
for getter in (getattr(site, "getsitepackages", None), getattr(site, "getusersitepackages", None)):
    if getter is None:
        continue
    value = getter()
    if isinstance(value, str):
        roots.append(pathlib.Path(value))
    else:
        roots.extend(pathlib.Path(path) for path in value)

seen = set()
for root in roots:
    try:
        resolved = root.resolve()
    except OSError:
        resolved = root
    if resolved in seen:
        continue
    seen.add(resolved)
    cuda_root = root / "nvidia" / f"cu{cuda_major}"
    include_dir = cuda_root / "include"
    library_dir = cuda_root / "lib"
    if (include_dir / "cusolverMp.h").is_file() and (library_dir / library_name).is_file():
        print(include_dir)
        print(library_dir)
        sys.exit(0)

sys.exit(1)
PY
)" || {
    echo "Unable to find cuSOLVERMp headers/library in the Python environment." >&2
    echo "Install the matching NVIDIA wheel, e.g. ${PYTHON} -m pip install nvidia-cusolvermp-cu${CUDA_MAJOR}." >&2
    echo "Alternatively set JAXMG_CUSOLVERMP_INCLUDE_DIR and JAXMG_CUSOLVERMP_LIBRARY_DIR." >&2
    exit 1
  }
  CUSOLVERMP_INCLUDE_DIR="$(printf '%s\n' "${CUSOLVERMP_PATHS}" | sed -n '1p')"
  CUSOLVERMP_LIBRARY_DIR="$(printf '%s\n' "${CUSOLVERMP_PATHS}" | sed -n '2p')"
fi

if [[ ! -f "${CUSOLVERMP_INCLUDE_DIR}/cusolverMp.h" ]]; then
  echo "cuSOLVERMp header not found: ${CUSOLVERMP_INCLUDE_DIR}/cusolverMp.h" >&2
  exit 1
fi
if [[ ! -f "${CUSOLVERMP_LIBRARY_DIR}/${CUSOLVERMP_LIBRARY_NAME}" ]]; then
  echo "cuSOLVERMp library not found: ${CUSOLVERMP_LIBRARY_DIR}/${CUSOLVERMP_LIBRARY_NAME}" >&2
  exit 1
fi

echo "Using JAX checkout (provides @xla): ${JAX_SRC}"
echo "Using cuSOLVERMp include directory: ${CUSOLVERMP_INCLUDE_DIR}"
echo "Using cuSOLVERMp library directory: ${CUSOLVERMP_LIBRARY_DIR}"
echo "Building for compute capabilities: ${CUDA_COMPUTE_CAPABILITIES}"

# Assemble the backend Bazel package inside the JAX checkout so that bazel can
# build it while resolving the `@xla` external repo. Copy (rather than symlink)
# the sources so the package survives container mount boundaries.
BACKEND_PKG="${JAX_SRC}/jaxmg_backend"
rm -rf "${BACKEND_PKG}"
mkdir -p \
  "${BACKEND_PKG}/xla_ffi" \
  "${BACKEND_PKG}/memory_redist" \
  "${BACKEND_PKG}/cusolvermp_routines"
for src in handlers.cc runtime.cc runtime.h; do
  cp -f "${JAXMG_ROOT}/src/cuda/xla_ffi/${src}" \
    "${BACKEND_PKG}/xla_ffi/${src}"
done
for src in block_cyclic_2d.cc edge_padding_2d.cc layout_convert.cu.cc layout_convert.h memory_redist.h rectangle_pack.cc scratch.cc; do
  cp -f "${JAXMG_ROOT}/src/cuda/memory_redist/${src}" \
    "${BACKEND_PKG}/memory_redist/${src}"
done
for src in cusolvermp_common.cc cusolvermp_common.h cusolvermp_lu_solve.cc cusolvermp_potrs.cc cusolvermp_routines.h cusolvermp_syevd.cc; do
  cp -f "${JAXMG_ROOT}/src/cuda/cusolvermp_routines/${src}" \
    "${BACKEND_PKG}/cusolvermp_routines/${src}"
done
ln -sfn "${CUSOLVERMP_INCLUDE_DIR}" "${BACKEND_PKG}/cusolvermp_include"
ln -sfn "${CUSOLVERMP_LIBRARY_DIR}" "${BACKEND_PKG}/cusolvermp_lib"

sed "s/@JAXMG_CUDA_MAJOR@/${CUDA_MAJOR}/g" \
  "${BACKEND_BUILD_TEMPLATE}" > "${BACKEND_PKG}/BUILD.bazel"

cd "${JAX_SRC}"
export HERMETIC_CUDA_COMPUTE_CAPABILITIES="${CUDA_COMPUTE_CAPABILITIES}"
export TF_CUDA_COMPUTE_CAPABILITIES="${CUDA_COMPUTE_CAPABILITIES}"
BAZEL_CONFIG_ARGS=()
for config in ${BAZEL_CONFIGS}; do
  BAZEL_CONFIG_ARGS+=(--config="${config}")
done
"${BAZEL}" build \
  "${BAZEL_CONFIG_ARGS[@]}" \
  --verbose_failures=true \
  --repo_env="HERMETIC_CUDA_COMPUTE_CAPABILITIES=${CUDA_COMPUTE_CAPABILITIES}" \
  --jobs="${BAZEL_JOBS}" \
  //jaxmg_backend:libjaxmg_xla_comm_backend.so

INSTALL_DIR="${JAXMG_ROOT}/src/jaxmg/cu${CUDA_MAJOR}"
mkdir -p "${INSTALL_DIR}"
install -m 755 \
  "${JAX_SRC}/bazel-bin/jaxmg_backend/libjaxmg_xla_comm_backend.so" \
  "${INSTALL_DIR}/libjaxmg_xla_comm_backend.so"

echo "Installed ${INSTALL_DIR}/libjaxmg_xla_comm_backend.so"
