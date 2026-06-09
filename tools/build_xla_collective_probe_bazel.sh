#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XLA_GIT_REPOSITORY="${XLA_GIT_REPOSITORY:-https://github.com/openxla/xla.git}"
XLA_GIT_TAG="${XLA_GIT_TAG:-9b635916ecc6df6efee62d8e4b0c7ef87ef84d69}"
XLA_SHORT_TAG="${XLA_GIT_TAG:0:12}"
XLA_SOURCE_ROOT="${JAXMG_XLA_SOURCE_ROOT:-${ROOT}/.jaxmg-xla}"
XLA_SRC="${XLA_SRC:-${XLA_SOURCE_ROOT}/xla-${XLA_SHORT_TAG}}"
BAZEL="${BAZEL:-bazel}"
BAZEL_CONFIGS="${JAXMG_XLA_BAZEL_CONFIGS:-bzlmod cuda}"
BAZEL_JOBS="${JAXMG_XLA_BAZEL_JOBS:-8}"
BAZEL_CACHE_ROOT="${JAXMG_XLA_BAZEL_CACHE_ROOT:-${TMPDIR:-/tmp}/jaxmg-bazel-${USER:-user}}"
BAZEL_OUTPUT_USER_ROOT="${JAXMG_XLA_BAZEL_OUTPUT_USER_ROOT:-${BAZEL_CACHE_ROOT}/output_user_root}"
BAZEL_REPOSITORY_CACHE="${JAXMG_XLA_BAZEL_REPOSITORY_CACHE:-${BAZEL_CACHE_ROOT}/repository_cache}"

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
  if [[ -x "${ROOT}/tools/bazelisk" ]]; then
    BAZEL="${ROOT}/tools/bazelisk"
  else
    echo "Unable to find Bazel/Bazelisk. Set BAZEL or install tools/bazelisk." >&2
    exit 1
  fi
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

PROBE_PKG="${XLA_SRC}/jaxmg_probe"
mkdir -p "${PROBE_PKG}/include" "${PROBE_PKG}/utils"
ln -sfn "${ROOT}/src/cuda/include/xla_comm_backend.h" \
  "${PROBE_PKG}/include/xla_comm_backend.h"
ln -sfn "${ROOT}/src/cuda/include/mpmd_ipc.h" \
  "${PROBE_PKG}/include/mpmd_ipc.h"
ln -sfn "${ROOT}/src/cuda/utils/xla_comm_common.cc" \
  "${PROBE_PKG}/utils/xla_comm_common.cc"
ln -sfn "${ROOT}/src/cuda/utils/mpmd_ipc.cc" \
  "${PROBE_PKG}/utils/mpmd_ipc.cc"
for src in \
  collective_diagnostics.cc \
  cyclic_1d.cc \
  ffi_handlers.cc \
  potri.cc \
  potrs.cc \
  syevd.cc; do
  ln -sfn "${ROOT}/src/cuda/${src}" "${PROBE_PKG}/${src}"
done

cat >"${PROBE_PKG}/BUILD.bazel" <<EOF
package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "libxla_comm_collective_probe.so",
    srcs = [
        "collective_diagnostics.cc",
        "cyclic_1d.cc",
        "ffi_handlers.cc",
        "potri.cc",
        "potrs.cc",
        "syevd.cc",
        "include/xla_comm_backend.h",
        "include/mpmd_ipc.h",
        "utils/xla_comm_common.cc",
        "utils/mpmd_ipc.cc",
    ],
    linkshared = True,
    linkstatic = True,
    copts = [
        "-Iexternal/rules_ml_toolchain~~cuda_redist_init_ext~cuda_cudart/include",
        "-Iexternal/rules_ml_toolchain~~cuda_redist_init_ext~cuda_cusolver/include",
        "-Iexternal/rules_ml_toolchain~~cuda_redist_init_ext~cuda_nvcc/include",
        "-Iexternal/rules_ml_toolchain~~cuda_redist_init_ext~cuda_cublas/include",
        "-Iexternal/_main~cccl_extension~cuda_cccl/libcudacxx/include",
    ],
    linkopts = [
        "-Lexternal/rules_ml_toolchain~~cuda_redist_init_ext~cuda_cusolver/lib",
        "-lcusolverMg",
    ],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@cuda_cudart//:cudart",
        "@cuda_cusolver//:cusolver",
        "//xla:xla_data_proto_cc",
        "//xla/backends/gpu/collectives:gpu_clique_key",
        "//xla/backends/gpu/collectives:gpu_collectives",
        "//xla/backends/gpu/collectives:gpu_communicator",
        "//xla/backends/gpu:ffi",
        "//xla/backends/gpu/runtime:collective_clique_requests",
        "//xla/backends/gpu/runtime:collective_cliques",
        "//xla/backends/gpu/runtime:collective_execution",
        "//xla/backends/gpu/runtime:collective_params",
        "//xla/core/collectives:communicator",
        "//xla/core/collectives:rank_id",
        "//xla/core/collectives:reduction_kind",
        "//xla/ffi",
        "//xla/ffi:ffi_api",
        "//xla:future",
        "//xla/runtime:device_id",
        "//xla/stream_executor:stream",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)
EOF

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
  //jaxmg_probe:libxla_comm_collective_probe.so

INSTALL_DIR="${ROOT}/src/jaxmg/cu${CUDA_MAJOR}"
mkdir -p "${INSTALL_DIR}"
install -m 755 \
  "${XLA_SRC}/bazel-bin/jaxmg_probe/libxla_comm_collective_probe.so" \
  "${INSTALL_DIR}/libxla_comm_collective_probe.so"

echo "Installed ${INSTALL_DIR}/libxla_comm_collective_probe.so"
