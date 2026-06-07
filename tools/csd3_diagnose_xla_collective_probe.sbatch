#!/usr/bin/env bash
#SBATCH --job-name=jaxmg_xla_probe_diag
#SBATCH --partition=ampere
#SBATCH --account=acedo-sl2-gpu
#SBATCH --qos=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=jaxmg_xla_probe_diag_%j.out

set -euo pipefail
set -x

module purge
module load gcc/11 cuda/12.1 cudnn/8.9_cuda-12.1

export CUDA_ROOT=/usr/local/software/cuda/12.1
export CUDA_HOME="${CUDA_ROOT}"
export BAZEL="${SLURM_SUBMIT_DIR}/tools/bazelisk"
export ROOT="${SLURM_SUBMIT_DIR}"
export BUILD_DIR="${ROOT}/build-cu12-jax0101-absl20240722"
export XLA_SRC="${BUILD_DIR}/_deps/xla-src"
export BAZEL_OUTPUT_USER_ROOT="${ROOT}/.bazel-cache/diag_output_user_root"
export BAZEL_REPO_CACHE="${ROOT}/.bazel-cache/repository_cache"
export PATH="${ROOT}/tools:${PATH}"

mkdir -p \
  "${BAZEL_OUTPUT_USER_ROOT}" \
  "${BAZEL_REPO_CACHE}" \
  "${XLA_SRC}/jaxmg_probe/include" \
  "${XLA_SRC}/jaxmg_probe/utils"
ln -sfn "${ROOT}/src/cuda/include/xla_comm_backend.h" \
  "${XLA_SRC}/jaxmg_probe/include/xla_comm_backend.h"
ln -sfn "${ROOT}/src/cuda/utils/xla_comm_common.cc" \
  "${XLA_SRC}/jaxmg_probe/utils/xla_comm_common.cc"
for src in \
  collective_diagnostics.cc \
  cyclic_1d.cc \
  ffi_handlers.cc \
  potri.cc \
  potrs.cc \
  syevd.cc; do
  ln -sfn "${ROOT}/src/cuda/${src}" "${XLA_SRC}/jaxmg_probe/${src}"
done

cat >"${XLA_SRC}/jaxmg_probe/BUILD.bazel" <<'BUILD'
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
        "utils/xla_comm_common.cc",
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
BUILD

cd "${XLA_SRC}"
"${BAZEL}" --output_user_root="${BAZEL_OUTPUT_USER_ROOT}" shutdown || true

echo "=== bazel version ==="
"${BAZEL}" --output_user_root="${BAZEL_OUTPUT_USER_ROOT}" version

echo "=== bazel query ==="
timeout 15m "${BAZEL}" \
  --output_user_root="${BAZEL_OUTPUT_USER_ROOT}" \
  query \
  --config=bzlmod \
  --repository_cache="${BAZEL_REPO_CACHE}" \
  --show_progress_rate_limit=5 \
  --ui_event_filters=info,debug,warning,error \
  --noshow_loading_progress \
  'deps(//jaxmg_probe:libxla_comm_collective_probe.so)' \
  >"${ROOT}/jaxmg_xla_probe_diag_query_${SLURM_JOB_ID}.txt"

echo "=== query completed ==="
wc -l "${ROOT}/jaxmg_xla_probe_diag_query_${SLURM_JOB_ID}.txt"

echo "=== bazel build ==="
timeout 35m "${BAZEL}" \
  --output_user_root="${BAZEL_OUTPUT_USER_ROOT}" \
  build \
  --config=bzlmod \
  --config=cuda \
  --repository_cache="${BAZEL_REPO_CACHE}" \
  --jobs="${SLURM_CPUS_PER_TASK:-8}" \
  --show_progress_rate_limit=5 \
  --ui_event_filters=info,debug,warning,error \
  --output_filter= \
  --verbose_failures \
  //jaxmg_probe:libxla_comm_collective_probe.so

echo "=== build completed ==="
ls -lh bazel-bin/jaxmg_probe/libxla_comm_collective_probe.so
install -m 755 bazel-bin/jaxmg_probe/libxla_comm_collective_probe.so \
  "${ROOT}/src/jaxmg/cu12/libxla_comm_collective_probe.so"
ls -lh "${ROOT}/src/jaxmg/cu12/libxla_comm_collective_probe.so"
