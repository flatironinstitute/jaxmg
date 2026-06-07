#!/usr/bin/env bash
#SBATCH --job-name=jaxmg_xla_backend
#SBATCH --partition=icelake
#SBATCH --account=acedo-sl2-cpu
#SBATCH --qos=cpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=jaxmg_xla_backend_%j.out

set -euo pipefail

module purge
module load gcc/11 cuda/12.1 cudnn/8.9_cuda-12.1

ROOT="${SLURM_SUBMIT_DIR}"
PROJECT_ROOT="$(dirname "${ROOT}")"
ENV_DIR="${PROJECT_ROOT}/conda-env"

export PYTHONNOUSERSITE=1
export CUDA_ROOT=/usr/local/software/cuda/12.1
export CUDA_HOME="${CUDA_ROOT}"
export JAX_VERSION="${JAX_VERSION:-0.10.1}"
export BAZEL="${ROOT}/tools/bazelisk"
export JAXMG_XLA_BAZEL_JOBS="${SLURM_CPUS_PER_TASK:-8}"
export JAXMG_XLA_BAZEL_CONFIGS="${JAXMG_XLA_BAZEL_CONFIGS:-bzlmod cuda}"

if [[ -x "${ENV_DIR}/bin/python" ]]; then
  "${ENV_DIR}/bin/python" -m pip install --no-deps -e "${ROOT}"
fi

cd "${ROOT}"
cmake -S . -B build-cu12-jax0101-absl20240722 \
  -DJAX_VERSION="${JAX_VERSION}"
cmake --build build-cu12-jax0101-absl20240722 \
  --target xla_comm_cusolvermg_backend \
  -j "${SLURM_CPUS_PER_TASK:-8}"

find src/jaxmg/cu12 -maxdepth 1 -name 'libxla_comm_collective_probe.so' -print | sort
