#!/usr/bin/env bash
#SBATCH --job-name=jaxmg_xla_probe_build
#SBATCH --partition=icelake
#SBATCH --account=acedo-sl2-cpu
#SBATCH --qos=cpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=jaxmg_xla_probe_build_%j.out

set -euo pipefail

module purge
module load gcc/11 cuda/12.1 cudnn/8.9_cuda-12.1

export CUDA_ROOT=/usr/local/software/cuda/12.1
export CUDA_HOME="${CUDA_ROOT}"
export BAZEL="${SLURM_SUBMIT_DIR}/tools/bazelisk"
export JAXMG_XLA_BAZEL_JOBS="${SLURM_CPUS_PER_TASK:-8}"
export JAXMG_XLA_BAZEL_CONFIGS="${JAXMG_XLA_BAZEL_CONFIGS:-bzlmod cuda}"

cd "${SLURM_SUBMIT_DIR}"
tools/build_xla_collective_probe_bazel.sh
