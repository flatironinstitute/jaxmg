#!/usr/bin/env bash
#SBATCH --job-name=jaxmg_1d_prod
#SBATCH --partition=ampere
#SBATCH --account=acedo-sl2-gpu
#SBATCH --qos=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=jaxmg_1d_prod_%j.out

set -euo pipefail

module purge
module load gcc/11 cuda/12.1 cudnn/8.9_cuda-12.1 miniconda/3

ROOT="${SLURM_SUBMIT_DIR}"
PROJECT_ROOT="$(dirname "${ROOT}")"
ENV_DIR="${PROJECT_ROOT}/conda-env"

export PYTHONNOUSERSITE=1
export CUDA_ROOT=/usr/local/software/cuda/12.1
export CUDA_HOME="${CUDA_ROOT}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_DIR}"
set -u

NVIDIA_LIB_DIRS="$("${ENV_DIR}/bin/python" - <<'PY'
import importlib
import pathlib

modules = (
    "cuda_runtime",
    "cuda_nvrtc",
    "cuda_cupti",
    "cublas",
    "cusparse",
    "cusolver",
    "cufft",
    "cudnn",
    "nccl",
    "nvjitlink",
    "nvshmem",
)
paths = []
for module in modules:
    try:
        package = importlib.import_module(f"nvidia.{module}")
    except Exception:
        continue
    for root in getattr(package, "__path__", []):
        lib_dir = pathlib.Path(root) / "lib"
        if lib_dir.exists():
            paths.append(str(lib_dir))
print(":".join(dict.fromkeys(paths)))
PY
)"
if [[ -n "${NVIDIA_LIB_DIRS}" ]]; then
  export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
fi

cd "${ROOT}"
python -m pip install --no-deps -e .
python tools/csd3_test_single_node_1d_production.py \
  --devices "${JAXMG_SMOKE_DEVICES:-4}" \
  --n "${JAXMG_SMOKE_N:-256}" \
  --eig-n "${JAXMG_SMOKE_EIG_N:-128}" \
  --tile "${JAXMG_SMOKE_TILE:-64}"
