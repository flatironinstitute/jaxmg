# Contributing
## TODO (last updated: June 6th, 2026)

### Small effort

- ~~**Get rid of compiler warnings** There is some unused code that needs to be removed. There are warnings due to things in JAXlib that we probably can't get rid of though.~~ (#10)

- **Better error handling**. There are parts of the code that simply throw `std::runtime_error`. We need to make the error handling compatible with the XLA_FFI error handlers like: `FFI_ASSIGN_OR_RETURN`, `JAX_FFI_RETURN_IF_GPU_ERROR`, etc... 

### Large effort

- Change to the CusolverMp API that's available for CUDA 13.

There has been a discussion with the cuSOLVERMp team at NVIDIA who can potentially assist with this. The main problem is communicating between the different threads/processes that launch cuSOLVERMp from JAX. Since JAX launches a thread/process for each GPU, we need to synchronize these invocations and orchestrate calls to cuSOLVER from a designated master process. The current migration branch now uses the XLA-owned communicator from FFI for the single-node 1D cuSOLVERMg path instead of the old shared-memory/cudaMemcpyPeerAsync shuffler. Multi-node cuSOLVERMp support still needs the 2D block-cyclic redistribution and a multi-node validation path.

**Update May 28th:**
There seems to be a pathway to use the XLA-communicator directly, which is discussed here: 
https://github.com/openxla/xla/discussions/42689

## Build from source

This branch pins JAX/JAXLIB to `0.10.1` and builds the native CUDA backend as a
Bazel target inside the matching OpenXLA source tree. CMake is only used to
materialize the pinned JAX/OpenXLA sources and hand off to Bazel.

On CSD3, the expected source build is:

```bash
module purge
module load gcc/11 cuda/12.1 cudnn/8.9_cuda-12.1

export CUDA_ROOT=/usr/local/software/cuda/12.1
export CUDA_HOME="${CUDA_ROOT}"
export JAX_VERSION=0.10.1
export BAZEL="${PWD}/tools/bazelisk"

python -m pip install "jax[cuda12]==0.10.1"
python -m pip install --no-deps -e .

cmake -S . -B build-cu12-jax0101-absl20240722 -DJAX_VERSION="${JAX_VERSION}"
cmake --build build-cu12-jax0101-absl20240722 --target xla_comm_cusolvermg_backend
```

This installs `src/jaxmg/cu12/libxla_comm_collective_probe.so`, which registers
the production `potrs_mg`, `potri_mg`, `syevd_mg`, `syevd_no_V_mg`, and
`xla_comm_matrix_column_native_plan` FFI targets.

The equivalent Slurm build helper is:

```bash
sbatch tools/csd3_build_native_cuda12.sbatch
```

To verify the single-node 1D backend on four GPUs:

```bash
sbatch tools/csd3_test_single_node_1d_production.sbatch
```

## JAX and CUDA

As of version 0.6.2, JAX can be installed for GPU usage in two ways:

1. `pip install "jax[cuda12]"`: Install a NVIDIA python module along side the jax installation and rely on those binaries for CUDA functionality.

2. `pip install "jax[cuda12-local]"`: Rely on a local installation.

As of version 0.7.2, JAX is compatible with CUDA 13:

1. `pip install "jax[cuda13]"`:

2. `pip install "jax[cuda13-local]"`

At compilation time, we do not need to worry about the distinction between `cudax` and `cudax-local`, since the symbols we link again are resolved at runtime via `import jax`. However, 

Jaxlib contains C++ headers that have to be compiled against. To compile against a specific Jaxlib version, set the environment variable
`JAX_VERSION` before building. For CUDA 12, `JAX_VERSION=0.6.2` is backwards compatible up to `jax==0.8.x`, but for CUDA 13 you must set
`JAX_VERSION>=0.7.2` or you will get compilation errors.

## Continuous integration

We make use of Jenkins to build and test the code. We test the following configurations:

1. A manylinux docker images (quay.io/pypa/manylinux_2_28_x86_64) where we install CUDA, CUDNN and NCCL.

2. Python `3.11`, `3.12`, `3.13`, `3.14`

3. For CUDA 12:
   - JAX `0.6.2`, `0.7.1`, `0.8.1`

   For CUDA 13 **currently only building code but no testing due to lack of availibility of CC > 7.0 GPUs. Locally tested on Blackwell.**
   - JAX `0.7.2`, `0.8.1`

See `.jenkins/Jenkinsfile` for details

## Documentation setup

See https://olgarithms.github.io/sphinx-tutorial/docs/7-hosting-on-github-pages.html

Make sure you install `jaxmg[docs]` to be able to generate the documentation.
Run 

```bash
mkdocs serve
```

to serve the docs locally. On push to main, the docs are automatically deployed with the `.github/workflow/deploy-docs.yml` action.

## Publish Package

Get the latest built wheels from Jenkins:

```bash
mkdir dist
VERSION=0.0.8
CUDA_FLAVOR=cuda12-local
JAX_VERSION=0.8.1
for PY in 3.11 3.12 3.13 3.14; do
   PYTAG=cp${PY/./}
   URL="https://jenkins.flatironinstitute.org/job/jaxmg/job/main/lastBuild/artifact/${CUDA_FLAVOR}/${PY}/${JAX_VERSION}/dist_repaired/jaxmg-${VERSION}-${PYTAG}-${PYTAG}-manylinux_2_26_x86_64.whl"
   echo "Downloading ${URL}"
   wget -q -N --show-progress "${URL}" -P ./dist
done
```
Install twine
```bash
python -m pip install twine
```
Upload to testpypi
```bash
python -m twine upload --repository testpypi dist/*
```
Test the wheel
```bash
pip install -i https://test.pypi.org/simple/ "jaxmg[cuda12]==0.0.8" --extra-index-url https://pypi.org/simple
```
