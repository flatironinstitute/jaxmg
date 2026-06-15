# Contributing
## TODO (last updated: June 6th, 2026)

### Small effort

- ~~**Get rid of compiler warnings** There is some unused code that needs to be removed. There are warnings due to things in JAXlib that we probably can't get rid of though.~~ (#10)

- **Better error handling**. There are parts of the code that simply throw `std::runtime_error`. We need to make the error handling compatible with the XLA_FFI error handlers like: `FFI_ASSIGN_OR_RETURN`, `JAX_FFI_RETURN_IF_GPU_ERROR`, etc... 

### Large effort

- Change to the CusolverMp API that's available for CUDA 13.

There has been a discussion with the cuSOLVERMp team at NVIDIA who can potentially assist with this. The migration branch now builds a native backend against matching JAX/XLA sources so the fused FFI handlers can borrow the XLA-owned GPU communicator. The production path uses that communicator to redistribute 2D JAX-sharded inputs into cuSOLVERMp's 2D block-cyclic layout and then calls cuSOLVERMp directly.

**Update May 28th:**
There seems to be a pathway to use the XLA-communicator directly, which is discussed here: 
https://github.com/openxla/xla/discussions/42689

## Build from source

This branch pins JAX/JAXLIB to `0.10.1` and builds the native CUDA backend as a
Bazel target inside the matching OpenXLA source tree.

The expected source build is:

```bash
export CUDA_ROOT=/path/to/cuda
export CUDA_HOME="${CUDA_ROOT}"
export BAZEL=/path/to/bazel-or-bazelisk

python -m pip install "jax[cuda12]==0.10.1"
python -m pip install --no-deps -e .

./build_native_backend.sh
```

This installs `src/jaxmg/cu12/libjaxmg_xla_comm_backend.so`, which registers
the production `cusolvermp_potrs` and `cusolvermp_syevd` FFI
targets. Additional communicator and cuSOLVERMp probe targets may be registered
when present in the shared library; these are diagnostic targets, not public
solver APIs.

The build helper checks out the pinned OpenXLA revision if `XLA_SRC` is not set.
Use `XLA_SRC` to point at an existing checkout, or `JAXMG_XLA_SOURCE_ROOT` to
choose where the helper stores its managed checkout. It also checks any
installed `jax` package against the expected pinned version and warns if JAX is
not installed in the active Python environment; set
`JAXMG_SKIP_JAX_VERSION_CHECK=1` only when intentionally overriding this check.
Bazel output defaults to
`$TMPDIR` via `JAXMG_XLA_BAZEL_CACHE_ROOT`; override
`JAXMG_XLA_BAZEL_OUTPUT_USER_ROOT` and `JAXMG_XLA_BAZEL_REPOSITORY_CACHE` if the
temporary filesystem is not suitable.

## JAX and CUDA

This branch pins the Python package to `jax==0.10.1` and builds the native
backend against the matching OpenXLA revision:

```bash
XLA_GIT_TAG=9b635916ecc6df6efee62d8e4b0c7ef87ef84d69
```

For CUDA 12, install JAX with either:

1. `pip install "jax[cuda12]==0.10.1"` to use NVIDIA Python wheels pulled in by JAX.

2. `pip install "jax[cuda12-local]==0.10.1"` to rely on a local CUDA installation.

CUDA 13 should use the same JAX pin with the CUDA 13 extras:

1. `pip install "jax[cuda13]==0.10.1"`

2. `pip install "jax[cuda13-local]==0.10.1"`

Changing the JAX version is not only a Python dependency change: the XLA
communicator APIs used here are internal OpenXLA APIs, so `XLA_GIT_TAG` in
`build_native_backend.sh` and the Bazel target/dependency list in
`bazel/jaxmg_backend.BUILD.bazel` must be audited together with any JAX/JAXLIB
version bump.

## Continuous integration

We make use of Jenkins to build and test the code. Jenkins builds the native
backend by calling `build_native_backend.sh` inside the CUDA
manylinux images, then packages the resulting shared library into wheels. We
test the following configurations:

1. A manylinux docker images (quay.io/pypa/manylinux_2_28_x86_64) where we install CUDA, CUDNN and NCCL.

2. Python `3.11`, `3.12`, `3.13`, `3.14`

3. For CUDA 12:
   - JAX `0.10.1`

   For CUDA 13 **currently only building code but no testing due to lack of availibility of CC > 7.0 GPUs. Locally tested on Blackwell.**
   - JAX `0.10.1`

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
JAX_VERSION=0.10.1
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
