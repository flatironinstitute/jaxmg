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

The native CUDA backend is built as a Bazel target **against XLA as the external
Bazel repo `@xla`**, from inside the official JAX CI container
(`tensorflow/ml-build:latest`) using a JAX checkout. We no longer clone a pinned
OpenXLA tree: the JAX checkout determines the XLA revision that `@xla` resolves
to, so the build configuration always matches JAX/XLA.

```bash
# 1. Clone JAX (provides @xla) at a tag compatible with the jax pin in
#    pyproject.toml, and start the JAX CI container.
git clone --branch jax-v0.10.1 https://github.com/jax-ml/jax.git
cd jax
./ci/utilities/run_docker_container.sh          # starts a container named "jax"

# 2. Generate .jax_configure.bazelrc (CUDA/cuDNN/Bazel config).
docker exec jax ./ci/build_artifacts.sh jax-cuda-plugin

# 3. Make this jaxmg checkout available inside the container (mounted at /jax),
#    install cuSOLVERMp, and build the backend against @xla.
#    Here JAXMG_ROOT is the path to this repo as seen inside the container.
docker exec jax bash -lc '
  python -m pip install nvidia-cusolvermp-cu12==0.8.0.3126 &&
  JAX_SRC=/jax JAXMG_ROOT=/path/to/jaxmg ./path/to/jaxmg/build_native_backend.sh'
```

This installs `src/jaxmg/cu12/libjaxmg_xla_comm_backend.so`, which registers the
production `cusolvermp_potrs` and `cusolvermp_syevd` FFI targets. Additional
communicator and cuSOLVERMp probe targets may be registered when present in the
shared library; these are diagnostic targets, not public solver APIs.

`build_native_backend.sh` assembles the backend sources into
`${JAX_SRC}/jaxmg_backend`, substitutes the CUDA major version into
`bazel/jaxmg_backend.BUILD.bazel`, and runs
`bazel build --config=cuda_libraries_from_stubs //jaxmg_backend:libjaxmg_xla_comm_backend.so`.
Key environment variables:

- `JAX_SRC` — the JAX checkout root (default `/jax`); this is where `@xla` is defined.
- `JAXMG_ROOT` — path to this repo (default: the script's own directory).
- `JAXMG_XLA_CUDA_COMPUTE_CAPABILITIES` — target GPU archs. Default is a broad
  list `sm_70,sm_80,sm_90,compute_90` (Volta→Hopper SASS + PTX). CUDA 13 drops
  Volta, so use e.g. `sm_80,sm_90,compute_90` there.
- `JAXMG_CUSOLVERMP_INCLUDE_DIR` / `JAXMG_CUSOLVERMP_LIBRARY_DIR` — override the
  cuSOLVERMp wheel auto-discovery.

## JAX and CUDA

The Python package pins `jax==0.10.1` (see `pyproject.toml`). The native backend
depends on **internal** OpenXLA APIs, so changing the JAX version is not just a
Python dependency change: the `@xla` targets in `bazel/jaxmg_backend.BUILD.bazel`
must be audited against the XLA revision behind the new JAX checkout, and
`JAX_GIT_TAG` in `.jenkins/Jenkinsfile` updated to match.

For CUDA 12, install JAX with either:

1. `pip install "jax[cuda12]==0.10.1"` to use NVIDIA Python wheels pulled in by JAX.

2. `pip install "jax[cuda12-local]==0.10.1"` to rely on a local CUDA installation.

CUDA 13 should use the same JAX pin with the CUDA 13 extras:

1. `pip install "jax[cuda13]==0.10.1"`

2. `pip install "jax[cuda13-local]==0.10.1"`

## Continuous integration

We make use of Jenkins to build and test the code. Jenkins builds the native
backend by calling `build_native_backend.sh` inside the official JAX CI
container (`tensorflow/ml-build:latest`) against a JAX checkout, then packages
the resulting shared library into wheels. We test the following configurations:

1. The JAX CI container `tensorflow/ml-build:latest`, building against `@xla`
   from a `jax` checkout (`JAX_GIT_TAG` in `.jenkins/Jenkinsfile`).

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
