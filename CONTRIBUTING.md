# Contributing
## Build from source

To build from source:

```bash
mkdir build
cd build
cmake ..
cmake --build . --target install
```

This installs the CUDA binaries into `src/jaxmg/bin`

Dependencies are managed with [CPM-CMAKE](https://github.com/cpm-cmake/CPM.cmake),
including **abseil-cpp**, **jaxlib**, **XLA** for compilation. Compilation requires C++20 or later and an installation of CUDA Toolkit 12.x or 13.x.

To build specific targets only, for example potrs:
```bash
cmake ..
cmake --build . --target potrs && cmake --install .
```

then install the package with 

```bash
pip install .
```

To verify the installation (requires at least one GPU) run

```bash
pytest 
```
There are two types of tests:

1. SPMD tests: Single Process Multiple GPU tests.
3. MPMD: Multiple Processes Multiple GPU tests.

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

1. Two Linux docker images with CUDA 12 and 13 Toolkit installations:
   - `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`
   - `nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04`
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

```bash
mkdir dist && cd dist
VERSION=0.0.2
CUDA_FLAVOR=cuda12-local
JAX_VERSION=0.8.1
for PY in 3.11 3.12 3.13 3.14; do
   PYTAG=cp${PY/./}
   URL="https://jenkins.flatironinstitute.org/job/jaxmg/job/jenkins/lastBuild/artifact/${CUDA_FLAVOR}/${PY}/${JAX_VERSION}/dist/jaxmg-${VERSION}-${PYTAG}-${PYTAG}-linux_x86_64.whl"
   echo "Downloading ${URL}"
   wget -q --show-progress "${URL}"
done
```