# Building the Native Backend

Most users should install a prebuilt wheel from the
[Installation](../install.md) page. Build from source when changing the
C++/CUDA backend, updating JAX or CUDA, or targeting a platform not covered by
the release wheels.

## Why a separate native build is needed

JAXMg uses internal XLA C++ interfaces for FFI and GPU communicator access.
These interfaces are available through Bazel's `@xla` repository inside the
matching JAX source tree; they are not distributed as a standalone library.

The build therefore combines:

- the JAXMg C++/CUDA sources;
- the JAX `0.10.1` checkout and its pinned XLA revision;
- the selected CUDA and cuSOLVERMp dependencies.

The result is `libjaxmg_xla_comm_backend.so`, installed under
`src/jaxmg/cu12/` or `src/jaxmg/cu13/`.

```text
JAXMg sources + JAX/XLA checkout + cuSOLVERMp
                        |
                        v
               pinned Bazel/CUDA build
                        |
                        v
        src/jaxmg/cuXX/libjaxmg_xla_comm_backend.so
```

## Requirements

You need Linux on `x86_64` or `aarch64`, Git, a Docker-compatible container
runtime, and enough storage for the JAX checkout and Bazel cache.

CUDA 12 builds support:

```text
sm_70,sm_80,sm_90,sm_120,compute_90
```

CUDA 13 builds omit Volta:

```text
sm_80,sm_90,sm_120,compute_90
```

## 1. Start the JAX build container

Clone the JAX version pinned by JAXMg and start its CI container:

```bash
git clone --branch jax-v0.10.1 https://github.com/jax-ml/jax.git
cd jax
./ci/utilities/run_docker_container.sh
```

Configure the workspace for the CUDA major version being built:

```bash
CUDA_MAJOR=12
docker exec -e JAXCI_CUDA_VERSION="$CUDA_MAJOR" jax bash -lc '
  source ci/envs/default.env
  source ci/utilities/setup_build_environment.sh
  arch=$(uname -m)
  python build/build.py build \
    --wheels=jax-cuda-plugin \
    --configure_only \
    --bazel_options=--config="ci_linux_${arch}_cuda${JAXCI_CUDA_VERSION}" \
    --python_version="${JAXCI_HERMETIC_PYTHON_VERSION}" \
    --cuda_major_version="${JAXCI_CUDA_VERSION}" \
    --output_path="${JAXCI_OUTPUT_DIR}"'
```

Use `CUDA_MAJOR=13` when building the CUDA 13 backend.

## 2. Copy the JAXMg build inputs

The container mounts the JAX checkout at `/jax`. Copy JAXMg into that checkout
so Bazel can compile it alongside `@xla`:

```bash
JAXMG=/absolute/path/to/jaxmg
JAXCO=/absolute/path/to/jax

mkdir -p "$JAXCO/jaxmg"
cp -r \
  "$JAXMG"/src \
  "$JAXMG"/bazel \
  "$JAXMG"/build_native_backend.sh \
  "$JAXMG"/pyproject.toml \
  "$JAXMG"/setup.py \
  "$JAXCO/jaxmg/"
```

## 3. Compile

Choose the values for the backend:

```bash
CUDA_MAJOR=12
CUSOLVERMP_PACKAGE=nvidia-cusolvermp-cu12==0.9.0.6427
CAPABILITIES=sm_70,sm_80,sm_90,sm_120,compute_90
```

Then run:

```bash
docker exec \
  -e CUDA_MAJOR \
  -e CUSOLVERMP_PACKAGE \
  -e CAPABILITIES \
  jax bash -lc '
    python -m pip install "$CUSOLVERMP_PACKAGE"
    JAX_SRC=/jax \
    JAXMG_ROOT=/jax/jaxmg \
    CUDA_MAJOR="$CUDA_MAJOR" \
    JAXMG_XLA_CUDA_COMPUTE_CAPABILITIES="$CAPABILITIES" \
    JAXMG_XLA_BAZEL_JOBS=$(nproc) \
    /jax/jaxmg/build_native_backend.sh'
```

For CUDA 13, use:

```bash
CUDA_MAJOR=13
CUSOLVERMP_PACKAGE=nvidia-cusolvermp-cu13==0.9.0.6427
CAPABILITIES=sm_80,sm_90,sm_120,compute_90
```

The script discovers cuSOLVERMp, stages the Bazel package, compiles the shared
library, and installs it under `/jax/jaxmg/src/jaxmg/cuXX/`.

Copy the result back to the original checkout:

```bash
cp \
  "$JAXCO/jaxmg/src/jaxmg/cu${CUDA_MAJOR}/libjaxmg_xla_comm_backend.so" \
  "$JAXMG/src/jaxmg/cu${CUDA_MAJOR}/"
```

## 4. Install and test

Install the source tree with the matching runtime extra:

```bash
cd "$JAXMG"
python -m venv .venv
.venv/bin/python -m pip install -e ".[cuda12]"
```

Run the [Cholesky example](../examples/potrs.md) to verify a solver call. Native
contributors should then run the GPU smoke suite described in
[Contributing](contributing.md).
