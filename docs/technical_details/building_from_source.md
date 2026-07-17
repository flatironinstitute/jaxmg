# Building the Native Backend

These are the steps to compile `libjaxmg_xla_comm_backend.so` against XLA (`@xla`)
inside the official JAX CI container (`tensorflow/ml-build:latest`).

`libjaxmg_xla_comm_backend.so` is a C++/CUDA shared library that links against
XLA's internal C++ API (the FFI bindings, GPU collectives, and communicator
cliques). That API is **not** shipped as a normal library — it is only available
as the Bazel external repo `@xla`, which is defined inside a **JAX source
checkout**.

To ensure maximum compatibility with the current state of JAX, the build runs inside the official JAX CI Docker container
(`ml-build:latest`), which provides Bazel and the hermetic CUDA toolchain. The
container bind-mounts a jax checkout at `/jax`, and `@xla` resolves to the XLA
revision pinned by that jax version.

For JAXMg, `build_native_backend.sh` orchestrates the build:

1. **Discover cuSOLVERMp** headers/library from the installed
   `nvidia-cusolvermp-cuXX` wheel (or from `JAXMG_CUSOLVERMP_*` overrides).
2. **Stage the sources** as a Bazel package: it copies the C++/CUDA sources
   (`src/cuda/...`) and the `bazel/jaxmg_backend.BUILD.bazel` template from
   `JAXMG_ROOT` into `$JAX_SRC/jaxmg_backend/` inside the jax tree, substituting
   the CUDA major version into the BUILD file. Bazel can only see `@xla` and the
   hermetic CUDA repos from inside the jax workspace, which is why the sources
   are copied in rather than built in place.
3. **Compile** with `bazel build //jaxmg_backend:libjaxmg_xla_comm_backend.so`,
   linking the hermetic CUDA libs (cudart, cusolver, cuda driver stub) plus the
   externally-provided `libcusolverMp.so`, and building for a range of GPU
   architectures.
4. **Install** the resulting `.so` back into `JAXMG_ROOT/src/jaxmg/cuXX/`.

At runtime, the Python layer (`_setup.py`) loads this `.so` with `ctypes`,
registers its exported XLA FFI targets with JAX, and resolves plain `extern "C"`
helper symbols (e.g. the VMM-support query) directly.

## 1. JAX CI container

The build needs a `jax` checkout — that is where `@xla` is defined as a Bazel
external repo. Clone it (at a tag compatible with the `jax==0.10.1` pin in
`pyproject.toml`), start the container, and generate the CUDA/Bazel config:

```bash
git clone --branch jax-v0.10.1 https://github.com/jax-ml/jax.git
cd jax
./ci/utilities/run_docker_container.sh                  # starts a container named "jax", mounts this dir at /jax
docker exec jax ./ci/build_artifacts.sh jax-cuda-plugin # writes /jax/.jax_configure.bazelrc
```

If a `jax` container is already running, you can reuse it (`docker ps`); skip the
clone/run/build_artifacts steps as long as `/jax/.jax_configure.bazelrc` exists.

## 2. Stage the jaxmg sources into the jax checkout

`build_native_backend.sh` builds a Bazel package inside the jax tree, so the jaxmg
sources must be visible under the mounted `/jax`. Copy this repo's build inputs into
`<jax-checkout>/jaxmg`:

```bash
JAXMG=/path/to/jaxmg            # this repo (no relative paths)
JAXCO=/path/to/jax              # the jax checkout (mounted at /jax)
mkdir -p "$JAXCO/jaxmg"
cp -r "$JAXMG"/{src,bazel,build_native_backend.sh,pyproject.toml,setup.py} "$JAXCO/jaxmg/"
```

## 3. Compile

Inside the container, install cuSOLVERMp (for headers/lib discovery) and run the
build. The container has no system `nvcc`, so `CUDA_MAJOR` must be set explicitly.
The architecture list builds native SASS for Volta, Ampere, Hopper, and RTX
Blackwell GPUs, with PTX retained for forward compatibility. The `sm_80` SASS
is also compatible with Ada (`sm_89`) workstation GPUs.

```bash
docker exec jax bash -lc '
  python -m pip install nvidia-cusolvermp-cu12==0.8.0.3126 &&
  JAX_SRC=/jax JAXMG_ROOT=/jax/jaxmg CUDA_MAJOR=12 \
  JAXMG_XLA_CUDA_COMPUTE_CAPABILITIES=sm_70,sm_80,sm_90,sm_120,compute_90 \
  JAXMG_XLA_BAZEL_JOBS=$(nproc) \
  /jax/jaxmg/build_native_backend.sh'
```

This installs the library to `<jax-checkout>/jaxmg/src/jaxmg/cu12/libjaxmg_xla_comm_backend.so`.
Copy it back into this repo:

```bash
cp "$JAXCO/jaxmg/src/jaxmg/cu12/libjaxmg_xla_comm_backend.so" "$JAXMG/src/jaxmg/cu12/"
```

## 4. Install and smoke-test

From the root, install the package
```bash
cd $JAXMG
python -m venv .venv
.venv/bin/python -m pip install -e ".[cuda12]"
```

For a single-GPU functional check, save the following as `smoke.py`:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding
from jaxmg import potrs

jax.distributed.initialize(coordinator_address="localhost:12399", num_processes=1, process_id=0)

T_A = 3
N = T_A * jax.process_count()
A = jnp.diag(jnp.arange(N, dtype=jnp.float64) + 1)
b = jnp.ones((N, 1), dtype=jnp.float64)
mesh = jax.make_mesh((jax.process_count(), 1), ("pr", "pc"))
sh = NamedSharding(mesh, P("pr", "pc"))
out = potrs(jax.device_put(A, sh), jax.device_put(b, sh), T_A=T_A)
print(out.flatten())   # -> [1. 0.5 0.333...]
```

Run with one GPU visible:

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda .venv/bin/python smoke.py
```

A trailing `WatchTasks ... CANCELLED` gRPC message and a "donated buffers were not
usable" warning at exit are harmless.
