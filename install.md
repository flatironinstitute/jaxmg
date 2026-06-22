# Building the native backend (workstation notes)

These are the steps to compile `libjaxmg_xla_comm_backend.so` against XLA (`@xla`)
inside the official JAX CI container (`tensorflow/ml-build:latest`) on a Flatiron
workstation, then install and smoke-test the package. See `CONTRIBUTING.md` for the
general (non-site-specific) flow.

## 0. Load modules

```bash
module load docker
```

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
JAXMG=/path/to/jaxmg            # this repo
JAXCO=/path/to/jax              # the jax checkout (mounted at /jax)
mkdir -p "$JAXCO/jaxmg"
cp -r "$JAXMG"/{src,bazel,build_native_backend.sh,pyproject.toml,setup.py} "$JAXCO/jaxmg/"
```

## 3. Compile

Inside the container, install cuSOLVERMp (for headers/lib discovery) and run the
build. The container has no system `nvcc`, so `CUDA_MAJOR` must be set explicitly.
The arch list builds broadly (Volta→Hopper SASS + PTX); `sm_80` SASS is forward-
compatible to Ada (sm_89) workstation GPUs.

```bash
docker exec jax bash -lc '
  python -m pip install nvidia-cusolvermp-cu12==0.8.0.3126 &&
  JAX_SRC=/jax JAXMG_ROOT=/jax/jaxmg CUDA_MAJOR=12 \
  JAXMG_XLA_CUDA_COMPUTE_CAPABILITIES=sm_70,sm_80,sm_90,compute_90 \
  JAXMG_XLA_BAZEL_JOBS=$(nproc) \
  /jax/jaxmg/build_native_backend.sh'
```

This installs the library to `<jax-checkout>/jaxmg/src/jaxmg/cu12/libjaxmg_xla_comm_backend.so`.
Copy it back into this repo:

```bash
cp "$JAXCO/jaxmg/src/jaxmg/cu12/libjaxmg_xla_comm_backend.so" "$JAXMG/src/jaxmg/cu12/"
```

Sanity checks (host, needs `cuda/12.8`):

```bash
cuobjdump --list-elf src/jaxmg/cu12/libjaxmg_xla_comm_backend.so   # expect sm_70/sm_80/sm_90
nm -D src/jaxmg/cu12/libjaxmg_xla_comm_backend.so | grep XlaCusolverMp  # 4 *FFI symbols exported
```

## 4. Install and smoke-test

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[cuda12]"   # jax[cuda12]==0.10.1 + cuSOLVERMp + jaxmg
```

Single-GPU functional check (one process, 1×1 mesh):

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
