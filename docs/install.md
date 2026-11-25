# Installation

Install the package with the extra matching your CUDA setup:

| CUDA setup | Command |
|---|---|
| CUDA 12 | `pip install "jaxmg[cuda12]"` |
| Local CUDA 12 | `pip install "jaxmg[cuda12-local]"` |
| CUDA 13 | `pip install "jaxmg[cuda13]"` |
| Local CUDA 13 | `pip install "jaxmg[cuda13-local]"` |

!!! note

    `pip install jaxmg` installs a CPU-only version of JAX. JAXMg is a GPU-only
    package, so it will warn you to install a GPU-compatible version of JAX.

## JAX compatibility

This development branch uses the public Collectives C FFI, available from JAX
`0.11.2`, without linking XLA. Compatibility testing covers JAX `0.11.2`,
`0.12.0.1`, and `0.12.1`. Released versions use the following JAX ranges:

| JAXMg | JAX |
|---|---|
| `1.0.0` to `1.1.1` | `0.10.1` |
| `1.2.0` | `0.11.0` |
| `1.3.0` | `0.11.1` |
| `1.4.0` to `1.4.2` | `0.11.1`, `0.11.2`, `0.12.0`, `0.12.0.1` |

## Older releases

Use the version selector next to the site title to switch between releases.

Documentation for JAXMg `0.0.9` is archived at
<https://flatironinstitute.github.io/jaxmg/0.0.9/>. That was the last release
built on NVIDIA's single-node [cuSolverMg](https://docs.nvidia.com/cuda/cusolver/index.html#using-the-cuSolverMg-api)
API, so its interface differs substantially from this one: it provides `potri`
and the `cyclic_1d` layout helpers, has no `lu_solve`, and distributes matrices
with a 1D block-cyclic layout rather than the 2D block-cyclic layout used here.

## Supported systems

Prebuilt Linux wheels are provided for `x86_64` and `aarch64`. The supported
NVIDIA GPU families are:

| CUDA setup | Supported GPUs |
|---|---|
| CUDA 12 | V100, A100, H100/H200, and Blackwell GPUs |
| CUDA 13 | A100, H100/H200, and Blackwell GPUs |

The prebuilt binaries use cuSOLVERMp `0.10.0.27368`. See
[Building from source](technical_details/building_from_source.md) for the native
build procedure.

## Runtime requirements

CUDA 13 installations require CUDA 13.0 Update 2 or newer and cuBLAS 13.1 or
newer. Use NCCL 2.32.6 or newer with either CUDA version. The `cuda12`
and `cuda13` extras install dependencies satisfying these requirements
automatically.

For local installations using older NCCL, set `NCCL_NVLS_ENABLE=0` to avoid
the NVLS correctness issue documented by
[NVIDIA](https://docs.nvidia.com/cuda/cusolvermp/release_notes/index.html).

cuSOLVERMp requires a one-to-one mapping between processes and GPUs. Multi-GPU
and multi-node JAXMg jobs must therefore be launched with one Python process per
GPU and initialized with
`jax.distributed.initialize()` before the global device mesh is constructed.
See [Distributed execution](examples/execution.md).

For large solves close to the GPU memory limit, CUDA Virtual Memory Management
can improve allocator behaviour. Enable it by setting these variables before
Python starts:

```bash
export XLA_PYTHON_CLIENT_ALLOCATOR=vmm
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.99
```
