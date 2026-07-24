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

## Supported systems

Prebuilt Linux wheels are provided for `x86_64` and `aarch64`. The supported
NVIDIA GPU families are:

| CUDA setup | Supported GPUs |
|---|---|
| CUDA 12 | V100, A100, H100/H200, and Blackwell GPUs |
| CUDA 13 | A100, H100/H200, and Blackwell GPUs |

The binaries use JAX `0.10.1` and cuSOLVERMp `0.8.0.3126`. See
[Building from source](technical_details/building_from_source.md) for the native
build procedure.

## Runtime requirements

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
