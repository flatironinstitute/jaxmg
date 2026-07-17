# Installation

The package is available on PyPI. Choose the installation that matches how CUDA
is provided on your system:

1. `pip install "jaxmg[cuda12]"` installs JAX with its NVIDIA CUDA 12 runtime
   wheels.

2. `pip install "jaxmg[cuda12-local]"` installs JAX against an existing local
   CUDA 12 installation.

Support for CUDA 13 is currently under development.

!!! note

    `pip install jaxmg` installs a CPU-only version of JAX. JAXMg is a GPU-only
    package, so it will warn you to install a GPU-compatible version of JAX.

## What the provided binaries support

JAXMg's native backend uses internal XLA communicator interfaces. It is
therefore built with Bazel against the XLA revision associated with a specific
JAX release. The provided binaries currently use:

| Component | Version |
|---|---|
| JAX | `0.10.1` |
| cuSOLVERMp | `nvidia-cusolvermp-cu12==0.8.0.3126` |

The CUDA 12 backend includes native code for these representative GPU
families:

| GPU family | Compute capability | Build target |
|---|---:|---:|
| NVIDIA V100 | 7.0 | `sm_70` |
| NVIDIA A100 | 8.0 | `sm_80` |
| NVIDIA H100/H200 | 9.0 | `sm_90` |
| NVIDIA RTX PRO 6000 Blackwell | 12.0 | `sm_120` |

The backend also includes `compute_90` PTX for forward compatibility. CUDA 13
does not support Volta GPUs, so V100 systems must use CUDA 12.

See [Building from source](technical_details/building_from_source.md) for the
complete native-backend build procedure.

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
