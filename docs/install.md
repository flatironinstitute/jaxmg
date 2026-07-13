# Installation

The package is available on PyPI and can be installed with

```bash
python -m pip install "jaxmg[cuda12]"
```

This installs the CUDA 12 build of JAX, the cuSOLVERMp runtime, and a JAXMg
wheel containing the compiled native backend.

1. `python -m pip install "jaxmg[cuda12]"` installs the NVIDIA CUDA runtime
   wheels required by JAX.

2. `python -m pip install "jaxmg[cuda12-local]"` uses an existing local CUDA
   12 installation for JAX.

JAXMg currently distributes a CUDA 12 cuSOLVERMp backend. CUDA 13 packaging
requires a matching cuSOLVERMp CUDA 13 development distribution and is not
part of the current release.

!!! note
    Installing `jaxmg` without a CUDA extra installs CPU-only JAX. JAXMg's
    numerical routines require NVIDIA GPUs.

## What the wheel contains

JAXMg's native backend uses internal XLA communicator interfaces. The shared
library must therefore be built against the XLA revision associated with the
packaged JAX version. The release build currently uses:

| Component | Version or source |
|---|---|
| JAX | `0.10.1` |
| XLA | the revision selected by the `jax-v0.10.1` checkout |
| cuSOLVERMp | `nvidia-cusolvermp-cu12==0.8.0.3126` |
| Native library | `libjaxmg_xla_comm_backend.so` |

Installing a published wheel does not invoke Bazel. Bazel is used when the
wheel is produced: it builds the native backend inside the JAX build
environment, links it against the matching XLA and cuSOLVERMp interfaces, and
packages the resulting shared library into the wheel.

See [Contributing](https://github.com/flatironinstitute/jaxmg/blob/main/CONTRIBUTING.md#build-from-source)
for the complete source-build procedure.

## Runtime requirements

Every JAXMg process must see exactly one GPU. Multi-GPU and multi-node jobs must
be launched with one Python process per GPU and initialized with
`jax.distributed.initialize()` before the global device mesh is constructed.
See [Distributed execution](execution.md).

For large solves close to the GPU memory limit, CUDA Virtual Memory Management
can improve allocator behaviour:

```bash
export XLA_PYTHON_CLIENT_ALLOCATOR=vmm
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.99
```

Set these variables before Python starts. Systems whose driver does not support
the VMM allocator should use:

```bash
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
```

Some older driver and container combinations also require:

```bash
export NCCL_CUMEM_ENABLE=0
```

This disables NCCL's cuMem-based allocation path; it does not change JAXMg's
solver or redistribution algorithms.
