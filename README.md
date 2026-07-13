<div align="center">
    <img src="https://raw.githubusercontent.com/flatironinstitute/jaxmg/main/docs/_static/logo.png" alt="JAXMg" width="300">
</div>

#  JAXMg: A multi-GPU linear solver in JAX

[![Docs](https://img.shields.io/badge/docs-site-blue?style=flat-square)](https://flatironinstitute.github.io/jaxmg/)
[![Releases](https://img.shields.io/github/v/release/flatironinstitute/jaxmg?style=flat-square)](https://github.com/flatironinstitute/jaxmg/releases)
[![Build Status](https://jenkins.flatironinstitute.org/job/jaxmg/job/main/lastBuild/badge/icon)](https://jenkins.flatironinstitute.org/job/jaxmg/job/main/)


# JAXMg
JAXMg provides a jittable C++/CUDA interface between [JAX](https://github.com/jax-ml/jax) and [cuSOLVERMp](https://docs.nvidia.com/cuda/cusolvermp/), NVIDIA's distributed linear algebra runtime. The public API exposes three fused cuSOLVERMp routines:

- [cusolverMpPotrf/Potrs](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html): solves symmetric (Hermitian) positive-definite systems on a 2D process grid via `jaxmg.potrs`.
- [cusolverMpGetrf/Getrs](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html): solves general nonsingular systems on a 2D process grid via `jaxmg.lu_solve`.
- [cusolverMpSyevd](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html): computes eigenvalues and eigenvectors on a 2D process grid via `jaxmg.syevd`.

All three routines accept ordinary 2D JAX-sharded arrays, locally pad shards when needed, enter one fused native FFI call, redistribute to cuSOLVERMp's 2D block-cyclic layout, call cuSOLVERMp, and redistribute results back to the JAX-facing layout.

For more details, see the [API](docs/api/index.md).

## Installation

The package is available on PyPI and can be installed with

```bash
pip install jaxmg[cuda12]
```

This installs a GPU-compatible version of JAX.

1. `pip install "jaxmg[cuda12]"`: Install JAX with its CUDA 12 runtime wheels.

2. `pip install "jaxmg[cuda12-local]"`: Use an existing local CUDA 12 installation.

JAXMg currently distributes a CUDA 12 cuSOLVERMp backend. CUDA 13 packaging
requires a matching cuSOLVERMp CUDA 13 development distribution and is not
part of the current release.

The provided binaries are compiled with

|**JAXMg** | **CUDA** | **cuDNN** |
|---|---|---| 
| `cuda12`,`cuda12-local` | 12.8.0 | 9.17.1.4|

For large cuSOLVERMp runs close to the GPU memory limit, set JAX's allocator
policy before launching Python:

```bash
export XLA_PYTHON_CLIENT_ALLOCATOR=vmm
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.99
```

These variables are intentionally launch-time settings rather than package
imports, because they affect JAX's whole GPU allocator for the process.

The wheel contains the Bazel-built `libjaxmg_xla_comm_backend.so` native
backend. Installing a wheel does not run Bazel. Details for compiling from
source can be found in the
[native-backend build guide](https://flatironinstitute.github.io/jaxmg/technical_details/building_from_source/).

> **Note:** `pip install jaxmg` installs a CPU-only version of JAX. Since JAXMg
> is a GPU-only package, install one of the CUDA extras shown above.

## Example

JAXMg's cuSOLVERMp backend uses one Python process per GPU. A minimal
rank-per-GPU Cholesky solve is:

```python
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding
from jaxmg import potrs


jax.distributed.initialize()

T_A = 3
dtype = jnp.float64

num_procs = jax.process_count()
N = T_A * num_procs

A = jnp.diag(jnp.arange(N, dtype=dtype) + 1)
b = jnp.ones((N, 1), dtype=dtype)

mesh = jax.make_mesh((num_procs, 1), ("pr", "pc"))
sharding = NamedSharding(mesh, P("pr", "pc"))
A = jax.device_put(A, sharding)
b = jax.device_put(b, sharding)

out = potrs(A, b, T_A=T_A)
out.block_until_ready()

expected_out = 1.0 / (jnp.arange(N, dtype=dtype) + 1)
is_correct = jnp.allclose(out.flatten(), expected_out)
is_correct.block_until_ready()

if jax.process_index() == 0:
    print(out)
    print(is_correct)
```
On four ranks, this gives
```bash
[[1.        ]
 [0.5       ]
 [0.33333333]
 [0.25      ]
 [0.2       ]
 [0.16666667]
 [0.14285714]
 [0.125     ]
 [0.11111111]
 [0.1       ]
 [0.09090909]
 [0.08333333]]
True
```
as expected.

Use `potrs` or `lu_solve` for a direct solve. When the solve must instead be
embedded inside a larger function compiled by the application, use
`potrs_shardmap_ctx` or `lu_solve_shardmap_ctx`. These context interfaces run
the same native solver pipelines but leave the outer `jax.jit` boundary to the
caller and expose the donated input-matrix work buffer so `donate_argnums` can
be used safely.

## Projects that use JAXMg

- [JAXMg Benchmarks](https://github.com/therooler/jaxmg_benchmark): Benchmarks for various Multi-GPUs setups.
- [JAXMg + Netket](https://github.com/therooler/netket_jaxmg): Implementation of the MinSR Netket driver that uses JAXMg for inverting the S-matrix. Tested on Multi-node settings.
- [JAXMg for blurred sampling](https://github.com/therooler/nqs_blurred_sampling): Implementation of t-VMC that makes use JAXMg for inverting the QGT.

## cuSOLVERMp
The cuSOLVERMp backend uses JAX's XLA-owned NCCL communicator through FFI,
redistributes ordinary 2D JAX-sharded matrices into cuSOLVERMp's 2D
block-cyclic layout, and calls cuSOLVERMp directly. JAXMg supports Cholesky
solves, LU solves, and symmetric/Hermitian eigensolves.

## Citations
```
@misc{2601.14466,
Author = {Roeland Wiersema},
Title = {JAXMg: A multi-GPU linear solver in JAX},
Year = {2026},
Eprint = {arXiv:2601.14466},
}
```

## Acknowledgements
I acknowledge support from the Flatiron Institute. The Flatiron Institute is a
division of the Simons Foundation.
