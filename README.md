<div align="center">
    <img src="https://raw.githubusercontent.com/flatironinstitute/jaxmg/main/docs/_static/logo.png" alt="JAXMg" width="300">
</div>

#  JAXMg: A multi-GPU linear solver in JAX

[![Docs](https://img.shields.io/badge/docs-site-blue?style=flat-square)](https://flatironinstitute.github.io/jaxmg/)
[![Releases](https://img.shields.io/github/v/release/flatironinstitute/jaxmg?style=flat-square)](https://github.com/flatironinstitute/jaxmg/releases)
[![Build Status](https://jenkins.flatironinstitute.org/job/jaxmg/job/main/lastBuild/badge/icon)](https://jenkins.flatironinstitute.org/job/jaxmg/job/main/)


# JAXMg

JAXMg provides a C++/CUDA interface between
[JAX](https://github.com/jax-ml/jax) and
[cuSOLVERMp](https://docs.nvidia.com/cuda/cusolvermp/), NVIDIA's distributed
multi-GPU linear algebra library. We provide a jittable API for the following
routines:

- [`potrs`](docs/api/potrs.md): Solves the system of linear equations $Ax=B$,
  where $A$ is an $N\times N$ symmetric (Hermitian) positive-definite matrix,
  via a Cholesky decomposition
  ([`cusolverMpPotrf`](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html#cusolvermppotrf)
  and [`cusolverMpPotrs`](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html#cusolvermppotrs)).
- [`lu_solve`](docs/api/lu_solve.md): Solves the system of linear equations
  $Ax=B$, where $A$ is an $N\times N$ general nonsingular matrix, via a pivoted
  LU decomposition
  ([`cusolverMpGetrf`](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html#cusolvermpgetrf)
  and [`cusolverMpGetrs`](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html#cusolvermpgetrs)).
- [`syevd`](docs/api/syevd.md): Computes the eigenvalues and eigenvectors of an
  $N\times N$ symmetric (Hermitian) matrix
  ([`cusolverMpSyevd`](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html#cusolvermpsyevd)).

## What this allows you to do

Pass JAXMg an ordinary JAX matrix sharded over a two-dimensional device mesh.
JAXMg then handles the native local-memory conversion, global redistribution
into cuSOLVERMp's 2D block-cyclic layout, distributed solver execution, and
restoration of the result.

This allows your JAX linear algebra calculation to scale across multiple GPUs
and nodes, supporting matrices far beyond the memory and computational limits
of native JAX implementations.

For more details, see the [API reference](docs/api/index.md) and the
[accompanying paper](https://arxiv.org/abs/2601.14466).

## Installation

The package is available on PyPI. Choose the installation that matches how CUDA
is provided on your system:

1. `pip install "jaxmg[cuda12]"` installs JAX with its NVIDIA CUDA 12 runtime
   wheels.

2. `pip install "jaxmg[cuda12-local]"` installs JAX against an existing local
   CUDA 12 installation.

`pip install jaxmg` installs a CPU-only version of JAX. JAXMg is a GPU-only
package, so install one of the CUDA extras shown above.

The native backend uses internal XLA communicator interfaces. It is therefore
built with Bazel against the XLA revision associated with a specific JAX
release. The provided binaries currently use JAX `0.10.1` and
`nvidia-cusolvermp-cu12==0.8.0.3126`. See the
[native-backend build guide](https://flatironinstitute.github.io/jaxmg/technical_details/building_from_source/)
for the complete build procedure.

The CUDA 12 backend contains native code for the following representative GPU
families:

| GPU family | Compute capability | Build target |
|---|---:|---:|
| NVIDIA V100 | 7.0 | `sm_70` |
| NVIDIA A100 | 8.0 | `sm_80` |
| NVIDIA H100/H200 | 9.0 | `sm_90` |
| NVIDIA RTX PRO 6000 Blackwell | 12.0 | `sm_120` |

The wheel also retains `compute_90` PTX for forward compatibility. CUDA 13
builds do not support Volta GPUs; V100 systems must use the CUDA 12 package.

cuSOLVERMp requires a one-to-one mapping between processes and GPUs. Multi-GPU
and multi-node jobs must therefore be launched with one Python process per GPU
and initialized with `jax.distributed.initialize()` before the global device
mesh is constructed. See the
[distributed execution example](https://flatironinstitute.github.io/jaxmg/examples/execution/).

For large solves close to the GPU memory limit, CUDA Virtual Memory Management
can improve allocator behaviour. Enable it before Python starts:

```bash
export XLA_PYTHON_CLIENT_ALLOCATOR=vmm
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.99
```

## Example

A minimal rank-per-GPU Cholesky solve and log-determinant calculation is:

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
a_sharding = NamedSharding(mesh, P("pr", "pc"))
b_sharding = NamedSharding(mesh, P("pr", None))
A = jax.device_put(A, a_sharding)
b = jax.device_put(b, b_sharding)

out, logdet = potrs(A, b, T_A=T_A, return_logdet=True)
out.block_until_ready()
logdet.block_until_ready()

expected_out = 1.0 / (jnp.arange(N, dtype=dtype) + 1)
expected_logdet = jnp.sum(jnp.log(jnp.arange(N, dtype=dtype) + 1))
is_correct = jnp.allclose(out.flatten(), expected_out) & jnp.allclose(
    logdet, expected_logdet
)
is_correct.block_until_ready()

if jax.process_index() == 0:
    print(out)
    print(logdet)
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
19.987214495661885
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
