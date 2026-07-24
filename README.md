<div align="center">
    <img src="https://raw.githubusercontent.com/flatironinstitute/jaxmg/main/docs/_static/logo.png" alt="JAXMg" width="300">
</div>

# JAXMg: Distributed dense linear algebra for JAX

[![Docs](https://img.shields.io/badge/docs-site-blue?style=flat-square)](https://flatironinstitute.github.io/jaxmg/)
[![Releases](https://img.shields.io/github/v/release/flatironinstitute/jaxmg?style=flat-square)](https://github.com/flatironinstitute/jaxmg/releases)
[![Build Status](https://jenkins.flatironinstitute.org/job/jaxmg/job/main/lastBuild/badge/icon)](https://jenkins.flatironinstitute.org/job/jaxmg/job/main/)


# JAXMg

JAXMg brings distributed dense linear algebra to JAX, allowing calculations to
scale across multiple GPUs and compute nodes. This enables large-scale matrix
operations far beyond native JAX routines, approaching the combined memory
capacity of the available GPU resources while retaining a familiar JAX
interface.

JAXMg currently provides a jittable API for the following routines:

- [`potrs`](docs/api/potrs.md): Solves the system of linear equations $Ax=B$,
  where $A$ is an $N \times N$ symmetric or Hermitian positive-definite matrix,
  using a Cholesky decomposition. It can also return the log determinant of
  $A$.
- [`lu_solve`](docs/api/lu_solve.md): Solves the system of linear equations
  $Ax=B$, where $A$ is an $N \times N$ general nonsingular matrix, using a
  pivoted LU decomposition.
- [`syevd`](docs/api/syevd.md): Computes the eigenvalues and eigenvectors of an
  $N \times N$ symmetric or Hermitian matrix.

## How JAXMg works

JAXMg connects JAX to NVIDIA's distributed
[cuSOLVERMp](https://docs.nvidia.com/cuda/cusolvermp/) routines through a native
C++/CUDA backend.

Users supply a JAX matrix sharded over a two-dimensional device mesh. JAXMg
handles the local memory-layout conversion, redistribution into cuSOLVERMp's 2D
block-cyclic layout, distributed numerical computation, and restoration of the
result to its original JAX layout.

The operations are implemented using:

- `potrs`: [`cusolverMpPotrf`](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html#cusolvermppotrf)
  and [`cusolverMpPotrs`](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html#cusolvermppotrs)
- `lu_solve`: [`cusolverMpGetrf`](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html#cusolvermpgetrf)
  and [`cusolverMpGetrs`](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html#cusolvermpgetrs)
- `syevd`: [`cusolverMpSyevd`](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html#cusolvermpsyevd)

For more details, see the [API reference](docs/api/index.md) and the
[accompanying paper](https://arxiv.org/abs/2601.14466).

## Installation

Install the package with the extra matching your CUDA setup:

| CUDA setup | Command |
|---|---|
| CUDA 12 runtime wheels | `pip install "jaxmg[cuda12]"` |
| Local CUDA 12 installation | `pip install "jaxmg[cuda12-local]"` |
| CUDA 13 runtime wheels | `pip install "jaxmg[cuda13]"` |
| Local CUDA 13 installation | `pip install "jaxmg[cuda13-local]"` |

Prebuilt Linux wheels are provided for `x86_64` and `aarch64`. The CUDA 12
backend supports NVIDIA V100, A100, H100/H200, and RTX PRO 6000 Blackwell GPUs.
The CUDA 13 backend supports the same families except V100, which requires CUDA
12. The binaries use JAX `0.10.1` and cuSOLVERMp `0.8.0.3126`.

`pip install jaxmg` installs CPU-only JAX, so JAXMg users should select one of
the CUDA extras above. See [Installation](https://flatironinstitute.github.io/jaxmg/install/)
for details.

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
