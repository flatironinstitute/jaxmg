<div align="center">
    <img src="https://raw.githubusercontent.com/flatironinstitute/jaxmg/main/docs/_static/logo.png" alt="JAXMg" width="300">
</div>

# JAXMg: A multi-GPU linear solver in JAX

[![Docs](https://img.shields.io/badge/docs-site-blue?style=flat-square)](https://flatironinstitute.github.io/jaxmg/)
[![PyPI](https://img.shields.io/pypi/v/jaxmg?style=flat-square)](https://pypi.org/project/jaxmg/)
[![Releases](https://img.shields.io/github/v/release/flatironinstitute/jaxmg?style=flat-square)](https://github.com/flatironinstitute/jaxmg/releases)
[![Build](https://github.com/flatironinstitute/jaxmg/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/flatironinstitute/jaxmg/actions/workflows/ci.yml)
[![Tests](https://jenkins-new.flatironinstitute.org/job/CCQ/job/jaxmg/job/main/lastBuild/badge/icon)](https://jenkins-new.flatironinstitute.org/job/CCQ/job/jaxmg/job/main/)


# JAXMg

JAXMg brings distributed matrix solvers to JAX, allowing calculations to run
across multiple GPUs and nodes. This enables solvers to scale to matrices near
the combined memory capacity of the available GPUs, far beyond native JAX
routines, while retaining a familiar JAX interface.

JAXMg currently provides a jittable API for the following routines:

- [`potrs`](https://flatironinstitute.github.io/jaxmg/latest/api/potrs/): Solves the system of linear equations $Ax=B$,
  where $A$ is an $N \times N$ symmetric or Hermitian positive-definite matrix,
  using a Cholesky decomposition. It can also return the log determinant of
  $A$.
- [`lu_solve`](https://flatironinstitute.github.io/jaxmg/latest/api/lu_solve/): Solves the system of linear equations
  $Ax=B$, where $A$ is an $N \times N$ general nonsingular matrix, using a
  pivoted LU decomposition.
- [`syevd`](https://flatironinstitute.github.io/jaxmg/api/syevd/): Computes the
  eigenvalues $\lambda_i$ and optional eigenvectors $v_i$ of an $N \times N$
  symmetric or Hermitian matrix $A$, satisfying $Av_i=\lambda_i v_i$.
- [`gesvd`](https://flatironinstitute.github.io/jaxmg/latest/api/gesvd/): Computes
  the singular-value decomposition of an $M \times N$ matrix
  $A$ ($A = U \Sigma V^{\dagger}$), returning the singular values and optional
  left and right singular vectors.

## How JAXMg works

JAXMg connects JAX to NVIDIA's distributed
[cuSOLVERMp](https://docs.nvidia.com/cuda/cusolvermp/) routines through a native
C++/CUDA backend.

Supply a JAX matrix sharded over a one- or two-axis device mesh, and JAXMg
handles the local memory-layout conversion, redistribution into cuSOLVERMp's 2D
block-cyclic layout, distributed numerical computation, and restoration of the
result to its original JAX layout. The matrix data remains GPU-resident
throughout, with in-place transformations and bounded scratch storage minimizing
memory overhead.

The operations are implemented using:

- `potrs`: [`cusolverMpPotrf`](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html#cusolvermppotrf)
  and [`cusolverMpPotrs`](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html#cusolvermppotrs)
- `lu_solve`: [`cusolverMpGetrf`](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html#cusolvermpgetrf)
  and [`cusolverMpGetrs`](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html#cusolvermpgetrs)
- `syevd`: [`cusolverMpSyevd`](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html#cusolvermpsyevd)
- `gesvd`: [`cusolverMpGesvd`](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html#cusolvermpgesvd)

For more details, see the [API reference](https://flatironinstitute.github.io/jaxmg/latest/api/) and the
[accompanying paper](https://arxiv.org/abs/2601.14466).

> **Deprecation notice:** JAXMg 0.x, based on NVIDIA's deprecated cuSOLVERMg
> backend, has been superseded by JAXMg 1.0. Version 1.0 requires one Python
> process per GPU and removes `potri`; the previous API remains available with
> `pip install "jaxmg<1"`. See the
> [0.0.9 release](https://github.com/flatironinstitute/jaxmg/releases/tag/v0.0.9)
> and [0.0.9 documentation](https://flatironinstitute.github.io/jaxmg/0.0.9/).

## Installation

Install the package with the extra matching your CUDA setup:

| CUDA setup | Command |
|---|---|
| CUDA 12 | `pip install "jaxmg[cuda12]"` |
| Local CUDA 12 | `pip install "jaxmg[cuda12-local]"` |
| CUDA 13 | `pip install "jaxmg[cuda13]"` |
| Local CUDA 13 | `pip install "jaxmg[cuda13-local]"` |

Prebuilt Linux wheels are provided for `x86_64` and `aarch64`. The supported
NVIDIA GPU families are:

| CUDA setup | Supported GPUs |
|---|---|
| CUDA 12 | V100, A100, H100/H200, and Blackwell GPUs |
| CUDA 13 | A100, H100/H200, and Blackwell GPUs |

> **Note:** `pip install jaxmg` installs CPU-only JAX. Select one of the CUDA
> extras above when installing JAXMg for GPU use. See
> [Installation](https://flatironinstitute.github.io/jaxmg/latest/install/) for
> details.

## Example

JAXMg runs with one Python process per GPU. After launching one process for each
GPU, initialize distributed JAX before constructing the device mesh. See
[Distributed execution](https://flatironinstitute.github.io/jaxmg/latest/examples/execution/)
for launch details.

For JAX arrays `A` and `b`, a Cholesky solve and log-determinant calculation
requires only the distributed mesh, array placement, and solver call:

```python
import jax
from jax.sharding import NamedSharding, PartitionSpec as P
from jaxmg import potrs


jax.distributed.initialize()

mesh = jax.make_mesh((jax.process_count(), 1), ("pr", "pc"))
jax.set_mesh(mesh)

A = jax.device_put(A, NamedSharding(mesh, P("pr", "pc")))
b = jax.device_put(b, NamedSharding(mesh, P("pr", None)))

x, logdet = potrs(A, b, T_A=256, return_logdet=True)
```

<details>
<summary>Complete distributed example with validation</summary>

The following example constructs a diagonal system, distributes it over the
available GPUs, solves it, and checks the result:

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
jax.set_mesh(mesh)
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

</details>

Call `potrs` or `lu_solve` directly for standard solves. For use inside a larger
`jax.jit`-compiled function, see the advanced
[Cholesky](https://flatironinstitute.github.io/jaxmg/latest/examples/potrs/) and
[LU](https://flatironinstitute.github.io/jaxmg/latest/examples/lu_solve/) examples.

## Projects that use JAXMg

- [JAXMg Benchmarks](https://github.com/therooler/jaxmg_benchmark): Benchmarks for various Multi-GPUs setups.
- [JAXMg + Netket](https://github.com/therooler/netket_jaxmg): Implementation of the MinSR Netket driver that uses JAXMg for inverting the S-matrix. Tested on Multi-node settings.
- [JAXMg for blurred sampling](https://github.com/therooler/nqs_blurred_sampling): Implementation of t-VMC that makes use JAXMg for inverting the QGT.

## Citations
```
@misc{2601.14466,
Author = {Jacob Tutt and Roeland Wiersema},
Title = {JAXMg: A multi-GPU linear solver in JAX},
Year = {2026},
Eprint = {arXiv:2601.14466},
}
```

## Acknowledgements
We acknowledge support from the Flatiron Institute. The Flatiron Institute is a
division of the Simons Foundation.
