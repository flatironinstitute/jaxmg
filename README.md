<div align="center">
    <img src="https://raw.githubusercontent.com/therooler/jaxmg/main/docs/_static/logo.png" alt="Jaxmg" width="300">
</div>

#  JAXMg: A multi-GPU linear solver in JAX

[![Docs](https://img.shields.io/badge/docs-site-blue?style=flat-square)](https://flatironinstitute.github.io/jaxmg/)
[![Releases](https://img.shields.io/github/v/release/therooler/jaxmg?style=flat-square)](https://github.com/therooler/jaxmg/releases)
[![Build Status](https://jenkins.flatironinstitute.org/job/jaxmg/job/main/lastBuild/badge/icon)](https://jenkins.flatironinstitute.org/job/jaxmg/job/main/)


# JAXMg
JAXMg provides a C++ interface between [JAX](https://github.com/google/jax), [cuSolverMg](https://docs.nvidia.com/cuda/cusolver/index.html#using-the-cuSolverMg-api), and the newer [cuSolverMp](https://docs.nvidia.com/cuda/cusolvermp/) distributed linear algebra runtime. We provide a jittable API for the following routines.

- [cusolverMgPotrs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolvermgpotrs-deprecated): Solves the system of linear equations: $Ax=b$ where $A$ is an $N\times N$ symmetric (Hermitian) positive-definite matrix via a Cholesky decomposition 
- [cusolverMgSyevd](https://docs.nvidia.com/cuda/cusolver/index.html#cusolvermgsyevd-deprecated): Computes eigenvalues and eigenvectors of an $N\times N$ symmetric (Hermitian) matrix.
- [cusolverMpPotrf/Potrs](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html): Solves symmetric (Hermitian) positive-definite systems on a 2D process grid via `jaxmg.potrs_mp`.
- [cusolverMpSyevd](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html): Computes eigenvalues and eigenvectors on a 2D process grid via `jaxmg.syevd_mp`.

For more details, see the [API](api/potrs.md).

## Installation

The package is available on PyPi and can be installed with

```bash
pip install jaxmg[cuda12]
```

This will install a GPU compatible version of JAX. 

1. `pip install "jaxmg[cuda12]"`: Use CUDA 12 (only works for `jax>=0.6.2`).

2. `pip install "jaxmg[cuda12-local]"`: Use locally available CUDA 12 installation.

3. `pip install "jaxmg[cuda13]"`: Use CUDA 13 (only works for `jax>=0.7.2`).

4. `pip install "jaxmg[cuda13-local]"`: Use locally available CUDA 13 installation.

The provided binaries are compiled with

|**JAXMg** | **CUDA** | **cuDNN** |
|---|---|---| 
| `cuda12`,`cuda12-local` | 12.8.0 | 9.17.1.4|
| `cuda13`,`cuda13-local` | 13.0.0 | 9.17.1.4|

Details for compiling the from source code can be found in `CONTRIBUTING.md`.

> **_Note:_** `pip install jaxmg` will install a CPU-only version of JAX. Since `jaxmg` is a GPU-only package you will receive a warning to install a GPU-compatible version of jax. 

## Example

A minimal example that runs the code is:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding
from jaxmg import potrs
print(f"Devices: {jax.devices()}")
# Assumes we have at least one GPU available
devices = jax.devices("gpu")
N = 12
T_A = 3
dtype = jnp.float64
# Create diagonal matrix and `b` all equal to one
A = jnp.diag(jnp.arange(N, dtype=dtype) + 1)
b = jnp.ones((N, 1), dtype=dtype)
ndev = len(devices)
# Make mesh and place data (rows sharded)
mesh = jax.make_mesh((ndev,), ("x",))
A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
b = jax.device_put(b, NamedSharding(mesh, P(None, None)))
# Call potrs
out = potrs(A, b, T_A=T_A, mesh=mesh, in_specs=(P("x", None), ))
print(out)
expected_out = 1.0 / (jnp.arange(N, dtype=dtype) + 1)
print(jnp.allclose(out.flatten(), expected_out))
```
which gives
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
## Projects that use JAXMg

- [JAXMg Benchmarks](https://github.com/therooler/jaxmg_benchmark): Benchmarks for various Multi-GPUs setups.
- [JAXMg + Netket](https://github.com/therooler/netket_jaxmg): Implementation of the MinSR Netket driver that uses JAXMg for inverting the S-matrix. Tested on Multi-node settings.
- [JAXMg for blurred sampling](https://github.com/therooler/nqs_blurred_sampling): Implementation of t-VMC that makes use JAXMg for inverting the QGT.

## cuSolverMp
The cuSolverMp backend is the migration path for multi-node execution. It uses
JAX's XLA-owned NCCL communicator through FFI, redistributes ordinary 2D
JAX-sharded matrices into cuSolverMp's 2D block-cyclic layout, and calls the
cuSolverMp routines directly. Explicit inverse support is intentionally absent:
current cuSolverMp releases do not expose a direct explicit-inverse routine.

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
