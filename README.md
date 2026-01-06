<div align="center">
    <img src="https://raw.githubusercontent.com/therooler/jaxmg/main/docs/_static/logo.png" alt="Jaxmg" width="300">
</div>

# JAXMg: A distributed linear solver in JAX with cuSolverMg

[![Docs](https://img.shields.io/badge/docs-site-blue?style=flat-square)](https://flatironinstitute.github.io/jaxmg/)
[![Releases](https://img.shields.io/github/v/release/therooler/jaxmg?style=flat-square)](https://github.com/therooler/jaxmg/releases)
[![Build Status](https://jenkins.flatironinstitute.org/job/jaxmg/job/jenkins/badge/icon)](https://jenkins.flatironinstitute.org/job/jaxmg/job/jenkins/)

# JAXMg
JAXMg provides a C++ interface between [JAX](https://github.com/google/jax) and [cuSolverMg](https://docs.nvidia.com/cuda/cusolver/index.html#using-the-cuSolverMg-api), NVIDIA’s multi-GPU linear solver.  We provide a jittable API for the following routines.

- [cusolverMgPotrs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolvermgpotrs-deprecated): Solves the system of linear equations: $Ax=b$ where $A$ is an $N\times N$ symmetric (Hermitian) positive-definite matrix via a Cholesky decomposition 
- [cusolverMgPotrs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolvermgpotri-deprecated): Computes the inverse of an $N\times N$ symmetric (Hermitian) positive-definite matrix via a Cholesky decomposition.
- [cusolverMgPotrs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolvermgsyevd-deprecated): Computes eigenvalues and eigenvectors of an $N\times N$ symmetric (Hermitian) matrix.

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
out = potrs(A, b, T_A=T_A, mesh=mesh, in_specs=(P("x", None), P(None, None)))
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

## cuSolverMp
As of CUDA 13, there is a new distributed linear algebra library called [cuSolverMp](https://docs.nvidia.com/cuda/cusolvermp/) with similar capabilities as cuSolverMg, that does support multi-node computations as well as >16 devices. Given the similarities in syntax, it should be straightforward to eventually switch to this API. This will require sharding data into a cyclic 2D form and handling the solver orchestration with MPI.

## Citations
```
@software{Wiersema_JAXMg_distributed_linear_2025,
author = {Wiersema, Roeland},
month = dec,
title = {{JAXMg: distributed linear solvers in JAX}},
url = {https://github.com/flatironinstitute/jaxmg},
version = {0.0.3},
year = {2025}
}
```

## Acknowledgements
I acknowledge support from the Flatiron Institute. The Flatiron Institute is a
division of the Simons Foundation.