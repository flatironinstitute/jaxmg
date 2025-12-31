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
pip install jaxmg
```

This will install a GPU compatible version of JAX. By default we use `jax[cuda12]`, but `jaxmg` is compatible with the following alternative JAX configurations:

1. `pip install "jaxmg[cuda12-local]"`: Use locally available CUDA 12 installation.

2. `pip install "jaxmg[cuda13]"`: Use CUDA 13 (only works for `jax>=0.7.2`)

3. `pip install "jaxmg[cuda13-local]"`: Use locally available CUDA 13 installation.


### cuSolverMp
As of CUDA 13, there is a new distributed linear algebra library called [cuSolverMp](https://docs.nvidia.com/cuda/cusolvermp/) with similar capabilities as cuSolverMg, that does support multi-node computations as well as >16 devices. Given the similarities in syntax, it should be straightforward to eventually switch to this API. This will require sharding data into a cyclic 2D form and handling the solver orchestration with MPI.

## Citations
(Citation details will be available soon.)

## Acknowledgements
I acknowledge support from the Flatiron Institute. The Flatiron Institute is a
division of the Simons Foundation.