# JAXMg

<figure markdown>
  ![Title](_static/jaxmg_gpu_light.png#only-light){ width="600" }
  ![Title](_static/jaxmg_gpu_dark.png#only-dark){ width="600" } 
</figure>

JAXMg provides a C++ interface between [JAX](https://github.com/google/jax), [cuSolverMg](https://docs.nvidia.com/cuda/cusolver/index.html#using-the-cuSolverMg-api), and [cuSolverMp](https://docs.nvidia.com/cuda/cusolvermp/), NVIDIA's distributed linear algebra runtimes. We provide a jittable API for the following routines.

- [cusolverMgPotrs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolvermgpotrs-deprecated): Solves the system of linear equations: $Ax=b$ where $A$ is an $N\times N$ symmetric (Hermitian) positive-definite matrix via a Cholesky decomposition 
- [cusolverMgSyevd](https://docs.nvidia.com/cuda/cusolver/index.html#cusolvermgsyevd-deprecated): Computes eigenvalues and eigenvectors of an $N\times N$ symmetric (Hermitian) matrix.
- [cusolverMpPotrf/Potrs](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html): Solves symmetric (Hermitian) positive-definite systems on a 2D process grid via `jaxmg.potrs_mp`.
- [cusolverMpSyevd](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html): Computes eigenvalues and eigenvectors on a 2D process grid via `jaxmg.syevd_mp`.

For more details, see the [API](api/potrs.md) and the accompanying [paper](https://arxiv.org/abs/2601.14466).
