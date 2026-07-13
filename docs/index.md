# JAXMg

<figure markdown>
  ![Title](_static/jaxmg_gpu_light.png#only-light){ width="600" }
  ![Title](_static/jaxmg_gpu_dark.png#only-dark){ width="600" }
</figure>

JAXMg provides a C++/CUDA interface between
[JAX](https://github.com/jax-ml/jax) and
[cuSOLVERMp](https://docs.nvidia.com/cuda/cusolvermp/), NVIDIA's distributed
multi-GPU linear algebra library. We provide a jittable API for the following
routines:

- [`potrs`](api/potrs.md): Solves the system of linear equations $Ax=B$, where
  $A$ is an $N\times N$ symmetric (Hermitian) positive-definite matrix, via a
  Cholesky decomposition
  ([`cusolverMpPotrf`](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html#cusolvermppotrf)
  and [`cusolverMpPotrs`](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html#cusolvermppotrs)).
- [`lu_solve`](api/lu_solve.md): Solves the system of linear equations $Ax=B$,
  where $A$ is an $N\times N$ general nonsingular matrix, via a pivoted LU
  decomposition
  ([`cusolverMpGetrf`](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html#cusolvermpgetrf)
  and [`cusolverMpGetrs`](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html#cusolvermpgetrs)).
- [`syevd`](api/syevd.md): Computes the eigenvalues and eigenvectors of an
  $N\times N$ symmetric (Hermitian) matrix
  ([`cusolverMpSyevd`](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html#cusolvermpsyevd)).

**What this allows you to do**

Pass JAXMg an ordinary JAX matrix sharded over a two-dimensional device mesh.
JAXMg then handles the native local-memory conversion, global
redistribution into cuSOLVERMp's 2D block-cyclic layout, distributed solver
execution, and restoration of the result.

This allows your JAX linear algebra calculation to scale across multiple GPUs
and nodes, supporting matrices far beyond the memory and computational limits
of native JAX implementations.

For more details, see the [API reference](api/index.md) and the accompanying
[paper](https://arxiv.org/abs/2601.14466).
