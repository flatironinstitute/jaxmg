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
  Cholesky decomposition.
- [`lu_solve`](api/lu_solve.md): Solves the system of linear equations $Ax=B$,
  where $A$ is an $N\times N$ general nonsingular matrix, via a pivoted LU
  decomposition.
- [`syevd`](api/syevd.md): Computes the eigenvalues and eigenvectors of an
  $N\times N$ symmetric (Hermitian) matrix.

The public functions accept ordinary JAX arrays sharded over a two-dimensional
device mesh. Inside one fused FFI call, the native backend converts the local
memory layout, redistributes the matrix into cuSOLVERMp's 2D block-cyclic
layout, executes the solver, and restores the JAX-facing layout.

JAXMg uses one Python process per GPU. See [Distributed
execution](execution.md) before launching a multi-GPU or multi-node program.

For more details, see the [API reference](api/index.md), the [native
workflow](technical_details/index.md), and the accompanying
[paper](https://arxiv.org/abs/2601.14466).

The native implementations use the following cuSOLVERMp routines:

- `potrs`: [`cusolverMpPotrf`](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html#cusolvermppotrf)
  and [`cusolverMpPotrs`](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html#cusolvermppotrs)
  ([implementation](https://github.com/flatironinstitute/jaxmg/blob/xla_docker_build/src/cuda/cusolvermp_routines/cusolvermp_potrs.cc)).
- `lu_solve`: [`cusolverMpGetrf`](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html#cusolvermpgetrf)
  and [`cusolverMpGetrs`](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html#cusolvermpgetrs)
  ([implementation](https://github.com/flatironinstitute/jaxmg/blob/xla_docker_build/src/cuda/cusolvermp_routines/cusolvermp_lu_solve.cc)).
- `syevd`: [`cusolverMpSyevd`](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html#cusolvermpsyevd)
  ([implementation](https://github.com/flatironinstitute/jaxmg/blob/xla_docker_build/src/cuda/cusolvermp_routines/cusolvermp_syevd.cc)).
