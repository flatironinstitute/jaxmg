# JAXMg

<figure markdown>
  ![Title](_static/jaxmg_gpu_light.png#only-light){ width="600" }
  ![Title](_static/jaxmg_gpu_dark.png#only-dark){ width="600" }
</figure>

JAXMg brings distributed matrix solvers to JAX, allowing calculations to run
across multiple GPUs and nodes. This enables solvers to scale to matrices near
the combined memory capacity of the available GPUs, far beyond native JAX
routines, while retaining a familiar JAX interface.

JAXMg currently provides a jittable API for the following routines:

- [`potrs`](api/potrs.md): Solves the system of linear equations $Ax=B$, where
  $A$ is an $N \times N$ symmetric or Hermitian positive-definite matrix, using
  a Cholesky decomposition. It can also return the log determinant of $A$.
- [`lu_solve`](api/lu_solve.md): Solves the system of linear equations $Ax=B$,
  where $A$ is an $N \times N$ general nonsingular matrix, using a pivoted LU
  decomposition.
- [`syevd`](api/syevd.md): Computes the eigenvalues $\lambda_i$ and optional
  eigenvectors $v_i$ of an $N \times N$ symmetric or Hermitian matrix $A$,
  satisfying $Av_i=\lambda_i v_i$.
- [`gesvd`](api/gesvd.md): Computes the singular-value decomposition of an
  $M \times N$ real or complex matrix $A$ ($A = U \Sigma V^{\dagger}$),
  returning the singular values and optional left and right singular vectors.

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

For more details, see the [API reference](api/index.md) and the accompanying
[paper](https://arxiv.org/abs/2601.14466).

> **Deprecation notice:** JAXMg 0.x, based on NVIDIA's deprecated cuSOLVERMg
> backend, has been superseded by JAXMg 1.0. Version 1.0 requires one Python
> process per GPU and removes `potri`; the previous API remains available with
> `pip install "jaxmg<1"`. See the
> [0.0.9 release](https://github.com/flatironinstitute/jaxmg/releases/tag/v0.0.9)
> and [0.0.9 documentation](https://flatironinstitute.github.io/jaxmg/0.0.9/).
