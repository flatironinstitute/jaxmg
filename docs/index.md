# JAXMg

<figure markdown>
  ![Title](_static/jaxmg_gpu_light.png#only-light){ width="600" }
  ![Title](_static/jaxmg_gpu_dark.png#only-dark){ width="600" }
</figure>

JAXMg brings distributed matrix solvers to JAX, allowing calculations to
scale across multiple GPUs and compute nodes. This enables matrix operations to
reach the memory limits imposed by the available GPU resources, far beyond
native JAX routines, while retaining a familiar JAX interface.

JAXMg currently provides a jittable API for the following routines:

- [`potrs`](api/potrs.md): Solves the system of linear equations $Ax=B$, where
  $A$ is an $N \times N$ symmetric or Hermitian positive-definite matrix, using
  a Cholesky decomposition. It can also return the log determinant of $A$.
- [`lu_solve`](api/lu_solve.md): Solves the system of linear equations $Ax=B$,
  where $A$ is an $N \times N$ general nonsingular matrix, using a pivoted LU
  decomposition.
- [`syevd`](api/syevd.md): Computes the eigenvalues and eigenvectors of an
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

For more details, see the [API reference](api/index.md) and the accompanying
[paper](https://arxiv.org/abs/2601.14466).
