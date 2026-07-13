# JAXMg

<figure markdown>
  ![Title](_static/jaxmg_gpu_light.png#only-light){ width="600" }
  ![Title](_static/jaxmg_gpu_dark.png#only-dark){ width="600" }
</figure>

JAXMg provides a jittable C++/CUDA interface between
[JAX](https://github.com/jax-ml/jax) and
[cuSOLVERMp](https://docs.nvidia.com/cuda/cusolvermp/), NVIDIA's distributed
multi-GPU linear algebra library. It exposes three fused solver workflows:

- [`potrs`](api/potrs.md) factors a symmetric or Hermitian positive-definite
  matrix with Cholesky factorization and solves $Ax=B$.
- [`lu_solve`](api/lu_solve.md) factors a general nonsingular matrix with
  pivoted LU factorization and solves $Ax=B$.
- [`syevd`](api/syevd.md) computes the eigenvalues and eigenvectors of a
  symmetric or Hermitian matrix.

The public functions accept ordinary JAX arrays sharded over a two-dimensional
device mesh. Inside one fused FFI call, the native backend converts the local
memory layout, redistributes the matrix into cuSOLVERMp's 2D block-cyclic
layout, executes the solver, and restores the JAX-facing layout.

JAXMg uses one Python process per GPU. See [Distributed
execution](execution.md) before launching a multi-GPU or multi-node program.

For more details, see the [API reference](api/index.md), the [native workflow](technical_details/index.md), and the accompanying [paper](https://arxiv.org/abs/2601.14466).
