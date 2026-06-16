# JAXMg

<figure markdown>
  ![Title](_static/jaxmg_gpu_light.png#only-light){ width="600" }
  ![Title](_static/jaxmg_gpu_dark.png#only-dark){ width="600" } 
</figure>

JAXMg provides a C++ interface between [JAX](https://github.com/google/jax) and [cuSOLVERMp](https://docs.nvidia.com/cuda/cusolvermp/), NVIDIA's distributed linear algebra runtime. We provide a jittable API for the following routines.

- [cusolverMpPotrf/cusolverMpPotrs](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html): Solves the system of linear equations $Ax=b$, where $A$ is an $N\times N$ symmetric (Hermitian) positive-definite matrix.
- [cusolverMpSyevd](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html): Computes eigenvalues and eigenvectors of an $N\times N$ symmetric (Hermitian) matrix.

Both routines accept ordinary 2D JAX-sharded arrays. JAXMg pads local shards when needed, enters a fused native FFI call, redistributes to cuSOLVERMp's 2D block-cyclic layout, calls cuSOLVERMp, and redistributes results back to the JAX-facing layout.

For more details, see the [API](api/potrs.md) and the accompanying [paper](https://arxiv.org/abs/2601.14466).
