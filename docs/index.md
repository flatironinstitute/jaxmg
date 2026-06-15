# JAXMg

<figure markdown>
  ![Title](_static/jaxmg_gpu_light.png#only-light){ width="600" }
  ![Title](_static/jaxmg_gpu_dark.png#only-dark){ width="600" } 
</figure>

JAXMg provides a jittable C++/CUDA interface between [JAX](https://github.com/google/jax) and [cuSOLVERMp](https://docs.nvidia.com/cuda/cusolvermp/), NVIDIA's distributed linear algebra runtime. We provide a fused API for the following routines.

- [cusolverMpPotrf/Potrs](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html): solves symmetric (Hermitian) positive-definite systems on a 2D process grid via `jaxmg.potrs`.
- [cusolverMpSyevd](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html): computes eigenvalues and eigenvectors on a 2D process grid via `jaxmg.syevd`.

For more details, see the [API](api/potrs.md) and the accompanying [paper](https://arxiv.org/abs/2601.14466).
