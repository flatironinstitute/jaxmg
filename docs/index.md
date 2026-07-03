# JAXMg

<figure markdown>
  ![Title](_static/jaxmg_gpu_light.png#only-light){ width="600" }
  ![Title](_static/jaxmg_gpu_dark.png#only-dark){ width="600" } 
</figure>

JAXMg provides a C++/CUDA interface between [JAX](https://github.com/google/jax) and [cuSOLVERMp](https://docs.nvidia.com/cuda/cusolvermp/), NVIDIA's distributed multi-GPU linear algebra runtime.  We provide a jittable API for the following fused routines.

- [cusolverMpPotrf/Potrs](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html): solves $Ax=B$ where $A$ is an $N\times N$ symmetric (Hermitian) positive-definite matrix via a Cholesky decomposition.
- [cusolverMpGetrf/Getrs](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html): solves $Ax=B$ where $A$ is an $N\times N$ general nonsingular matrix via an LU factorization with pivoting.
- [cusolverMpSyevd](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html): computes eigenvalues and eigenvectors of an $N\times N$ symmetric (Hermitian) matrix.

For more details, see the [API](api/index.md) and the accompanying [paper](https://arxiv.org/abs/2601.14466).
