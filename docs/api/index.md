# API Reference

This page highlights the three primary public functions from the `jaxmg` package. Supported datatypes
are `jax.numpy.float32`, `jax.numpy.float64`, `jax.numpy.complex64` and `jax.numpy.complex128`.

All multi-GPU solvers in called by JAXMg expect a 1D block-cyclic column layout at the device level
— a tiled, round-robin distribution of columns across devices driven by the tile width `T_A` used by the native kernels. 
The conversion between the natural row-sharded JAX input and the 1D block-cyclic layout is performed
internally in the C++/CUDA layer. Users can pass normal row-sharded matrices to the
high-level functions; the library handles the remapping and padding required by the native kernels so you don't have to manage the cyclic layout yourself.

!!! Warning
    The user must supply a tile width `T_A` to the solvers. Choose `T_A` carefully: very small values (e.g. < 128) can make the native kernels much slower. Furthermore, if the shard size of the matrix is not a multiple of `T_A` we must add per-device padding to fit the last tile — that padding requires copying data and increases memory use and runtime. In short: prefer a reasonably large `T_A` (>=128) and, where possible, pick `T_A` so that your shard size is an exact multiple to avoid copying and unnecessary slowdown.

## potrs

Multi-GPU Cholesky linear solver for symmetric (Hermitian) positive-definite matrices.

$$
A x = B, \quad A = L L^{\top} \;\text{(real)} \quad \text{or} \quad A = L L^{\dagger}\;\text{(complex)}
$$

Solve for $x$ using the Cholesky factors.

[Full potrs module →](potrs.md)

---

## potri

Multi-GPU matrix inversion helper for symmetric (Hermitian) positive-definite matrices.

$$
A^{-1} = (L L^{\top})^{-1} = L^{-\top} L^{-1} \quad\text{(real)}\quad\text{or}\quad A^{-1} = L^{-\dagger} L^{-1} \;\text{(complex)}
$$

Compute the inverse (or the upper-triangular part) of using Cholesky the Cholesky factors.

[Full potri module →](potri.md)

---

## syevd

Multi-GPU eigensolver for symmetric (Hermitian) matrices.

$$
A v = \lambda v \quad\Rightarrow\quad A = V \Lambda V^{\top} \;\text{(real)}\quad\text{or}\quad A = V \Lambda V^{\dagger} \;\text{(complex)}
$$

Compute eigenvalues $\Lambda$ and (optionally) eigenvectors $V$ of a symmetric (Hermitian) matrix.

[Full syevd module →](syevd.md)