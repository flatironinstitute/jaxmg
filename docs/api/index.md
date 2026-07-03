# API Reference

This page highlights the primary public functions from the `jaxmg` package. Supported datatypes
are `jax.numpy.float32`, `jax.numpy.float64`, `jax.numpy.complex64` and `jax.numpy.complex128`.

The multi-GPU solvers called by JAXMg expect cuSOLVERMp's 2D block-cyclic
layout with column-major local GPU buffers. The conversion between ordinary
JAX-sharded inputs and that cuSOLVERMp layout is performed internally in the
C++/CUDA layer. Users can pass normal 2D JAX-sharded matrices to the high-level
functions; the library handles the local layout conversion, remapping, and
padding required by the native kernels.

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

## lu_solve

Multi-GPU LU linear solver for general nonsingular matrices.

$$
A x = B, \quad P A = L U
$$

Solve for $x$ using an LU factorization with pivoting. Use this routine when
the matrix is not known to be symmetric (Hermitian) positive definite. If the
matrix is positive definite, `potrs` is usually the better choice.

[Full lu_solve module →](lu_solve.md)

---

## syevd

Multi-GPU eigensolver for symmetric (Hermitian) matrices.

$$
A v = \lambda v \quad\Rightarrow\quad A = V \Lambda V^{\top} \;\text{(real)}\quad\text{or}\quad A = V \Lambda V^{\dagger} \;\text{(complex)}
$$

Compute eigenvalues $\Lambda$ and eigenvectors $V$ of a symmetric (Hermitian) matrix.

[Full syevd module →](syevd.md)
