# API Reference

This page highlights the two primary public functions from the `jaxmg` package. Supported datatypes
are `jax.numpy.float32`, `jax.numpy.float64`, `jax.numpy.complex64` and `jax.numpy.complex128`.

JAXMg accepts ordinary 2D JAX-sharded arrays and calls cuSOLVERMp through a fused
native FFI backend. The native backend redistributes each padded input from the
JAX-facing layout to cuSOLVERMp's 2D block-cyclic layout, calls the solver, and
redistributes results back to the original JAX-facing layout.

!!! Warning
    The user must supply a tile size `T_A` to the solvers. Choose `T_A` carefully: very small values (e.g. < 128) can make the native kernels much slower. Furthermore, if a local shard dimension is not a multiple of `T_A`, JAXMg must add per-device padding to fit the last tile. That padding requires copying data and increases memory use and runtime. In short: prefer a reasonably large `T_A` (>=128) and, where possible, pick `T_A` so that the local shard dimensions are exact multiples.

## potrs

Multi-GPU Cholesky linear solver for symmetric (Hermitian) positive-definite matrices.

$$
A x = B, \quad A = L L^{\top} \;\text{(real)} \quad \text{or} \quad A = L L^{\dagger}\;\text{(complex)}
$$

Solve for $x$ using the Cholesky factors.

[Full potrs module →](potrs.md)

---

## syevd

Multi-GPU eigensolver for symmetric (Hermitian) matrices.

$$
A v = \lambda v \quad\Rightarrow\quad A = V \Lambda V^{\top} \;\text{(real)}\quad\text{or}\quad A = V \Lambda V^{\dagger} \;\text{(complex)}
$$

Compute eigenvalues $\Lambda$ and eigenvectors $V$ of a symmetric (Hermitian) matrix.

[Full syevd module →](syevd.md)
