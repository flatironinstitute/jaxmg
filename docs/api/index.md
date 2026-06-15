# API Reference

This page highlights the public solver functions from the `jaxmg` package.
Supported datatypes are `jax.numpy.float32`, `jax.numpy.float64`,
`jax.numpy.complex64`, and `jax.numpy.complex128`.

The cuSOLVERMp backend uses a 2D JAX process grid and redistributes locally
padded JAX shards into the 2D block-cyclic, column-major local layout expected
by cuSOLVERMp. The redistribution and solver call are fused into one native
FFI call per routine.

!!! Warning
    The user must supply a tile width `T_A` to the solvers. Choose `T_A` carefully: very small values (e.g. < 128) can make the native kernels much slower. Furthermore, if the shard size of the matrix is not a multiple of `T_A` we must add per-device padding to fit the last tile — that padding requires copying data and increases memory use and runtime. In short: prefer a reasonably large `T_A` (>=128) and, where possible, pick `T_A` so that your shard size is an exact multiple to avoid copying and unnecessary slowdown.

## potrs

cuSOLVERMp Cholesky linear solver for symmetric (Hermitian)
positive-definite matrices on a 2D process grid.

$$
A x = B, \quad A = L L^{\top} \;\text{(real)} \quad \text{or} \quad A = L L^{\dagger}\;\text{(complex)}
$$

`potrs` accepts a 2D block-sharded JAX matrix, performs native GPU-to-GPU 2D block-cyclic redistribution,
runs cuSOLVERMp `potrf`/`potrs`, and reverse-redistributes the solved right-hand side to the original
JAX-facing layout.

[Full potrs module →](potrs.md)

---

## Distributed Setup

cuSOLVERMp runs should use ordinary `jax.distributed.initialize()` before any
device discovery, array creation, or JIT compilation. The `syevd` path follows
cuSOLVERMp's rank-per-GPU model: one Python process per participating GPU. Build
the 2D process mesh with normal JAX APIs such as `jax.make_mesh`.

## syevd

cuSOLVERMp eigensolver for symmetric (Hermitian) matrices on a 2D process grid.

$$
A v = \lambda v \quad\Rightarrow\quad A = V \Lambda V^{\top} \;\text{(real)}\quad\text{or}\quad A = V \Lambda V^{\dagger} \;\text{(complex)}
$$

`syevd` accepts a 2D block-sharded JAX matrix, performs native GPU-to-GPU 2D
block-cyclic redistribution, runs cuSOLVERMp `syevd`, and returns eigenvalues
and eigenvectors.

[Full syevd module ->](syevd.md)
