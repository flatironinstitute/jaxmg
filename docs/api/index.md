# API reference

This page highlights the three primary public functions from the jaxmg package.
Supported datatypes are `jax.numpy.float32`, `jax.numpy.float64`,
`jax.numpy.complex64`, and `jax.numpy.complex128`.

The routines accept ordinary JAX arrays sharded over a two-dimensional mesh.
The fused C++/CUDA backend converts those arrays into cuSOLVERMp's column-major,
2D block-cyclic layout and restores the JAX-facing layout before returning.

!!! Warning
    Each solver requires a tile width `T_A`. Very small tiles can substantially
    reduce solver performance. If a local shard dimension is not divisible by
    `T_A`, JAXMg adds tile-aligned capacity before entering native code. Prefer
    `T_A >= 128` and choose a tile size that divides the local shard dimensions
    when possible. See [Choosing a Tile Size](../examples/choose_tile_size.md)
    for further guidance.

## `potrs`

Multi-GPU Cholesky linear solver for symmetric (Hermitian) positive-definite matrices.

$$
A x = B, \quad A = L L^{\top} \;\text{(real)} \quad \text{or} \quad A = L L^{\dagger}\;\text{(complex)}
$$

Solve for $x$ using the Cholesky factors.

[POTRS API and usage](potrs.md)

---

## `lu_solve`

Multi-GPU pivoted LU solver for general nonsingular matrices.

$$
P A = L U, \qquad A x = B.
$$

[`lu_solve` API and usage](lu_solve.md)

---

## `syevd`

Multi-GPU eigensolver for symmetric (Hermitian) matrices.

$$
A v = \lambda v \quad\Rightarrow\quad A = V \Lambda V^{\top} \;\text{(real)}\quad\text{or}\quad A = V \Lambda V^{\dagger} \;\text{(complex)}
$$

Compute eigenvalues $\Lambda$ and eigenvectors $V$ of a symmetric or Hermitian
matrix.

[`syevd` API and usage](syevd.md)
