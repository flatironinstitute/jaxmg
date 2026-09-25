# API reference

This page highlights the primary public functions from the jaxmg package.
Supported datatypes are `jax.numpy.float32`, `jax.numpy.float64`,
`jax.numpy.complex64`, and `jax.numpy.complex128`.

The routines accept ordinary JAX arrays sharded over a one- or two-axis device
mesh.
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

## `least_squares`

Multi-GPU least-squares solver for overdetermined systems.

$$
X = \underset{X}{\operatorname{argmin}}\;\lVert A X-B\rVert_2,
\qquad A\in\mathbb{F}^{M\times N},\quad M\geq N.
$$

[`least_squares` API and usage](least_squares.md)

---

## `qr`

Compute the reduced QR decomposition of a tall or square matrix:

$$
A = Q R, \qquad Q^{\dagger}Q=I,
$$

where $Q$ is $M\times N$ and $R$ is $N\times N$ and upper triangular.

[`qr` API and usage](qr.md)

---

## `syevd`

Multi-GPU eigensolver for symmetric (Hermitian) matrices.

$$
A v = \lambda v \quad\Rightarrow\quad A = V \Lambda V^{\top} \;\text{(real)}\quad\text{or}\quad A = V \Lambda V^{\dagger} \;\text{(complex)}
$$

Compute eigenvalues $\Lambda$ and optionally eigenvectors $V$ of a symmetric or
Hermitian matrix.

[`syevd` API and usage](syevd.md)

---

## `gesvd`

Compute the singular-value decomposition of an $M \times N$ real or complex
matrix $A$ ($A = U \Sigma V^{\dagger}$), returning the singular values and
independently selected left and right singular vectors in reduced or full form.

[`gesvd` API and usage](gesvd.md)

---

## `polar`

Compute the polar decomposition of a tall or square matrix $A$:

$$
A = U_p H,
$$

returning the polar factor $U_p$ and optionally the Hermitian
positive-semidefinite factor $H$.

[`polar` API and usage](polar.md)
