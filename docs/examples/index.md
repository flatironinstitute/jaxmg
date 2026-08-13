# Examples

We provide a complete walkthrough of the JAXMg workflow, from
launching a distributed JAX program to running a cuSOLVERMp routine.

Ready-to-run scripts are provided in the repository's
[examples directory](https://github.com/flatironinstitute/jaxmg/tree/main/examples).

## 1. Configure Distributed Execution

Start by initializing distributed JAX,
assigning one GPU to each process, and constructing the device mesh used by the
matrix.

[Configure distributed execution](execution.md)

## 2. Choose a tile size

Choose a cuSOLVERMp tile size $T_A$ that divides both dimensions of each local matrix
shard. Choosing an appropriate tile size can substantially improve runtime and
reduce memory use.

[Choose a tile size](choose_tile_size.md)

## 3. Run a solver

- [Cholesky solve](potrs.md) solves a symmetric or Hermitian positive-definite
  linear system with `potrs`.
- [LU solve](lu_solve.md) solves a general nonsingular linear system with
  `lu_solve`.
- [Symmetric or Hermitian eigensolve](syevd.md) computes eigenvalues and
  optional eigenvectors with `syevd`.
- [Singular-value decomposition](gesvd.md) computes the singular values and
  optional left and right singular vectors of a real or complex matrix with
  `gesvd`.
