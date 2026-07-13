# Examples

The examples in this section introduce the complete JAXMg workflow, from
launching a distributed JAX program to running a cuSOLVERMp routine. Work
through them in the following order.

## 1. Configure distributed execution

JAXMg uses one Python process per GPU. Start by initializing distributed JAX,
assigning one GPU to each process, and constructing the two-dimensional device
mesh used by the matrix.

[Configure distributed execution](../execution.md)

## 2. Choose a tile size $T_A$

Choose a cuSOLVERMp tile size that divides both dimensions of each local matrix
shard. This avoids coefficient-matrix padding and controls the native scratch
allocation.

[Choose a tile size](choose_tile_size.md)

## 3. Run a solver

The solver examples use the same distributed execution and tile-size setup:

- [Cholesky solve](potrs.md) solves a symmetric or Hermitian positive-definite
  linear system with `potrs`.
- [LU solve](lu_solve.md) solves a general nonsingular linear system with
  `lu_solve`.
- [Symmetric or Hermitian eigensolve](syevd.md) computes eigenvalues and
  eigenvectors with `syevd`.

Each example passes ordinary JAX-sharded arrays to the public API. JAXMg handles
the local memory conversion, 2D block-cyclic redistribution, cuSOLVERMp call,
and reverse redistribution inside the native backend.
