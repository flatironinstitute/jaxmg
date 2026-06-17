# Column major layout expectations

This note explains where column-major storage enters JAXMg and what it means for
memory use.

cuSOLVERMp follows the usual Fortran-style convention: each local matrix owned
by a process is stored in column-major order. JAX users, however, normally think
in terms of logical arrays and sharding, not in terms of the physical order of
each device-local buffer. JAXMg therefore keeps the public boundary as ordinary
row-major JAX storage and performs the local row-major/column-major conversion
inside native code before the solver-specific redistribution.

Row-major vs column-major
- Row-major (C order): rows are laid out contiguously. For an $m\times n$
  matrix $A$, the memory is the concatenation of rows $1,2,\dots,m$.
- Column-major (Fortran order): columns are laid out contiguously. The memory
  is the concatenation of columns $1,2,\dots,n$.

Concretely, for a small $4\times 4$ matrix

$$
A = \begin{bmatrix}
a_{11} & a_{12} & a_{13} & a_{14} \\
a_{21} & a_{22} & a_{23} & a_{24} \\
a_{31} & a_{32} & a_{33} & a_{34} \\
a_{41} & a_{42} & a_{43} & a_{44}
\end{bmatrix}
$$

- Row-major flattening produces

$$
\operatorname{row\_major}(A) = [a_{11}, a_{12}, a_{13}, a_{14},\; a_{21}, a_{22},\dots,a_{44}]
$$

- Column-major flattening produces

$$
\operatorname{col\_major}(A) = [a_{11}, a_{21}, a_{31}, a_{41},\; a_{12}, a_{22},\dots,a_{44}]
$$

Where this happens in JAXMg
- The public functions accept normal 2D JAX-sharded arrays.
- JAXMg pads local shards in JAX when the local shard dimensions are not
  multiples of the tile size `T_A`.
- The FFI wrappers pass ordinary row-major local buffers to native code.
- Native code converts each donated local shard to column-major physical
  storage using bounded scratch, redistributes those padded JAX-facing buffers
  into cuSOLVERMp's 2D block-cyclic layout, and calls cuSOLVERMp.
- After the solver finishes, native code redistributes results back to the
  JAX-facing layout, restores row-major local storage for user-visible outputs,
  and Python removes any JAX-visible padding.

This is different from allocating a second full distributed matrix in Python.
The expensive layout conversion to cuSOLVERMp's 2D block-cyclic form happens
inside one fused native FFI call. The native redistribution uses bounded scratch
buffers for rectangular moves rather than keeping a complete second copy of the
matrix in Python.

Memory implications
- Padding in JAX can create a larger JAX array. This cost is real and is why the
  tile size should be chosen so that local shard dimensions are already
  tile-aligned where possible.
- The local row-major/column-major conversion is performed in-place on donated
  native work buffers using `O(max(local_rows, local_cols))` scratch. This
  avoids asking XLA for a column-major FFI layout, which can otherwise
  materialize a second full local shard.
- The native cuSOLVERMp redistribution itself is designed not to allocate a
  second full distributed matrix. It uses per-rank scratch for the rectangle
  moves needed by edge padding and 2D block-cyclic redistribution.

So the column-major requirement is still relevant, but it is now a local FFI and
cuSOLVERMp-layout concern rather than something users should solve manually by
constructing a special column-sharded input array.
