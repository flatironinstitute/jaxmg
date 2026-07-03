# Memory distribution

This page describes how JAXMg moves data between the JAX-facing layout and the
layout required by cuSOLVERMp.  The key point is that the public JAX arrays keep
their normal row-major interface, while the native backend converts and
redistributes the donated GPU buffers inside one fused FFI call.

At a high level, the path is:

```text
JAX sharded arrays
row-major local GPU buffers
        |
        | Python: validate shapes, dtype, mesh, tile size
        | Python: add local tile-aligned capacity if required
        v
Fused native FFI call
        |
        | Stage 1: local row-major -> column-major conversion
        v
column-major local GPU buffers
        |
        | Stage 2: top-left / edge-padding alignment
        v
tile-aligned global matrix
padding consolidated on the right and bottom edges
        |
        | Stage 3: 2D block-cyclic redistribution
        v
cuSOLVERMp column-major block-cyclic local buffers
        |
        | cuSOLVERMp POTRS / LU / SYEVD
        v
reverse Stage 3 -> reverse Stage 2 -> reverse Stage 1
        |
        v
JAX-facing row-major result
```

The native path is designed around a single scratch allocation.  The scratch
size is set by Stage 3, the 2D block-cyclic redistribution:

```text
scratch_elements =
    3 * max(tile_cols * local_rows,
            tile_rows * local_cols)
```

The earlier stages reuse this same scratch allocation.  They do not allocate a
second full local matrix.  Instead, they process as much data as possible within
the Stage 3 memory budget.

## Terminology

There are three separate layout concepts that are easy to confuse.

### Local memory layout (Internal to GPU)

This is the physical order of elements inside each GPU buffer.

```text
row-major local memory:
  elements from each row are contiguous

column-major local memory:
  elements from each column are contiguous
```

JAX normally presents rank-2 arrays in row-major local memory.  cuSOLVERMp
expects local matrix buffers in column-major memory.

### Process-grid mapping (Layout of GPUs)

This is how communicator ranks are placed into the 2D cuSOLVERMp process grid.
For a 2 x 3 process grid, row-major mapping is:

$$
\begin{bmatrix}
r_0 & r_1 & r_2 \\
r_3 & r_4 & r_5
\end{bmatrix},
\qquad
\mathrm{rank} = \mathrm{process\_row} \cdot \mathrm{process\_cols}
              + \mathrm{process\_col}.
$$

Column-major mapping is:

$$
\begin{bmatrix}
r_0 & r_2 & r_4 \\
r_1 & r_3 & r_5
\end{bmatrix},
\qquad
\mathrm{rank} = \mathrm{process\_col} \cdot \mathrm{process\_rows}
              + \mathrm{process\_row}.
$$

JAXMg supports these regular row-major and column-major rank maps.  Exotic rank
maps are rejected because cuSOLVERMp exposes standard grid-mapping modes rather
than an arbitrary rank-to-coordinate table.


## Stage 1: local layout conversion

The first native stage changes each GPU's local physical memory order from
JAX's row-major convention to cuSOLVERMp's column-major convention.  The
logical matrix is not transposed.

For a local 2 x 3 shard,

$$
A =
\begin{bmatrix}
a_{00} & a_{01} & a_{02} \\
a_{10} & a_{11} & a_{12}
\end{bmatrix}.
$$

The logical matrix is unchanged, but the physical memory vector changes:

$$
\begin{aligned}
\operatorname{row\_major}(A)
  &= [a_{00}, a_{01}, a_{02}, a_{10}, a_{11}, a_{12}], \\
\operatorname{col\_major}(A)
  &= [a_{00}, a_{10}, a_{01}, a_{11}, a_{02}, a_{12}].
\end{aligned}
$$

The native conversion therefore performs:

```text
row-major physical memory
        |
        | Stage 1: in-place physical layout conversion
        v
column-major physical memory
```

The address maps are:

```text
row-major address:    row * cols + col
column-major address: col * rows + row
```

Asking XLA to provide the FFI argument with `input_layouts=(1, 0)` would also
produce column-major local memory, but it can require a full additional local
matrix allocation.  JAXMg instead performs the conversion inside the fused
native call using bounded scratch.

The implementation follows the decomposition described by Catanzaro, Keller,
and Garland, [*A Decomposition for In-place Matrix Transposition*](https://research.nvidia.com/sites/default/files/pubs/2014-02_A-Decomposition-for/ppopp2014.pdf),
PPoPP 2014.  The algorithm decomposes the rectangular in-place permutation into
up to three substeps:

```text
row-major physical order
        |
        | optional column pre-shuffle
        v
intermediate order
        |
        | row shuffle / modular permutation
        v
intermediate order
        |
        | optional column post-shuffle
        v
column-major physical order
```

The row shuffle is the core modular permutation.  The column pre-shuffle and
column post-shuffle are enabled or skipped depending on the local shape and the
greatest common divisor of the local row and column counts.

For a local 4 x 8 shard, row-major storage has offset
$\mathrm{row} \cdot 8 + \mathrm{col}$:

$$
\begin{array}{c}
\text{logical matrix with row-major physical offsets} \\
\begin{bmatrix}
0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 \\
8 & 9 & 10 & 11 & 12 & 13 & 14 & 15 \\
16 & 17 & 18 & 19 & 20 & 21 & 22 & 23 \\
24 & 25 & 26 & 27 & 28 & 29 & 30 & 31
\end{bmatrix}
\end{array}
$$

Column-major storage for the same logical matrix has offset
$\mathrm{col} \cdot 4 + \mathrm{row}$:

$$
\begin{array}{c}
\text{logical matrix with column-major physical offsets} \\
\begin{bmatrix}
0 & 4 & 8 & 12 & 16 & 20 & 24 & 28 \\
1 & 5 & 9 & 13 & 17 & 21 & 25 & 29 \\
2 & 6 & 10 & 14 & 18 & 22 & 26 & 30 \\
3 & 7 & 11 & 15 & 19 & 23 & 27 & 31
\end{bmatrix}
\end{array}
$$

So the first logical column is stored in the first four physical slots:
$0,1,2,3$, and the first logical row is spaced by four slots:
$0,4,8,12,\ldots$.  The decomposition reaches that target through the
following physical-offset states:

$$
\begin{array}{c}
\text{row-major offsets} \\
\begin{bmatrix}
0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 \\
8 & 9 & 10 & 11 & 12 & 13 & 14 & 15 \\
16 & 17 & 18 & 19 & 20 & 21 & 22 & 23 \\
24 & 25 & 26 & 27 & 28 & 29 & 30 & 31
\end{bmatrix}
\end{array}
\quad
\xrightarrow{\text{column pre-shuffle}}
\quad
\begin{array}{c}
\text{offsets after pre-shuffle} \\
\begin{bmatrix}
0 & 1 & 26 & 27 & 20 & 21 & 14 & 15 \\
8 & 9 & 2 & 3 & 28 & 29 & 22 & 23 \\
16 & 17 & 10 & 11 & 4 & 5 & 30 & 31 \\
24 & 25 & 18 & 19 & 12 & 13 & 6 & 7
\end{bmatrix}
\end{array}
$$

$$
\begin{array}{c}
\text{offsets after pre-shuffle} \\
\begin{bmatrix}
0 & 1 & 26 & 27 & 20 & 21 & 14 & 15 \\
8 & 9 & 2 & 3 & 28 & 29 & 22 & 23 \\
16 & 17 & 10 & 11 & 4 & 5 & 30 & 31 \\
24 & 25 & 18 & 19 & 12 & 13 & 6 & 7
\end{bmatrix}
\end{array}
\quad
\xrightarrow{\text{row shuffle}}
\quad
\begin{array}{c}
\text{offsets after row shuffle} \\
\begin{bmatrix}
0 & 4 & 24 & 28 & 16 & 20 & 8 & 12 \\
9 & 13 & 1 & 5 & 25 & 29 & 17 & 21 \\
18 & 22 & 10 & 14 & 2 & 6 & 26 & 30 \\
27 & 31 & 19 & 23 & 11 & 15 & 3 & 7
\end{bmatrix}
\end{array}
$$

$$
\begin{array}{c}
\text{offsets after row shuffle} \\
\begin{bmatrix}
0 & 4 & 24 & 28 & 16 & 20 & 8 & 12 \\
9 & 13 & 1 & 5 & 25 & 29 & 17 & 21 \\
18 & 22 & 10 & 14 & 2 & 6 & 26 & 30 \\
27 & 31 & 19 & 23 & 11 & 15 & 3 & 7
\end{bmatrix}
\end{array}
\quad
\xrightarrow{\text{column post-shuffle}}
\quad
\begin{array}{c}
\text{column-major offsets} \\
\begin{bmatrix}
0 & 4 & 8 & 12 & 16 & 20 & 24 & 28 \\
1 & 5 & 9 & 13 & 17 & 21 & 25 & 29 \\
2 & 6 & 10 & 14 & 18 & 22 & 26 & 30 \\
3 & 7 & 11 & 15 & 19 & 23 & 27 & 31
\end{bmatrix}
\end{array}
$$

### Parallelism

Stage 1 is local to each GPU:

```text
GPU 0: local layout conversion
GPU 1: local layout conversion
GPU 2: local layout conversion
...
```

There is no NCCL communication in this stage.  The CUDA kernels run on the
current XLA stream and batch multiple rows or columns according to the available
scratch.  If the scratch budget is large, many rows/columns are processed per
launch; if it is small, the implementation falls back to smaller batches.


## Stage 2: top-left / edge-padding alignment

Python adds enough local capacity for tile-aligned execution when padding is
requested.  The native backend then compacts the real matrix data so that real
entries occupy the global top-left and padding is consolidated on the right and
bottom edges.

Consider a 4 x 2 process grid:

$$
\begin{bmatrix}
P_{00} & P_{01} \\
P_{10} & P_{11} \\
P_{20} & P_{21} \\
P_{30} & P_{31}
\end{bmatrix}.
$$

Suppose each process starts with a 5 x 5 local physical block.  The top-left
4 x 4 region contains real data and the right/bottom edge is padding.  We
write real entries as $r$ and padding capacity as $\square$:

$$
B^{(0)} =
\begin{bmatrix}
r & r & r & r & \square \\
r & r & r & r & \square \\
r & r & r & r & \square \\
r & r & r & r & \square \\
\square & \square & \square & \square & \square
\end{bmatrix}.
$$

Before native alignment the process grid is:

$$
\begin{array}{cc}
P_{00}:B^{(0)} & P_{01}:B^{(0)} \\
P_{10}:B^{(0)} & P_{11}:B^{(0)} \\
P_{20}:B^{(0)} & P_{21}:B^{(0)} \\
P_{30}:B^{(0)} & P_{31}:B^{(0)}
\end{array}
$$

This is tile-aligned local capacity, but it is not yet global edge padding.
Every local block has its own right and bottom padding.  Stage 2 compacts this
layout in two passes:

```text
1. horizontal compaction:
   push column padding to the global right edge

2. vertical compaction:
   push row padding to the global bottom edge
```

This is not a closed cyclic permutation.  Padding creates holes, so the
operation is an open-chain compaction.  Later real slabs are pulled into earlier
padded space.

### Horizontal compaction

Horizontal compaction acts along process columns.  In each process row, the
code shifts real columns left into earlier padding holes.  For process row
$i$, the initial column layout is:

$$
\begin{aligned}
P_{i0} &: [c_0, c_1, c_2, c_3, \square], \\
P_{i1} &: [c_4, c_5, c_6, c_7, \square],
\end{aligned}
$$

where each column symbol represents a full-height local column, including the
padded bottom row.  The compacted result is:

$$
\begin{aligned}
P_{i0} &: [c_0, c_1, c_2, c_3, c_4], \\
P_{i1} &: [c_5, c_6, c_7, \square, \square].
\end{aligned}
$$

The code builds this as a sequence of axis moves and lifts each move across all
process rows.  For this example, the horizontal waves are:

$$
\begin{aligned}
\text{wave 0:}\quad
&P_{i1}[:,0:1] \longrightarrow P_{i0}[:,4:5]
&&\text{for } i=0,1,2,3, \\
\text{wave 1:}\quad
&P_{i1}[:,1:4] \longrightarrow P_{i1}[:,0:3]
&&\text{for } i=0,1,2,3.
\end{aligned}
$$

So wave 0 contains four independent rectangle transfers:

```text
P01[:,0:1] -> P00[:,4:5]      P11[:,0:1] -> P10[:,4:5]
P21[:,0:1] -> P20[:,4:5]      P31[:,0:1] -> P30[:,4:5]
```

Wave 1 then performs the same local left-shift inside every process row:

```text
P01[:,1:4] -> P01[:,0:3]      P11[:,1:4] -> P11[:,0:3]
P21[:,1:4] -> P21[:,0:3]      P31[:,1:4] -> P31[:,0:3]
```

After horizontal compaction, the two local block shapes are:

$$
B^{(H)}_{\mathrm{left}} =
\begin{bmatrix}
r & r & r & r & r \\
r & r & r & r & r \\
r & r & r & r & r \\
r & r & r & r & r \\
\square & \square & \square & \square & \square
\end{bmatrix},
\qquad
B^{(H)}_{\mathrm{right}} =
\begin{bmatrix}
r & r & r & \square & \square \\
r & r & r & \square & \square \\
r & r & r & \square & \square \\
r & r & r & \square & \square \\
\square & \square & \square & \square & \square
\end{bmatrix}.
$$

The process grid is now:

$$
\begin{array}{cc}
P_{00}:B^{(H)}_{\mathrm{left}} & P_{01}:B^{(H)}_{\mathrm{right}} \\
P_{10}:B^{(H)}_{\mathrm{left}} & P_{11}:B^{(H)}_{\mathrm{right}} \\
P_{20}:B^{(H)}_{\mathrm{left}} & P_{21}:B^{(H)}_{\mathrm{right}} \\
P_{30}:B^{(H)}_{\mathrm{left}} & P_{31}:B^{(H)}_{\mathrm{right}}
\end{array}
$$

Column padding is now on the global right edge, but row padding is still present
inside every process row.

### Vertical compaction

Vertical compaction then acts along process rows.  For each process column
$j$, the row layout after horizontal compaction is:

$$
\begin{aligned}
P_{0j} &: [r_0, r_1, r_2, r_3, \square], \\
P_{1j} &: [r_4, r_5, r_6, r_7, \square], \\
P_{2j} &: [r_8, r_9, r_{10}, r_{11}, \square], \\
P_{3j} &: [r_{12}, r_{13}, r_{14}, r_{15}, \square],
\end{aligned}
$$

where $j \in \{0,1\}$ is the process-column index.  Each row symbol represents
the full local row width after horizontal compaction; in process column 1 that
row already includes right-edge padding.  The compacted result is:

$$
\begin{aligned}
P_{0j} &: [r_0, r_1, r_2, r_3, r_4], \\
P_{1j} &: [r_5, r_6, r_7, r_8, r_9], \\
P_{2j} &: [r_{10}, r_{11}, r_{12}, r_{13}, r_{14}], \\
P_{3j} &: [r_{15}, \square, \square, \square, \square].
\end{aligned}
$$

The code emits one wave at a time and lifts each wave across both process
columns:

$$
\begin{aligned}
\text{wave 0:}\quad &P_{1j}[0:1] \longrightarrow P_{0j}[4:5], \\
\text{wave 1:}\quad &P_{1j}[1:4] \longrightarrow P_{1j}[0:3], \\
\text{wave 2:}\quad &P_{2j}[0:2] \longrightarrow P_{1j}[3:5], \\
\text{wave 3:}\quad &P_{2j}[2:4] \longrightarrow P_{2j}[0:2], \\
\text{wave 4:}\quad &P_{3j}[0:3] \longrightarrow P_{2j}[2:5], \\
\text{wave 5:}\quad &P_{3j}[3:4] \longrightarrow P_{3j}[0:1],
\end{aligned}
\qquad j=0,1.
$$

For example, wave 2 contains two independent transfers:

```text
P20[0:2] -> P10[3:5]      P21[0:2] -> P11[3:5]
```

After the vertical waves, row padding is on the global bottom edge.  Because
there are four process rows, each with four real rows stored in five physical
rows, the last process row still owns one real row and four empty rows.  The
real matrix is now globally top-left aligned.

The final local block shapes are:

$$
B^{(F)}_{\mathrm{left}} =
\begin{bmatrix}
r & r & r & r & r \\
r & r & r & r & r \\
r & r & r & r & r \\
r & r & r & r & r \\
r & r & r & r & r
\end{bmatrix},
\qquad
B^{(F)}_{\mathrm{right}} =
\begin{bmatrix}
r & r & r & \square & \square \\
r & r & r & \square & \square \\
r & r & r & \square & \square \\
r & r & r & \square & \square \\
r & r & r & \square & \square
\end{bmatrix},
\qquad
B^{(F)}_{\mathrm{bottom,left}} =
\begin{bmatrix}
r & r & r & r & r \\
\square & \square & \square & \square & \square \\
\square & \square & \square & \square & \square \\
\square & \square & \square & \square & \square \\
\square & \square & \square & \square & \square
\end{bmatrix},
\qquad
B^{(F)}_{\mathrm{bottom,right}} =
\begin{bmatrix}
r & r & r & \square & \square \\
\square & \square & \square & \square & \square \\
\square & \square & \square & \square & \square \\
\square & \square & \square & \square & \square \\
\square & \square & \square & \square & \square
\end{bmatrix}.
$$

Thus the globally aligned 4 x 2 process grid is:

$$
\begin{array}{cc}
P_{00}:B^{(F)}_{\mathrm{left}} & P_{01}:B^{(F)}_{\mathrm{right}} \\
P_{10}:B^{(F)}_{\mathrm{left}} & P_{11}:B^{(F)}_{\mathrm{right}} \\
P_{20}:B^{(F)}_{\mathrm{left}} & P_{21}:B^{(F)}_{\mathrm{right}} \\
P_{30}:B^{(F)}_{\mathrm{bottom,left}}  & P_{31}:B^{(F)}_{\mathrm{bottom,right}}
\end{array}
$$

### Parallelism

The important implementation detail is that each wave is a dependency step, but
all moves inside the same wave are independent across the orthogonal process
axis.  Horizontal waves are parallel across process rows; vertical waves are
parallel across process columns.  Inter-rank moves use rectangle transfers
through the communicator, and same-rank moves use local CUDA movement.  Large
shifts are split into multiple waves so each rectangle fits in the Stage 3
scratch allocation.


## Stage 3: 2D block-cyclic redistribution

cuSOLVERMp uses a 2D block-cyclic matrix distribution.  For square tiles of
size `T_A`, ownership is:

$$
\mathrm{owner}(i,j)
  =
  \left(
    i \bmod \mathrm{process\_rows},
    j \bmod \mathrm{process\_cols}
  \right).
$$

JAX sharding gives each process a large rectangular shard.  Stage 3
redistributes tile slabs from that layout into cuSOLVERMp's 2D block-cyclic
layout.

The important change from the old 1D JAXMg redistribution is the unit of
movement.  Stage 3 moves whole tile slabs, not individual rows or columns.  A
tile slab may be:

- one or more tile columns, moved along a process row;
- one or more tile rows, moved along a process column.

Each slab is packed into the shared scratch buffer, transferred if it crosses
ranks, and unpacked into its destination.  If the source and destination are the
same rank, the same rectangle machinery is used with local CUDA movement rather
than communicator traffic.

### Example: 4 x 4 tiles on a 2 x 2 process grid

Assume the global matrix has already been converted to column-major local
memory and top-left aligned.  Let $A_{ij}$ denote the tile in global tile row
$i$ and global tile column $j$.  The global tile matrix is:

$$
A =
\begin{bmatrix}
A_{00} & A_{01} & A_{02} & A_{03} \\
A_{10} & A_{11} & A_{12} & A_{13} \\
A_{20} & A_{21} & A_{22} & A_{23} \\
A_{30} & A_{31} & A_{32} & A_{33}
\end{bmatrix}.
$$

With a 2 x 2 JAX block-sharded mesh, each process starts with one large
rectangular shard:

$$
\begin{array}{cc}
P_{00}:
\begin{bmatrix}
A_{00} & A_{01} \\
A_{10} & A_{11}
\end{bmatrix}
&
P_{01}:
\begin{bmatrix}
A_{02} & A_{03} \\
A_{12} & A_{13}
\end{bmatrix}
\\[1.0em]
P_{10}:
\begin{bmatrix}
A_{20} & A_{21} \\
A_{30} & A_{31}
\end{bmatrix}
&
P_{11}:
\begin{bmatrix}
A_{22} & A_{23} \\
A_{32} & A_{33}
\end{bmatrix}
\end{array}
$$

The target cuSOLVERMp ownership rule is:

$$
\mathrm{owner}(A_{ij}) = (i \bmod 2,\; j \bmod 2).
$$

So the final cuSOLVERMp layout should be:

$$
\begin{array}{cc}
P_{00}:
\begin{bmatrix}
A_{00} & A_{02} \\
A_{20} & A_{22}
\end{bmatrix}
&
P_{01}:
\begin{bmatrix}
A_{01} & A_{03} \\
A_{21} & A_{23}
\end{bmatrix}
\\[1.0em]
P_{10}:
\begin{bmatrix}
A_{10} & A_{12} \\
A_{30} & A_{32}
\end{bmatrix}
&
P_{11}:
\begin{bmatrix}
A_{11} & A_{13} \\
A_{31} & A_{33}
\end{bmatrix}
\end{array}
$$

Stage 3 reaches this target in two separable phases.

### Stage 3a: column-owner phase

The first phase fixes the process-column owner of every tile.  It works within
each process row, so the top process row and bottom process row can perform the
same kind of exchange independently.

For process row 0, the initial state is:

$$
\begin{aligned}
P_{00} &:
\begin{bmatrix}
A_{00} & A_{01} \\
A_{10} & A_{11}
\end{bmatrix},
&
P_{01} &:
\begin{bmatrix}
A_{02} & A_{03} \\
A_{12} & A_{13}
\end{bmatrix}.
\end{aligned}
$$

Tiles in even tile columns must end up in process column 0, and tiles in odd
tile columns must end up in process column 1.  Therefore the column phase
exchanges one full-height tile column slab:

$$
\begin{aligned}
P_{00}\{A_{01}, A_{11}\}
    &\longrightarrow P_{01}, \\
P_{01}\{A_{02}, A_{12}\}
    &\longrightarrow P_{00}.
\end{aligned}
$$

The same exchange occurs at the same time in process row 1:

$$
\begin{aligned}
P_{10}\{A_{21}, A_{31}\}
    &\longrightarrow P_{11}, \\
P_{11}\{A_{22}, A_{32}\}
    &\longrightarrow P_{10}.
\end{aligned}
$$

After the column-owner phase, every tile is in the correct process column:

$$
\begin{array}{cc}
P_{00}:
\begin{bmatrix}
A_{00} & A_{02} \\
A_{10} & A_{12}
\end{bmatrix}
&
P_{01}:
\begin{bmatrix}
A_{01} & A_{03} \\
A_{11} & A_{13}
\end{bmatrix}
\\[1.0em]
P_{10}:
\begin{bmatrix}
A_{20} & A_{22} \\
A_{30} & A_{32}
\end{bmatrix}
&
P_{11}:
\begin{bmatrix}
A_{21} & A_{23} \\
A_{31} & A_{33}
\end{bmatrix}
\end{array}
$$

At this point the process rows are still wrong for half the tiles.  For
example, $A_{10}$ and $A_{12}$ are in $P_{00}$ but should be in process row 1.

### Stage 3b: row-owner phase

The second phase fixes the process-row owner of every tile.  It works within
each process column, so process column 0 and process column 1 can perform the
same kind of exchange independently.

For process column 0, the state after the column phase is:

$$
\begin{aligned}
P_{00} &:
\begin{bmatrix}
A_{00} & A_{02} \\
A_{10} & A_{12}
\end{bmatrix},
&
P_{10} &:
\begin{bmatrix}
A_{20} & A_{22} \\
A_{30} & A_{32}
\end{bmatrix}.
\end{aligned}
$$

Tiles in even tile rows must end up in process row 0, and tiles in odd tile
rows must end up in process row 1.  Therefore the row phase exchanges one
full-width tile row slab:

$$
\begin{aligned}
P_{00}\{A_{10}, A_{12}\}
    &\longrightarrow P_{10}, \\
P_{10}\{A_{20}, A_{22}\}
    &\longrightarrow P_{00}.
\end{aligned}
$$

The same exchange occurs at the same time in process column 1:

$$
\begin{aligned}
P_{01}\{A_{11}, A_{13}\}
    &\longrightarrow P_{11}, \\
P_{11}\{A_{21}, A_{23}\}
    &\longrightarrow P_{01}.
\end{aligned}
$$

After the row-owner phase, every tile satisfies the cuSOLVERMp ownership rule:

$$
\begin{array}{cc}
P_{00}:
\begin{bmatrix}
A_{00} & A_{02} \\
A_{20} & A_{22}
\end{bmatrix}
&
P_{01}:
\begin{bmatrix}
A_{01} & A_{03} \\
A_{21} & A_{23}
\end{bmatrix}
\\[1.0em]
P_{10}:
\begin{bmatrix}
A_{10} & A_{12} \\
A_{30} & A_{32}
\end{bmatrix}
&
P_{11}:
\begin{bmatrix}
A_{11} & A_{13} \\
A_{31} & A_{33}
\end{bmatrix}
\end{array}
$$

This is the column-major local, 2D block-cyclic layout expected by
cuSOLVERMp.

### How the native code executes a slab move

Each slab move is executed as a rectangle transfer:

```text
source local column-major buffer
        |
        | pack logical rectangle into contiguous scratch
        v
packed scratch
        |
        | same-rank move: local CUDA copy/kernel
        | cross-rank move: NCCL-backed communicator transfer
        v
destination scratch
        |
        | unpack into destination logical rectangle
        v
destination local column-major buffer
```

The plan is fixed by the process grid, tile size, padded matrix shape, and
rank mapping.  For a given JIT trace, the native call receives static metadata
for these values and runs the same movement schedule each time that compiled
call is executed.

### Parallelism

The column-owner phase is independent across process rows:

$$
\begin{array}{l}
\text{process row 0: } P_{00} \leftrightarrow P_{01} \leftrightarrow \cdots \\
\text{process row 1: } P_{10} \leftrightarrow P_{11} \leftrightarrow \cdots \\
\vdots
\end{array}
$$

The row-owner phase is independent across process columns:

$$
\begin{array}{l}
\text{process column 0: } P_{00} \leftrightarrow P_{10} \leftrightarrow \cdots \\
\text{process column 1: } P_{01} \leftrightarrow P_{11} \leftrightarrow \cdots \\
\vdots
\end{array}
$$

The implementation is intentionally conservative.  It moves one tile slab or
cycle per independent process-row or process-column group.  That keeps scratch
bounded and avoids overlapping writes while still moving much larger units than
the old single-row or single-column redistribution strategy.

For the 2 x 2 example above, the column-owner phase has one dependency wave:

$$
\begin{array}{ll}
P_{00}\{A_{01}, A_{11}\} \rightarrow P_{01},
&
P_{01}\{A_{02}, A_{12}\} \rightarrow P_{00},
\\
P_{10}\{A_{21}, A_{31}\} \rightarrow P_{11},
&
P_{11}\{A_{22}, A_{32}\} \rightarrow P_{10}.
\end{array}
$$

Those four transfers are the same logical step lifted across both process
rows.  The row-owner phase then has one dependency wave:

$$
\begin{array}{ll}
P_{00}\{A_{10}, A_{12}\} \rightarrow P_{10},
&
P_{10}\{A_{20}, A_{22}\} \rightarrow P_{00},
\\
P_{01}\{A_{11}, A_{13}\} \rightarrow P_{11},
&
P_{11}\{A_{21}, A_{23}\} \rightarrow P_{01}.
\end{array}
$$

Those four transfers are the same logical step lifted across both process
columns.  Larger grids use more waves, but the principle is the same: a wave is
sequential with respect to the next wave, while the independent row or column
groups inside that wave can run together.

## Reverse path

The native solver outputs data in cuSOLVERMp's local column-major block-cyclic
layout.  The fused call reverses the memory movement stages before returning to
JAX:

```text
cuSOLVERMp block-cyclic output
        |
        | reverse Stage 3
        v
top-left aligned column-major JAX shard layout
        |
        | reverse Stage 2, if padding alignment was needed
        v
padded column-major local shards
        |
        | reverse Stage 1
        v
row-major local shards
        |
        | Python: remove visible padding
        v
JAX-facing result
```

For `potrs` and `lu_solve`, a vector right-hand side is represented internally
as an `N x 1` matrix.  The public wrappers return a vector again, so the output
rank matches the input rank.

## Solver-specific native work

The redistribution stages are shared by the fused solver entry points.  The
main differences are inside the cuSOLVERMp call sequence and the workspace
requested by that sequence:

- `potrs` uses `cusolverMpPotrf` followed by `cusolverMpPotrs`.  It is intended
  for symmetric (Hermitian) positive-definite matrices and does not allocate a
  pivot vector.
- `lu_solve` uses `cusolverMpGetrf` followed by `cusolverMpGetrs`.  It is
  intended for general nonsingular matrices and allocates a pivot vector sized
  by the local cuSOLVERMp column ownership, `LOCc(N_A)`.
- `syevd` uses `cusolverMpSyevd` and has different output and workspace
  pressure because eigenvectors are materialized as a full distributed matrix.

## Python and native responsibilities

Python is responsible for:

- validating dtype, shape, mesh, and rank map;
- requiring one Python process per GPU for cuSOLVERMp execution;
- calculating and applying local padding capacity;
- rejecting rank maps and tile configurations that cuSOLVERMp cannot represent;
- constructing and caching the FFI wrapper.

Native CUDA/C++ is responsible for:

- local row-major/column-major conversion;
- top-left / edge-padding alignment;
- 2D block-cyclic redistribution;
- cuSOLVERMp handle, descriptor, workspace, and solver calls;
- reverse redistribution and layout conversion.

## Code map

The memory distribution implementation is split across:

```text
Stage 1:
  src/cuda/memory_redist/layout_convert.cu.cc

Stage 2:
  src/cuda/memory_redist/edge_padding_2d.cc

Stage 3:
  src/cuda/memory_redist/block_cyclic_2d.cc
  src/cuda/memory_redist/rectangle_pack.cc
  src/cuda/memory_redist/scratch.cc

Solver orchestration:
  src/cuda/cusolvermp_routines/cusolvermp_potrs.cc
  src/cuda/cusolvermp_routines/cusolvermp_lu_solve.cc
  src/cuda/cusolvermp_routines/cusolvermp_syevd.cc
```
