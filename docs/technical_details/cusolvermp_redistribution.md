# cuSOLVERMp 2D Redistribution Investigation

cuSOLVERMp expects a 2D block-cyclic matrix layout. For a process grid with
`P_r` process rows and `P_c` process columns, a matrix tile `(i, j)` is owned by

```text
process_row = i % P_r
process_col = j % P_c
```

The current cuSOLVERMg compatibility backend only performs a 1D redistribution.
It moves full local row/vector slots in the native buffer. Despite the legacy
"column" terminology in parts of the code, the current native offset model
treats a slot as a contiguous region of length equal to the matrix width.

The first cuSOLVERMp investigation therefore asks whether the 2D ownership map
can be reached by two 1D-style passes:

1. Column-owner phase: within each process row, move tile columns until every
   tile has the correct `process_col`.
2. Row-owner phase: within each process column, move tile rows until every tile
   has the correct `process_row`.

The CPU-only planner in `jaxmg._block_cyclic_2d_plan` confirms that this
factorization is correct at the ownership level. Applying the column-owner moves
and then the row-owner moves reaches the same owner as the direct 2D
block-cyclic rule above, including degenerate grids such as `1 x P` and
`P x 1`.

The planner also builds a concrete fragment transfer schedule. Each scheduled
transfer records:

1. the phase (`column_owner` or `row_owner`);
2. the source and target process-grid owner;
3. the source and target flattened rank;
4. the phase-parallel group, which is a process row for column-owner transfers
   and a process column for row-owner transfers;
5. the logical rectangle being moved; and
6. whether that rectangle is contiguous under the current local memory layout.

This schedule is the intended input shape for the future CUDA/NCCL movement
prototype.

The planner also groups transfers into conflict-free batches. Every
`column_owner` batch is ordered before every `row_owner` batch. Within one
batch, a rank appears at most once as a source and at most once as a target.
That is the scratch-memory invariant expected by the GPU implementation: one
packed send fragment and one receive destination per rank are enough for a
single communication round.

## Padding

The 2D planner applies the existing JAXMg padding rule independently in each
axis:

```text
row_padding_per_process = (-local_logical_rows) % MB_A
col_padding_per_process = (-local_logical_cols) % NB_A
```

This gives each process tile-aligned local capacity. It does not by itself mean
that every logical `MB_A x NB_A` tile is initially owned by a single process. If
the logical local shard size is not already a multiple of the tile size, then a
logical tile can cross an initial JAX block-sharding boundary. In that case the
planner reports a split tile.

Split tiles are important because they prevent a simple whole-tile permutation
from being correct. The implementation must either:

1. first compact/canonicalize the padded layout so real data is tile-aligned in
   the global physical address space; or
2. fall back to moving tile fragments until the layout is tile-aligned.

This is the 2D version of the edge case that the current 1D implementation avoids
by planning at slot granularity.

## Contiguity

The current native memory model is row-major for the movement engine: a full
row/vector slot is contiguous, while a sub-column region spanning multiple rows
is strided.

That has direct consequences for the proposed two-phase algorithm:

1. A column-owner move is naturally a column-tile region. Under the current
   row-major offset model this is strided and will need pack/unpack kernels, many
   smaller sends, or a different storage view.
2. A row-owner move can be grouped as a full-width row slab inside a process
   column. Under the current row-major offset model this can be contiguous.
3. Moving individual `MB_A x NB_A` tiles is generally strided unless the tile
   covers the complete local width or has only one row.

The fragment schedule is intentionally conservative: it marks each raw
tile-fragment move independently. For ordinary tile sizes this means both
column-owner and row-owner fragment moves require packing in the row-major model.
The row-owner phase may still be optimized later by grouping compatible
fragments into full-width row slabs before issuing communication.

The next implementation checkpoint should therefore not be a cuSOLVERMp solver
call. It should be a GPU movement prototype that proves the column-owner phase
can pack, send, receive, and unpack strided column-tile regions without requiring
a second full matrix allocation.

## First GPU Primitive

The first native checkpoint is `xla_rect_pack_unpack_probe`, registered as an
optional CUDA FFI diagnostic. It takes a rank-2 local matrix shard plus rank-1
scratch, then performs:

```text
source rectangle in matrix -> contiguous scratch -> target rectangle in matrix
```

using `cudaMemcpy2DAsync` on XLA's CUDA stream. This deliberately does not use
the XLA communicator yet. It isolates the local strided-addressing part of the
2D redistribution, which is needed before a packed fragment can be handed to
NCCL/XLA communication for the inter-rank movement.
