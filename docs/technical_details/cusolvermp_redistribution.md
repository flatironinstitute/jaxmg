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

This schedule was the input shape for the first CUDA/NCCL movement prototype.
The current production path rebuilds the equivalent schedule natively from
scalar shape metadata, but the Python plan remains the readable reference model.

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

### Edge-Padding Compaction

The preferred implementation for split-tile padding is a separate edge-padding
normalization pass before the true 2D block-cyclic redistribution. The goal is to
convert local per-shard padding into global edge padding:

```text
initial padded shards:
  real pad | real pad | real pad

edge-padded storage:
  real real real | pad pad pad
```

and similarly in the row direction. After this pass, real data is packed into
the global top-left of the padded domain, and padding exists only on the global
right and bottom edges. The later redistribution can then move full tile-aligned
slabs instead of arbitrary split-tile fragments.

This pass is still a shuffle. It is an open-chain compaction shuffle rather than
a closed permutation cycle. The free space is exactly the existing padding. A
move always writes into a padding hole, and the source region becomes the next
hole. In one process row, a conceptual slot sequence might look like:

```text
GPU0: A A A _
GPU1: B B B _
GPU2: C C C _
```

The first hole on GPU0 can receive the first real slot from GPU1:

```text
GPU0: A A A B
GPU1: _ B B _
GPU2: C C C _
```

The hole has moved to GPU1. Subsequent moves continue propagating holes to the
right until all horizontal padding sits at the right edge. No live data is
overwritten because every destination is known to be a hole before the move is
issued.

The production planner does not emit one move per tile slot unless it has to.
It operates on maximal aligned intervals:

```text
hole interval = first contiguous padding interval
real interval = next contiguous real interval to the right
move width    = min(width(hole interval), width(real interval))
```

That emits one large rectangle move such as:

```text
local_rows_cap x move_width
```

instead of many `local_rows_cap x NB_A` moves. After the move, the source
interval becomes the new hole interval. This gives a small number of dependency
waves in the common uniform-padding case, while preserving the low-memory
property of the hole-propagation algorithm.

The horizontal pass is independent for every process row. The native planner
therefore builds horizontal waves and batches all process rows that are at the
same dependency depth:

```text
for wave in horizontal_waves:
    execute independent row-compaction moves for all process rows
```

This means batching the same hole-propagation step across rows, not merging
different dependent steps from one row's chain.

Within a wave, a rank may participate in multiple logical process rows only if
the process grid mapping allows it without reusing the same scratch or
destination interval. The conservative invariant is the same as the current
rectangle executor: one send slot and one receive slot per rank per communication
round unless the native executor proves disjoint in-rank regions and has enough
scratch to support more.

After horizontal compaction, the vertical pass applies the same open-chain idea
inside each process column:

```text
hole interval = first contiguous bottom padding interval
real interval = next contiguous real interval below it
move height   = min(height(hole interval), height(real interval))
```

and emits large row-rectangle moves such as:

```text
move_height x local_cols_cap
```

The vertical pass can batch independent process columns by dependency wave. Its
local memory movement is more layout-sensitive: in cuSOLVERMp's column-major
local storage, column slabs are naturally contiguous, while row slabs are
strided. Vertical compaction therefore may need pack/unpack kernels or a custom
CUDA kernel even though the dependency logic is identical to horizontal
compaction.

The required ordering is:

```text
1. horizontal edge-padding compaction
2. vertical edge-padding compaction
3. column-owner block-cyclic redistribution
4. row-owner block-cyclic redistribution
```

The first two steps create a tile-addressable source domain. The last two steps
move that source domain into cuSOLVERMp ownership.

The current Python/JAX diagnostic path now exposes this ordering directly.
`execute_edge_padding_compaction_batches_shardmap` can still execute
CPU-planned compaction waves through the rectangle-transfer FFI for focused
debugging, while `execute_padded_block_cyclic_2d_shardmap` now calls one native
FFI target that plans and executes both edge compaction and tile-aligned 2D
redistribution internally. `jaxmg.potrs` now uses this movement path before
creating cuSOLVERMp descriptors and calling cuSOLVERMp `potrf`/`potrs`.

### Parallelism and Scratch Invariants

The native padded path does not ask Python to submit every rectangle move. Python
validates the physical padded shape and scratch size, then calls one FFI target:

```text
xla_rect_padded_2d_native_plan
```

The C++ handler then builds and executes the full schedule:

```text
1. horizontal edge-padding compaction
2. vertical edge-padding compaction
3. column-owner slab permutation
4. row-owner slab permutation
```

Parallelism means "the same dependency step across independent process
rows/columns", not "all unrelated cycle steps at once". This keeps the scratch
requirement fixed. Each rank has one scratch allocation split into three
equal-size slots:

```text
saved slab
send slab
receive slab
```

For a `2 x 2` process grid, with a `10 x 10` matrix and `4 x 4` cuSOLVERMp
tiles, each initial JAX block shard is `5 x 5`. The tile-aligned local capacity
is therefore `8 x 8`.

Horizontal compaction sees the column axis of each process row as:

```text
rank col 0: 5 real columns + 3 padding columns
rank col 1: 5 real columns + 3 padding columns
```

The generated waves are:

```text
wave 0:
  process row 0: rank 1 -> rank 0, move 8 x 3
  process row 1: rank 3 -> rank 2, move 8 x 3

wave 1:
  process row 0: rank 1 -> rank 1, move 8 x 2
  process row 1: rank 3 -> rank 3, move 8 x 2
```

The two moves in each wave are independent: no rank appears twice as a source or
target, so one send and one receive slot per rank are enough. Wave 1 cannot be
merged into wave 0 because it consumes the holes created by wave 0.

Vertical compaction then applies the same open-chain logic along rows:

```text
wave 0:
  process col 0: rank 2 -> rank 0, move 3 x 8
  process col 1: rank 3 -> rank 1, move 3 x 8

wave 1:
  process col 0: rank 2 -> rank 2, move 2 x 8
  process col 1: rank 3 -> rank 3, move 2 x 8
```

After these two compaction phases, the original `10 x 10` logical matrix is in
the top-left of the `16 x 16` physical padded domain. Padding exists only on the
global right and bottom edges.

The block-cyclic redistribution then uses slab cycles instead of individual
tiles. The column-owner phase fixes:

```text
target_process_col = tile_col % P_c
```

by moving slabs of shape:

```text
local_rows x NB_A
```

For the same `2 x 2`, `4 x 4` example this is `8 x 4`. A cycle step such as
`rank 1 -> rank 0` in process row 0 can be batched with the matching
`rank 3 -> rank 2` step in process row 1. These are the same logical transition
applied across independent process rows.

The row-owner phase then fixes:

```text
target_process_row = tile_row % P_r
```

by moving slabs of shape:

```text
MB_A x local_cols
```

For the example this is `4 x 8`. A step such as `rank 2 -> rank 0` in process
column 0 can be batched with the matching `rank 3 -> rank 1` step in process
column 1. These are the same logical transition applied across independent
process columns.

Closed permutation cycles are still dependency-ordered. The native code saves
one live slab in the saved slot, rotates the remaining slabs through the
send/receive slots, then restores the saved slab. Only conflict-free steps at
the same dependency depth are batched together.

The current layout check used a standalone cuSOLVERMp scatter verifier with the
same padded example (`10 x 10`, `2 x 2`, `4 x 4`). It compared NVIDIA
`cusolverMpMatrixScatterH2D` against the independent row-major-grid,
column-major-local-buffer reference with `LLD = 8` and local column capacity
`8`. That verifier passed, so the padded final layout matches cuSOLVERMp's
scatter layout for this case.

## Contiguity

The implemented cuSOLVERMp movement path uses column-major local addressing:

```text
offset = local_col * local_rows + local_row
```

This matches the local matrix convention required by cuSOLVERMp.  A column slab
is therefore naturally contiguous, while a row slab is strided across local
columns.  The native executor handles both cases with the same pack/unpack
primitive:

```text
source rectangle in matrix
  -> contiguous send scratch
  -> NCCL send/recv
  -> contiguous receive scratch
  -> target rectangle in matrix
```

`cudaMemcpy2DAsync` performs the column-major rectangle pack and unpack.  The
native scheduler then submits the communication through grouped NCCL send/recv
calls using the borrowed XLA-owned NCCL communicator.  This avoids allocating a
second full matrix while still allowing strided row-owner slabs to be moved as
single logical rectangles.

The diagnostic handlers remain useful when changing this code:

1. `xla_rect_pack_unpack_probe` isolates the local column-major addressing.
2. `xla_rect_transfer_probe` isolates one packed rectangle transfer.
3. `xla_rect_2d_native_plan` validates the tile-aligned slab scheduler.
4. `xla_rect_padded_2d_native_plan` is the production redistribution handler
   used by `jaxmg.potrs`.

The production roadmap from the single-node native NCCL/cuSOLVERMp path to
multi-node validation is documented in `cusolvermp_native_nccl_plan.md`.
