# Native workflow

JAXMg accepts ordinary JAX arrays sharded over a two-dimensional device mesh.
However, cuSOLVERMp requires column-major local buffers distributed in a 2D
block-cyclic layout. The native backend bridges these layouts inside one fused
C++/CUDA FFI call:

```text
JAX block-sharded arrays
        |
        | Python validation and local tile-capacity padding
        v
fused native FFI call
        |
        | 1. local row-major -> column-major conversion
        | 2. top-left / edge-padding alignment
        | 3. 2D block-cyclic redistribution
        v
cuSOLVERMp solver
        |
        | reverse 3 -> reverse 2 -> reverse 1
        v
JAX-facing result
```

Python calculates static metadata such as the process grid, rank map, logical
matrix size, tile size, and padded local capacity. C++/CUDA owns the data
movement, cuSOLVERMp descriptors and workspace, solver calls, and reverse
redistribution. Matrix data remains GPU-resident throughout this workflow, and
the in-place transformations reuse bounded native scratch storage to minimize
memory overhead.

Matrix-valued solver outputs follow the reverse path shown above. The
eigenvalues-only SYEVD mode returns the replicated eigenvalue array directly
after solver execution.

## Scratch allocation

The redistribution path uses one native scratch allocation per FFI call. Its
size is determined by the largest tile slab used by the 2D block-cyclic stage:

$$
N_{\mathrm{scratch}}
=
3\max\left(
  T_c N_{\mathrm{local\ rows}},
  T_r N_{\mathrm{local\ cols}}
\right).
$$

For square solver tiles, $T_r=T_c=T_A$. The layout-conversion and edge-padding
stages reuse this allocation and process data in bounded batches. They do not
allocate a second full local matrix.

## Memory-distribution stages

??? info "Stage 1: local memory layout conversion"
    JAX presents each local matrix shard in row-major physical memory.
    cuSOLVERMp expects column-major local memory. CUDA kernels apply an in-place
    rectangular permutation using bounded scratch; the logical matrix is not
    transposed. This stage is local to each GPU and requires no communicator
    traffic.

    [Layout-conversion details](memory_distribution.md#stage-1-local-layout-conversion)

??? info "Stage 2: top-left and edge-padding alignment"
    Python provides tile-aligned local capacity. Native horizontal compaction
    moves real column slabs toward the global left edge, then vertical
    compaction moves real row slabs toward the global top edge. Padding is
    consolidated on the global right and bottom edges.

    Moves in one horizontal wave are independent across process rows. Moves in
    one vertical wave are independent across process columns.

    [![Three stages of edge-padding alignment over a two-by-four GPU process grid.](../_static/memory_distribution/block_cyclic_padding_alignment.svg){ .memory-distribution-image }](../_static/memory_distribution/block_cyclic_padding_alignment.svg)

    [Edge-padding details](memory_distribution.md#stage-2-top-left-edge-padding-alignment)

??? info "Stage 3: 2D block-cyclic redistribution"
    Whole tile slabs are redistributed in two separable phases. The
    column-owner phase assigns each tile to the correct process column; the
    row-owner phase then assigns each tile to the correct process row.

    Same-rank moves use local CUDA operations. Cross-rank moves use the
    NCCL-backed communicator borrowed from XLA. The same communicator supplies
    the `ncclComm_t` used by cuSOLVERMp.

    [![Three stages of the two-dimensional block-cyclic redistribution over a two-by-four GPU process grid.](../_static/memory_distribution/block_cyclic_redistribution.svg){ .memory-distribution-image }](../_static/memory_distribution/block_cyclic_redistribution.svg)

    [2D block-cyclic details](memory_distribution.md#stage-3-2d-block-cyclic-redistribution)

## Solver workflows

The redistribution stages are shared by all public routines:

- `potrs` calls `cusolverMpPotrf` followed by `cusolverMpPotrs`. When requested,
  it reads the distributed factor diagonal and performs a one-scalar NCCL
  all-reduce to return `log(det(A))`.
- `lu_solve` calls `cusolverMpGetrf` followed by `cusolverMpGetrs` and manages
  the distributed pivot allocation.
- `syevd` calls `cusolverMpSyevd` and materializes distributed eigenvalues plus
  eigenvectors when requested.
- `gesvd` calls `cusolverMpGesvd` for rectangular matrices and restores only
  the requested U and Vh outputs. Its shared scratch allocation is sized to the
  largest redistribution requirement among A and those requested outputs.

Each Python process owns one GPU and contributes one rank to the XLA/NCCL and
cuSOLVERMp communicators. See [Distributed execution](../examples/execution.md)
for the required launch model.

## Detailed implementation

[Memory distribution](memory_distribution.md) gives the complete diagrams,
layout definitions, dependency waves, permutation cycles, reverse path, and
source-code map.
