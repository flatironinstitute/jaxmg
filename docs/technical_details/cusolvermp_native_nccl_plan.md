# Native NCCL cuSOLVERMp Status and Plan

This document records the current native cuSOLVERMp path and the remaining work
needed to harden the multi-node backend for larger problem sizes.

The end goal is:

```text
JAX block-sharded input
  -> donated local GPU buffers
  -> native 2D block-cyclic redistribution
  -> cuSOLVERMp factor/solve
  -> optional native reverse redistribution
  -> JAX-facing output
```

The production path does not allocate a second full distributed matrix. It uses
bounded per-rank scratch and executes the forward and reverse redistribution
inside native FFI handlers.

## Current Checkpoints

The branch has already proved these pieces independently:

1. XLA communicator access works through the Bazel-built native backend.
2. The existing 1D cuSolverMg redistribution can use the XLA communicator
   instead of the old CUDA peer/shared-memory path.
3. A local rectangle can be packed from a rank-local matrix into scratch and
   unpacked elsewhere.
4. A packed rectangle can be sent rank-to-rank using XLA `CollectivePermute`.
5. The Python planner can build executable source/target local rectangles for a
   two-phase 2D block-cyclic redistribution.
6. The Python executor can run those batches on a real `2 x 2` single-node GPU
   mesh.
7. cuSOLVERMp can create a handle/grid from the borrowed XLA/NCCL communicator
   and run `potrf`/`potrs` when NVIDIA's host scatter helper creates the input
   layout.
8. The solver diagnostic can also consume `A` and `B` buffers produced by
   JAXMg's native 2D redistribution, avoiding `cusolverMpMatrixScatterH2D` on
   the input side.
9. The native padded 2D redistribution now has a reverse mode. It applies the
   inverse slab scheduler first and then reverses edge-padding compaction so a
   cuSOLVERMp-layout output can be returned to a JAX-facing block-sharded
   logical layout.
10. `jaxmg.potrs_mp` wires the Cholesky solve path: local shard padding,
    forward 2D redistribution, production `cusolvermp_potrs`, reverse 2D
    redistribution, and local unpadding of the solved right-hand side.
11. `jaxmg.syevd_mp` has a native investigation path on the same redistribution
    backend, but it is not yet in the same validation state as `potrs_mp`. The
    current work is isolating cuSOLVERMp 0.7.2 `Syevd` behavior with the
    installed `compz` API, the XLA-owned NCCL communicator, and the FFI stream.

The current production redistribution path no longer loops over Python-planned
rectangle batches. The native backend builds and executes the slab schedule in
one FFI call, using bounded per-rank scratch and direct NCCL send/recv through
the borrowed XLA communicator. The lower-level Python-planned rectangle
executor remains only as a diagnostic test path.

## Native Redistribution Contract

The native redistribution receives:

```text
matrix buffer A
scratch buffer S
logical matrix shape
local padded shape
tile shape MB_A x NB_A
process grid P_r x P_c
rank identity
XLA/NCCL communicator
CUDA/XLA streams
```

It then executes:

```text
for phase in [column_owner, row_owner]:
    build conflict-free batches
    for batch in phase_batches:
        pack one rectangle per sending rank
        communicate packed rectangles
        unpack one rectangle per receiving rank
```

The target owner rule is:

```text
owner_process_row = tile_row % P_r
owner_process_col = tile_col % P_c
```

The final local coordinate rule is:

```text
local_row = (tile_row // P_r) * MB_A + row_inside_tile
local_col = (tile_col // P_c) * NB_A + col_inside_tile
```

For cuSOLVERMp local matrices, the physical address is column-major:

```text
offset = local_col * local_rows + local_row
```

## Scratch Policy

Each rank gets a local scratch buffer split into three slots:

```text
scratch_saved = S[0 : max_step_elements]
scratch_send  = S[max_step_elements : 2 * max_step_elements]
scratch_recv  = S[2 * max_step_elements : 3 * max_step_elements]
```

The saved slot is needed for closed permutation cycles. This remains bounded
because each execution batch maintains:

```text
each rank appears as source at most once
each rank appears as target at most once
```

The scratch size should be computed conservatively by Python at first and
verified inside native code. Later the native handler can expose a helper/query
to compute the required size from matrix shape, tile shape, and process grid.

## NCCL Execution Model

The production path uses the borrowed XLA-owned `ncclComm_t` directly. Each
conflict-free batch is submitted as grouped NCCL send/recv calls:

```c
ncclGroupStart();

if (rank sends in this batch) {
  ncclSend(scratch_send, count, dtype, target_rank, comm, stream);
}
if (rank receives in this batch) {
  ncclRecv(scratch_recv, count, dtype, source_rank, comm, stream);
}

ncclGroupEnd();
```

After the NCCL group is ordered on the chosen stream, the receive payload is
unpacked into the final destination rectangle.  The native code validates the
borrowed communicator with `ncclCommUserRank` and `ncclCommCount` before using
it.

The remaining risk is no longer whether the handle is NCCL on the CUDA backend.
The remaining risk is lifetime and ordering in larger distributed jobs: XLA
still owns the communicator, and JAXMg only borrows it during the FFI call.

## Implemented Single-Node Stages

The `cusolvermp_multi_node` branch has reached these stages on a single
multi-GPU node:

1. Column-major local layout validation against a CPU reference.
2. Independent cuSOLVERMp scatter-layout validation using
   `cusolverMpMatrixScatterH2D`.
3. Native schedule construction in C++ from scalar metadata.
4. Native edge-padding compaction and 2D slab redistribution.
5. Raw NCCL validation through the borrowed XLA communicator.
6. Native grouped NCCL send/recv execution with bounded scratch.
7. cuSOLVERMp `potrf`/`potrs` over redistributed JAX buffers.
8. Reverse native redistribution back to the JAX-facing block-sharded layout.

The schedule and scratch invariants are:

```text
one send slot per rank
one receive slot per rank
no duplicate source or target ranks inside a batch
```

## Current Production Solver Paths

The production cuSOLVERMp FFI target creates the cuSOLVERMp objects after
redistribution:

```text
cusolverMpHandle_t
cusolverMpGrid_t
cusolverMpMatrixDescriptor_t
```

The first target was `potrs`, because cuSOLVERMp directly supports the same
Cholesky sequence used by JAXMg:

```text
potrf
potrs
```

Current status:

```text
jaxmg.potrs_mp
  -> pad local 2D shards
  -> native padded 2D redistribution
  -> cusolverMpPotrf + cusolverMpPotrs
  -> reverse native padded 2D redistribution
  -> unpad local RHS shards
```

The Cholesky cuSOLVERMp FFI target is registered as `cusolvermp_potrs`. The
eigensolver target is registered as `cusolvermp_syevd`:

```text
jaxmg.syevd_mp(eigvecs=True) [under validation]
  -> pad local 2D shards
  -> native padded 2D redistribution
  -> cusolverMpSyevd(compz = "Z" on cuSOLVERMp 0.7.2)
  -> reverse native padded 2D redistribution of eigenvectors
  -> unpad local eigenvector shards

jaxmg.syevd_mp(eigvecs=False) [not supported on cuSOLVERMp 0.7.2 yet]
  -> pad local 2D shards
  -> native padded 2D redistribution
  -> cusolverMpSyevd(compz = "N")
  -> replicated eigenvalues only
```

The installed cuSOLVERMp 0.7.2 runtime currently reports that eigenvalue-only
SYEVD is not supported for `compz = "N"`, so the no-vector path should stay
guarded or disabled unless a newer runtime proves otherwise.

The older `cusolvermp_*_probe` functions remain internal diagnostics and are no
longer top-level `jaxmg` exports. Explicit inverse support is intentionally not
part of this cuSOLVERMp backend because the current cuSOLVERMp C API
documentation does not expose a direct explicit-inverse routine. JAXMg should
not silently emulate an inverse with a large distributed identity solve.

## Remaining Multi-Node Stages

The next target is broader SPMD multi-node validation. A distributed JAX job
should:

1. run the communicator diagnostic across nodes;
2. run rectangle transfer across nodes;
3. run full 2D redistribution across nodes;
4. run tiny cuSOLVERMp solve and eigensolver cases across nodes;
5. stress repeated solves/eigensolves for communicator lifetime and memory
   leaks.

MPMD support should come after the SPMD path is stable. The old cuSolverMg MPMD
path needed node-scoped host pointer exchange because cuSolverMg is single-node.
cuSOLVERMp is a distributed library, so MPMD support should be designed around a
global cuSOLVERMp process grid instead of copying the old node-local pointer
exchange scheme blindly.

## Non-Goals for the First Production Cut

The first optimized path does not need to solve every performance problem:

1. It does not need an optimal global all-to-all schedule.
2. It does not need to fuse differently shaped rectangle fragments into one
   custom kernel.
3. It does not need to support every possible padding edge case if the API
   clearly rejects unsupported layouts first.

The first production target should be:

```text
correct 2D block-cyclic column-major layout
bounded scratch
native schedule execution
single-node and small multi-node cuSOLVERMp correctness
clean path to larger multi-node benchmarking
```

## Open Risks

1. **Borrowed communicator lifetime:** XLA owns the communicator. We need a
   strict rule for when the raw NCCL handle is valid.
2. **Stream ordering:** cuSOLVERMp and direct NCCL must run on streams ordered
   consistently with XLA's FFI stream contexts.
3. **Rank ordering:** our process-grid rank formula must match the cuSOLVERMp
   grid construction.
4. **Scale testing:** tiny multi-node cases are not a substitute for larger
   runtime and memory-pressure benchmarks.
5. **Padding and split tiles:** validated cases should keep expanding, and
   unsupported layouts must fail clearly before touching device memory.
