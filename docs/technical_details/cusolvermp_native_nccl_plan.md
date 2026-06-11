# Native NCCL cuSOLVERMp Implementation Plan

This document records the intended path from the current Python-orchestrated
2D redistribution prototype to the production cuSOLVERMp backend.

The end goal is:

```text
JAX block-sharded input
  -> donated local GPU buffers
  -> native 2D block-cyclic redistribution
  -> cuSOLVERMp factor/solve
  -> optional native reverse redistribution
  -> JAX-facing output
```

The production path should not allocate a second full distributed matrix. It
should use bounded per-rank scratch and should execute the redistribution inside
one native FFI handler.

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

The current executor is still a bridge implementation. Python builds and loops
over transfer batches, while the native backend executes one rectangle-transfer
shape group at a time.

## Target Native Redistribution

The native redistribution should receive:

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

It should then execute:

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

Each rank gets a local scratch buffer split into two slots:

```text
scratch_send = S[0 : max_fragment_elements]
scratch_recv = S[max_fragment_elements : 2 * max_fragment_elements]
```

This is sufficient only because each execution batch maintains:

```text
each rank appears as source at most once
each rank appears as target at most once
```

The scratch size should be computed conservatively by Python at first and
verified inside native code. Later the native handler can expose a helper/query
to compute the required size from matrix shape, tile shape, and process grid.

## NCCL Execution Model

The current prototype uses XLA `CollectivePermute` for each transfer group. A
fully optimized direct-NCCL implementation should use the same batch structure
but submit the communication as grouped NCCL calls:

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

After the NCCL group completes on the communication stream, the compute stream
unpacks `scratch_recv` into the final destination rectangle.

The direct-NCCL path depends on proving that the borrowed XLA communicator
exposes a valid `ncclComm_t` for the CUDA backend and that its lifetime and
stream ordering rules are safe for this use.

## Implementation Stages

### Stage 1: Column-Major Executor Validation

Validate the current Python-orchestrated executor with `layout="column_major"`.
This proves the 2D schedule works with the same local memory convention that
cuSOLVERMp expects.

Required tests:

1. `2 x 2` GPU executor with column-major layout.
2. CPU reference that checks final local buffers using:

   ```text
   offset = local_col * local_rows + local_row
   ```

3. Non-square local shard cases if supported by the process grid and tile shape.

### Stage 2: Independent cuSOLVERMp Layout Reference

Compare our final local buffers against an independent reference.

Preferred reference:

1. Use NVIDIA's `cusolverMpMatrixScatterH2D` on a tiny matrix.
2. Gather or inspect the resulting local device buffers.
3. Compare against JAXMg's redistributed buffers.

Fallback reference:

1. Implement a CPU ScaLAPACK-style 2D block-cyclic local-buffer reference.
2. Validate owner rank, local coordinates, and column-major physical offsets.

This stage should settle whether our rank ordering,
`rank = process_row * P_c + process_col`, matches the cuSOLVERMp grid we create.

### Stage 3: Native Schedule Builder

Move the schedule construction from Python to C++.

Native code should construct:

1. tile extents;
2. split tile fragments where initial JAX shards cross tile boundaries;
3. source and target local rectangles;
4. column-owner and row-owner transfers;
5. conflict-free batches grouped by rectangle shape.

Python should pass only stable scalar metadata:

```text
logical rows/cols
local padded rows/cols
MB_A / NB_A
P_r / P_c
layout
```

This removes Python from the inner redistribution loop.

### Stage 4: Native XLA-Collective Executor

Before switching to raw NCCL, build a fused native executor using the existing
XLA `CollectivePermute` abstraction.

This stage should:

1. allocate or receive bounded scratch;
2. build all batches natively;
3. execute all pack -> `CollectivePermute` -> unpack steps inside one FFI call;
4. preserve stream ordering without a full device synchronization unless needed.

This gives a fair baseline for the fully native implementation while still
using the safer XLA communicator abstraction.

### Stage 5: Raw `ncclComm_t` Diagnostic

Add a diagnostic-only FFI target that proves:

1. `platform_comm().handle` is present;
2. the handle is actually an NCCL communicator on CUDA;
3. `ncclCommUserRank`, `ncclCommCount`, or an equivalent NCCL sanity call works;
4. a tiny direct NCCL send/recv or all-reduce succeeds using the borrowed handle.

This target should not change solver behavior. It is purely a safety gate before
direct NCCL is used for redistribution or passed to cuSOLVERMp.

### Stage 6: Native NCCL Batch Executor

Replace the per-batch communication implementation with direct NCCL
`ncclGroupStart` / `ncclSend` / `ncclRecv` / `ncclGroupEnd`.

Keep the same schedule and scratch invariants:

```text
one send slot per rank
one receive slot per rank
no duplicate source or target ranks inside a batch
```

Validate against the Stage 4 XLA-collective executor.

### Stage 7: cuSOLVERMp Handle and Descriptor Integration

Create the cuSOLVERMp objects after redistribution:

```text
cusolverMpHandle_t
cusolverMpGrid_t
cusolverMpMatrixDescriptor_t
```

This stage is deliberately narrower than "port all solvers". The first target
is `potrs`, because cuSOLVERMp directly supports the same Cholesky sequence used
by JAXMg:

```text
potrf
potrs
```

The first diagnostic should:

1. borrow the XLA-owned `ncclComm_t`;
2. create the cuSOLVERMp handle and grid on the callback stream;
3. create matrix descriptors for `A` and `B`;
4. query `cusolverMpPotrf_bufferSize` and `cusolverMpPotrs_bufferSize`;
5. allocate bounded host/device workspace;
6. scatter a tiny host matrix with `cusolverMpMatrixScatterH2D`;
7. run `cusolverMpPotrf` and `cusolverMpPotrs`.

That isolates the cuSOLVERMp boundary. The second diagnostic replaces the input
scatter with the JAXMg path:

1. create ordinary block-sharded JAX buffers;
2. edge-pad them according to the 2D redistribution planner;
3. run the native 2D GPU redistribution for both `A` and `B`;
4. create cuSOLVERMp descriptors over those redistributed local buffers;
5. run `cusolverMpPotrf` and `cusolverMpPotrs`;
6. gather only the solved `B` for residual checking.

This is the first end-to-end layout test for the intended production input
path. It is still diagnostic-only: the public `potrs` wrapper needs a dedicated
RHS padding policy, repeated-call memory handling, and output redistribution
before it should replace the cuSolverMg backend.

Only after this succeeds should the branch add `syevd`/`syevd_no_V`.
`cusolverMpSyevd` maps directly to both APIs through `jobz = "V"` and
`jobz = "N"`. `potri` does not have a direct `cusolverMpPotri` equivalent in
the current cuSOLVERMp C API documentation, so it needs a separate design
decision before migration. The first cuSOLVERMp backend should not support
`potri`; it should fail clearly or stay on a separate legacy path rather than
silently emulating an inverse with a large distributed identity solve.

The first goal is correctness on one node. Multi-node validation comes after the
single-node descriptor/layout semantics are proven.

### Stage 8: Reverse Redistribution

Decide what the public API should return.

For JAXMg compatibility, solver outputs should likely be converted back to a
normal JAX-facing sharding unless the user explicitly opts into cuSOLVERMp local
layout outputs.

Reverse redistribution can reuse the same schedule with source and target
rectangles swapped and phases reversed:

```text
row_owner reverse
then column_owner reverse
```

### Stage 9: Multi-Node Validation

Once single-node direct NCCL and cuSOLVERMp calls work:

1. run the communicator diagnostic across nodes;
2. run rectangle transfer across nodes;
3. run full 2D redistribution across nodes;
4. run tiny cuSOLVERMp solve across nodes;
5. stress repeated solves for communicator lifetime and memory leaks.

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
single-node cuSOLVERMp correctness
clean path to multi-node NCCL
```

## Open Risks

1. **Borrowed communicator lifetime:** XLA owns the communicator. We need a
   strict rule for when the raw NCCL handle is valid.
2. **Stream ordering:** cuSOLVERMp and direct NCCL must run on streams ordered
   consistently with XLA's FFI stream contexts.
3. **Rank ordering:** our process-grid rank formula must match the cuSOLVERMp
   grid construction.
4. **Column-major JAX buffers:** the full executor still needs column-major
   validation and an independent reference.
5. **Padding and split tiles:** support can start conservatively, but rejected
   cases must fail clearly before touching device memory.
