# cuSOLVERMp Multi-Node Validation Plan

The current `potrs_mp` implementation is a single-node cuSOLVERMp path. It uses
the XLA-owned NCCL communicator for native 2D redistribution and passes the
borrowed `ncclComm_t` to cuSOLVERMp, but it has not yet been validated across
multiple hosts.

The multi-node work should be staged in SPMD first. In this mode every host runs
the same Python program after `jax.distributed.initialize()`, and the global JAX
mesh spans all processes and devices. The intended rank mapping remains:

```text
rank = process_row * process_cols + process_col
```

The JAX mesh must be built explicitly so that this row-major order matches the
cuSOLVERMp process grid created with `CUSOLVERMP_GRID_MAPPING_ROW_MAJOR`.

## SPMD Checkpoints

The first multi-node checkpoints should be diagnostic and small:

1. Run the existing XLA communicator probe across all global devices.
2. Run the raw NCCL validation path across all global devices.
3. Run one rectangle transfer that crosses hosts.
4. Run forward plus reverse padded 2D redistribution across hosts.
5. Run a tiny `potrs_mp` solve across hosts.
6. Repeat the tiny solve many times to check communicator lifetime, workspace
   allocation, and memory leaks.

The first four checkpoints isolate communication and layout. The fifth is the
first true cuSOLVERMp multi-node solver check.

## MPMD Decision

MPMD should not be copied directly from the cuSolverMg compatibility backend.
The old MPMD cuSolverMg path needed CUDA IPC pointer exchange because cuSolverMg
is a single-node library and one host process had to assemble pointer arrays for
all local GPUs.

cuSOLVERMp is different: it is designed around a distributed process grid and a
communicator. If JAX is launched as one process per GPU, the preferred MPMD
design should still be a global cuSOLVERMp grid using the XLA-owned NCCL
communicator, not a node-local cuSolverMg-style pointer-table workaround.

The practical order is therefore:

1. make SPMD multi-node work first;
2. document the exact JAX distributed launch contract;
3. test whether the same native FFI handlers work unchanged when each process
   owns one local GPU; and
4. only add MPMD-specific code if the distributed JAX runtime exposes different
   stream, mesh, or communicator behavior that the SPMD path cannot handle.

## Current Unsupported Cases

The branch should fail clearly or remain undocumented for these cases until
they are validated:

1. multi-node `potrs_mp`;
2. non-row-major cuSOLVERMp process-grid mapping;
3. `MB_A != NB_A`;
4. cuSOLVERMp `syevd` / `syevd_no_V`;
5. a cuSOLVERMp replacement for `potri`.
