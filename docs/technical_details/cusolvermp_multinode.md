# cuSOLVERMp Multi-Node Validation Plan

The current cuSOLVERMp backend provides `potrs_mp` and `syevd_mp`. It uses the
XLA-owned NCCL communicator for native 2D redistribution and passes the
borrowed `ncclComm_t` to cuSOLVERMp. Small two-node validation is part of the
development test suite; larger multi-node benchmarks are still a release
hardening task.

The multi-node work should be staged in SPMD first. In this mode every host runs
the same Python program after `jax.distributed.initialize()`, and the global JAX
mesh spans all processes and devices. The intended rank mapping remains:

```text
rank = process_row * process_cols + process_col
```

The JAX mesh must be built explicitly so that this row-major order matches the
cuSOLVERMp process grid created with `CUSOLVERMP_GRID_MAPPING_ROW_MAJOR`.

## One Process Per Node Launch Contract

The first supported multi-node launch mode is one Slurm task per node:

```bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4

srun python run_potrs_mp_multinode.py
```

Each Python process should call `jaxmg.initialize_node_process()` before any
operation that may initialize the JAX backend, including `jax.devices()`, array
creation on GPU, or JIT compilation:

```python
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

import jaxmg

jaxmg.initialize_node_process()

mesh = jaxmg.make_cusolvermp_mesh(2, 4)
sharding = NamedSharding(mesh, P("pr", "pc"))
```

On Slurm and Open MPI, JAX can often infer `coordinator_address`,
`num_processes`, and `process_id` automatically. JAXMg still passes
`local_device_ids` explicitly because GPU Slurm/Open MPI launches can otherwise
default to one local device per process. If the scheduler environment does not
make the local GPU count visible, pass `local_device_ids` directly or set
`JAXMG_LOCAL_DEVICE_COUNT`.

`initialize_node_process()` also sets:

```text
JAXMG_EXECUTION_MODE=SPMD
JAXMG_NUMBER_OF_DEVICES=<local GPU count>
```

This keeps the backend in node-local SPMD mode even though
`jax.distributed.initialize()` has been called. Explicit one-process-per-GPU
MPMD support should use `JAXMG_EXECUTION_MODE=MPMD` once that mode is validated
for the cuSOLVERMp path.

## SPMD Checkpoints

The first multi-node checkpoints should be diagnostic and small:

1. Run the existing XLA communicator probe across all global devices.
2. Run the raw NCCL validation path across all global devices.
3. Run one rectangle transfer that crosses hosts.
4. Run forward plus reverse padded 2D redistribution across hosts.
5. Run tiny `potrs_mp` and `syevd_mp` cases across hosts.
6. Repeat small solves/eigensolves to check communicator lifetime, workspace
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

1. large multi-node performance benchmarks;
2. `MB_A != NB_A`;
3. cuSOLVERMp explicit inverse support.
