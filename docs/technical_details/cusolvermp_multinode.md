# cuSOLVERMp Multi-Node Validation Plan

The current cuSOLVERMp backend provides `potrs` and `syevd`. It uses the
XLA-owned NCCL communicator for native 2D redistribution and passes the
borrowed `ncclComm_t` to cuSOLVERMp. Small two-node validation is part of the
development test suite; larger multi-node benchmarks are still a release
hardening task.

The multi-node work should be staged with ordinary JAX distributed SPMD: every
rank runs the same Python program after `jax.distributed.initialize()`, and the
global JAX mesh spans the participating devices. cuSOLVERMp's CAL/NCCL model is
rank-oriented, so the supported solver launch contract is one Python process per
GPU. The intended rank mapping remains:

```text
rank = process_row * process_cols + process_col
```

The JAX mesh must be built explicitly so that this row-major order matches the
cuSOLVERMp process grid created with `CUSOLVERMP_GRID_MAPPING_ROW_MAJOR`.

## Rank-Per-GPU Launch Contract

The supported cuSOLVERMp launch mode is one Slurm task per GPU:

```bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1

srun python run_potrs_multinode.py
```

User code should call `jax.distributed.initialize()` before any operation that
may initialize the JAX backend, including `jax.devices()`, array creation on GPU,
or JIT compilation. JAXMg does not own distributed initialization; it validates
the resulting mesh and runtime contract before calling cuSOLVERMp.

```python
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

import jaxmg

jax.distributed.initialize(
    coordinator_address=coordinator_address,
    num_processes=num_processes,
    process_id=process_id,
    local_device_ids=[local_rank],
)

mesh = jax.make_mesh((2, 4), ("pr", "pc"))
sharding = NamedSharding(mesh, P("pr", "pc"))
```

On some Slurm installations `--gpus-per-task=1` makes each process see exactly
one visible GPU, so `local_device_ids=[0]` is correct. If all local GPUs remain
visible to every process, pass the Slurm local rank instead. `syevd` rejects
one-process-many-GPU launches because cuSOLVERMp SYEVD has proven sensitive to
the one-process-per-GPU assumption.

## Validation Checkpoints

The first multi-node checkpoints should be diagnostic and small:

1. Run the existing XLA communicator probe across all global devices.
2. Run the raw NCCL validation path across all global devices.
3. Run one rectangle transfer that crosses hosts.
4. Run forward plus reverse padded 2D redistribution across hosts.
5. Run tiny `potrs` and `syevd` cases across hosts.
6. Repeat small solves/eigensolves to check communicator lifetime, workspace
   allocation, and memory leaks.

The first four checkpoints isolate communication and layout. The fifth is the
first true cuSOLVERMp multi-node solver check.

## Process Model Decision

The rank-per-GPU path should not be copied from the removed single-node
compatibility backend. That backend needed CUDA IPC pointer exchange because one
host process had to assemble pointer arrays for all local GPUs.

cuSOLVERMp is different: it is designed around a distributed process grid and a
communicator. When JAX is launched as one process per GPU, the design remains a
global cuSOLVERMp grid using the XLA-owned NCCL communicator, not a node-local
pointer-table workaround.

The practical order is therefore:

1. make rank-per-GPU multi-node work first;
2. document the exact JAX distributed launch contract;
3. keep one-process-per-node support only where it has been explicitly
   validated; and
4. only add process-model-specific native code if the distributed JAX runtime
   exposes different stream, mesh, or communicator behavior that the common path
   cannot handle.

## Current Unsupported Cases

The branch should fail clearly or remain undocumented for these cases until
they are validated:

1. large multi-node performance benchmarks;
2. `MB_A != NB_A`;
3. cuSOLVERMp explicit inverse support.
