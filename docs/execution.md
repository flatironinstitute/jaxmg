# Distributed execution

JAXMg's cuSOLVERMp backend uses one Python process per GPU. Each process is one
communicator rank and must see exactly one local GPU:

```text
Python rank 0 -> GPU 0
Python rank 1 -> GPU 1
Python rank 2 -> GPU 2
...
```

The backend validates this requirement when it is initialized. A process that
sees more than one local GPU receives a `RuntimeError` before a solver is
called.

In JAXMg's launch terminology, MPMD means this rank-per-GPU configuration and
SPMD means one Python process controlling several local GPUs. These labels
describe the process launch model; they are separate from XLA's broader use of
SPMD to describe compiler partitioning.

The SPMD launch mode is not supported by JAXMg's cuSOLVERMp numerical routines.
The concrete process-to-device mapping is therefore the important requirement:
one Python process, one communicator rank, and one GPU.

## Initializing JAX

Initialize distributed JAX in every process before constructing the mesh or
importing code that queries GPU devices:

```python
import jax

jax.distributed.initialize()
```

On a supported cluster launcher, JAX can infer the coordinator address, process
count, and process identifier. Otherwise pass them explicitly:

```python
jax.distributed.initialize(
    coordinator_address=coordinator_address,
    num_processes=num_processes,
    process_id=process_id,
    local_device_ids=[local_device_id],
)
```

After initialization, `jax.devices()` contains the global devices visible to
the distributed program, while `jax.local_devices()` contains the single GPU
owned by the current process.

## Constructing the process grid

JAXMg uses an ordinary two-dimensional JAX mesh:

```python
from jax.sharding import PartitionSpec as P

num_processes = jax.process_count()
mesh = jax.make_mesh((num_processes, 1), ("pr", "pc"))
matrix_specs = P("pr", "pc")
```

The mesh dimensions define the cuSOLVERMp process grid. A `(P, 1)` or `(1, P)`
mesh is a valid degenerate 2D grid; balanced grids such as `(2, 4)` are also
supported when the product equals the number of participating ranks.

JAXMg accepts regular row-major and column-major device orderings. Arbitrary
rank permutations are rejected because cuSOLVERMp exposes standard process-grid
mapping modes rather than a general rank-to-coordinate table.

## Launching with Slurm

A typical two-node, eight-GPU launch uses eight tasks with one GPU per task:

```bash
srun \
  --nodes=2 \
  --ntasks=8 \
  --ntasks-per-node=4 \
  --gpus-per-task=1 \
  python solve.py
```

Exact GPU flags vary between clusters. The invariant is that each task receives
one GPU and all ranks enter the same JAX distributed computation.

## Mesh and right-hand-side sharding

Coefficient matrices are normally sharded over both process-grid axes:

```python
from jax.sharding import NamedSharding

a_sharding = NamedSharding(mesh, P("pr", "pc"))
a = jax.device_put(a, a_sharding)
```

A vector or narrow right-hand side does not need to be sharded over the process
column axis. Replicated and row-sharded right-hand sides are supported:

```python
b_vector = jax.device_put(b_vector, NamedSharding(mesh, P(None)))
b_matrix = jax.device_put(b_matrix, NamedSharding(mesh, P("pr", None)))
```

The solver wrapper pads and redistributes the right-hand side internally for
cuSOLVERMp.
