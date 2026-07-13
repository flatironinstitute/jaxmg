# Distributed Execution

JAXMg uses one Python process per GPU. Each process contributes one rank to the
JAX/XLA communicator borrowed by the native backend and must see exactly one
local GPU.

This guide uses two nodes with four GPUs per node, giving eight processes and
eight global devices:

```text
node 0                              node 1
-------------------------           -------------------------
rank 0 -> local GPU 0               rank 4 -> local GPU 0
rank 1 -> local GPU 1               rank 5 -> local GPU 1
rank 2 -> local GPU 2               rank 6 -> local GPU 2
rank 3 -> local GPU 3               rank 7 -> local GPU 3
```

The job launcher normally restricts each process to its assigned GPU. JAXMg
checks this requirement during initialization and raises a `RuntimeError` before
calling a solver if a process can see more than one local GPU.

All eight processes run the same Python program and must enter distributed JAX
and JAXMg collective operations in the same order.

## 1. Initialize distributed JAX

Call [`jax.distributed.initialize()`](https://docs.jax.dev/en/latest/_autosummary/jax.distributed.initialize.html)
before constructing the mesh or performing any JAX computation:

```python
import jax

jax.distributed.initialize()
```

When the program is launched with Slurm, Open MPI, or another supported cluster
environment, JAX can infer the distributed configuration. This is the preferred
form for the Slurm example below.

If automatic discovery is unavailable, provide the configuration explicitly:

```python
jax.distributed.initialize(
    coordinator_address="node-0-hostname:12345",
    num_processes=8,
    process_id=process_id,
    local_device_ids=[local_device_id],
)
```

The values have the following meanings:

- `coordinator_address` is the hostname or IP address of rank 0 followed by an
  available port. Every process must use the same address. Do not use
  `localhost` for a multi-node job.
- `num_processes` is the total number of Python processes participating in the
  computation: eight in this example.
- `process_id` is the unique global rank of the current process, from `0` to
  `num_processes - 1`.
- `local_device_ids` identifies the GPU assigned to the current process on its
  node. A supported Slurm or Open MPI launch normally determines this
  automatically.

## 2. Check the process and device view

After initialization, each process should report one local device, while the
global JAX client should contain all eight devices:

```python
print(
    f"rank {jax.process_index()} of {jax.process_count()}: "
    f"{jax.local_device_count()} local GPU, "
    f"{jax.device_count()} global GPUs"
)
```

The output from rank 0 should have the form

```text
rank 0 of 8: 1 local GPU, 8 global GPUs
```

`jax.local_devices()` contains the GPU owned by the current process.
`jax.devices()` contains the global devices participating across both nodes.

## 3. Construct the process grid

JAXMg uses an ordinary two-dimensional JAX mesh. For eight ranks, valid grid
shapes include $(8,1)$, $(4,2)$, $(2,4)$, and $(1,8)$. The following code
constructs a $4\times2$ grid:

```python
from jax.sharding import PartitionSpec as P


process_rows = 4
process_cols = 2
mesh = jax.make_mesh((process_rows, process_cols), ("pr", "pc"))
matrix_specs = P("pr", "pc")
```

The mesh dimensions become the cuSOLVERMp process-grid dimensions. The first
axis, `pr`, identifies process rows; the second, `pc`, identifies process
columns.

JAXMg accepts regular row-major and column-major rank mappings. For a
$4\times2$ grid these are

$$
\text{row-major}
=
\begin{bmatrix}
0 & 1\\
2 & 3\\
4 & 5\\
6 & 7
\end{bmatrix},
\qquad
\text{column-major}
=
\begin{bmatrix}
0 & 4\\
1 & 5\\
2 & 6\\
3 & 7
\end{bmatrix}.
$$

You can inspect the process-rank mapping selected by JAX:

```python
import numpy as np


rank_grid = np.asarray(
    [[device.process_index for device in row] for row in mesh.devices]
)
if jax.process_index() == 0:
    print(rank_grid)
```

To request an explicit regular mapping, construct the JAX mesh from devices
sorted by process rank:

```python
from jax.sharding import Mesh


devices = np.asarray(
    sorted(jax.devices(), key=lambda device: device.process_index),
    dtype=object,
)

row_major_mesh = Mesh(
    devices.reshape((process_rows, process_cols), order="C"),
    ("pr", "pc"),
)
column_major_mesh = Mesh(
    devices.reshape((process_rows, process_cols), order="F"),
    ("pr", "pc"),
)
```

Use one of these meshes as `mesh`. Arbitrary rank permutations are rejected
because cuSOLVERMp supports standard row-major and column-major process-grid
mappings rather than a general rank-to-coordinate table.

## 4. Place the matrix and right-hand side

`PartitionSpec` describes how each array dimension maps onto the process-grid
axes. A coefficient matrix is normally sharded over both axes:

```python
import jax.numpy as jnp
from jax.sharding import NamedSharding


N = 8192
a = jnp.eye(N, dtype=jnp.float32)
a_sharding = NamedSharding(mesh, P("pr", "pc"))
a = jax.device_put(a, a_sharding)
```

On the $4\times2$ grid, each process initially owns an
$(N/4)\times(N/2)$ rectangular shard of `a`.

A vector right-hand side can be sharded over process rows and replicated across
process columns:

```python
b_vector = jnp.ones((N,), dtype=a.dtype)
b_vector = jax.device_put(b_vector, NamedSharding(mesh, P("pr")))
```

For an $N\times\mathrm{NRHS}$ matrix right-hand side, shard the matrix rows and
leave its columns replicated:

```python
b_matrix = jnp.ones((N, 1), dtype=a.dtype)
b_matrix = jax.device_put(b_matrix, NamedSharding(mesh, P("pr", None)))
```

The public solver accepts either representation. JAXMg adds any routing or tile
padding required for a narrow right-hand side and redistributes it internally
for cuSOLVERMp.

## 5. Launch the program with Slurm

Save the initialization, mesh construction, array placement, and solver call in
the same Python script, for example `solve.py`. A typical two-node, eight-GPU
launch is

```bash
srun \
  --nodes=2 \
  --ntasks=8 \
  --ntasks-per-node=4 \
  --gpus-per-task=1 \
  python -u solve.py
```

Slurm flag names vary between clusters, but the required mapping does not: run
eight tasks, assign one GPU to each task, and execute the same program on every
rank. Each process calls `jax.distributed.initialize()`, constructs the same
global mesh, and enters the same solver call.

With distributed execution configured, continue to [Choose a tile size
$T_A$](choose_tile_size.md) before selecting a solver example.
