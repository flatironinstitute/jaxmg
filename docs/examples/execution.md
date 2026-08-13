# Distributed Execution

JAXMg requires one Python process per GPU. It checks this requirement during a
solver call and raises a `RuntimeError` if the execution setup is unsupported.

This guide assumes two nodes with four GPUs per node, giving eight global
devices:

```text
node 0                              node 1
-------------------------           -------------------------
rank 0 -> local GPU 0               rank 4 -> local GPU 0
rank 1 -> local GPU 1               rank 5 -> local GPU 1
rank 2 -> local GPU 2               rank 6 -> local GPU 2
rank 3 -> local GPU 3               rank 7 -> local GPU 3
```

## 1. Initialize distributed JAX

Call [`jax.distributed.initialize()`](https://docs.jax.dev/en/latest/_autosummary/jax.distributed.initialize.html)
before constructing the mesh or performing any JAX computation:

```python
import jax

jax.distributed.initialize()
```

While JAX supports automatic detection of the distributed configuration, the user can provide the configuration explicitly:

```python
jax.distributed.initialize(
    coordinator_address="node-0-hostname:12345",  # Rank 0 host and shared port.
    num_processes=8,  # Total number of Python processes.
    process_id=process_id,  # Unique global rank of this process.
    local_device_ids=[local_device_id],  # Local GPU assigned to this process.
)
```

## 2. Check the process and device view

After initialization, print the process and device information:

```python
print("Process index:", jax.process_index())
print("Process count:", jax.process_count())
print("Local devices:", jax.local_devices())
print("Global devices:", jax.devices())
```

On rank 0, the output should have the following form:

```text
Process index: 0
Process count: 8
Local devices: [CudaDevice(id=0)]
Global devices: [CudaDevice(id=0), ..., CudaDevice(id=7)]
```

## 3. Construct the process grid

JAXMg maps an ordinary one- or two-axis JAX mesh onto cuSOLVERMp's process grid.
For eight ranks, valid grid shapes include $(8,1)$, $(4,2)$, $(2,4)$, and
$(1,8)$. The following code constructs a $4\times2$ grid:

```python
from jax.sharding import PartitionSpec as P


process_rows = 4
process_cols = 2
mesh = jax.make_mesh((process_rows, process_cols), ("pr", "pc"))
jax.set_mesh(mesh)
matrix_specs = P("pr", "pc")
```

These JAX mesh dimensions are used throughout the backend and hence are inherited by cuSOLVERMp process-grid. The first
axis, `pr`, identifies process rows; the second, `pc`, identifies process columns.
Calling `jax.set_mesh(mesh)` makes this concrete mesh available while JAX traces
the explicitly sharded computation.

An existing one-axis mesh can be used directly. To shard matrix rows over its
axis, use a rank-1 `PartitionSpec`:

```python
single_axis_mesh = jax.make_mesh((8,), ("x",))
single_axis_specs = P("x")
```

For a rank-2 matrix, JAX treats `P("x")` as equivalent to `P("x", None)`.
To shard columns instead, use `P(None, "x")`.

The mesh passed to a solver does not have to be the one in context: each call
enters its own mesh, so a program that keeps a different mesh set globally can
call JAXMg without switching it.

You can inspect the resultant process-rank mapping selected by JAX:

```python
import numpy as np


rank_grid = np.asarray(
    [device.process_index for device in mesh.devices.flat]
).reshape(mesh.devices.shape)
if jax.process_index() == 0:
    print(rank_grid)
```

For the row-major $4\times2$ mesh used in this example, rank 0 prints

```text
[[0 1]
 [2 3]
 [4 5]
 [6 7]]
```


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

mesh = row_major_mesh  # Or select column_major_mesh.
jax.set_mesh(mesh)
```

## 4. Shard the matrix and solve input

`PartitionSpec` describes how each array dimension maps onto the process-grid
axes. A matrix can be sharded over both axes:

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

A vector solve input can be sharded over process rows and replicated across
process columns:

```python
b_vector = jnp.ones((N,), dtype=a.dtype)
b_vector = jax.device_put(b_vector, NamedSharding(mesh, P("pr")))
```

For a solve input with shape $N\times\mathrm{NRHS}$, shard its rows and leave
its columns replicated:

```python
b_matrix = jnp.ones((N, 1), dtype=a.dtype)
b_matrix = jax.device_put(b_matrix, NamedSharding(mesh, P("pr", None)))
```

The public solver accepts either representation. JAXMg adds any routing or tile
padding required for a narrow solve input and redistributes it internally for
cuSOLVERMp.

With distributed execution configured, continue to [Choose a tile size
$T_A$](choose_tile_size.md) before selecting a solver example.
