# jaxmg.potrs_mp

For multi-node runs, initialize JAX before constructing devices, arrays, or
meshes:

```python
import jax

jax.distributed.initialize(
    coordinator_address=coordinator_address,
    num_processes=num_processes,
    process_id=process_id,
    local_device_ids=[local_rank],
)
mesh = jax.make_mesh((2, 4), ("pr", "pc"))
```

The mesh uses row-major rank order, matching the native cuSOLVERMp descriptor:

```text
rank = process_row * process_cols + process_col
```

Shard inputs with ordinary JAX ``NamedSharding``. ``potrs_mp`` infers the mesh
and ``PartitionSpec`` from ``A`` by default:

```python
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

sharding = NamedSharding(mesh, P("pr", "pc"))
A = jax.device_put(jnp.asarray(A_host), sharding)
B = jax.device_put(jnp.asarray(B_host), sharding)

X = jaxmg.potrs_mp(A, B, T_A=128)
```

::: jaxmg.potrs_mp
