# jaxmg.potrs_mp

For multi-node runs, initialize JAX before constructing devices, arrays, or
meshes:

```python
import jaxmg

jaxmg.initialize_node_process()
mesh = jaxmg.make_cusolvermp_mesh(process_rows=2, process_cols=4)
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
