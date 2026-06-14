# jaxmg.syevd_mp

For multi-node runs, initialize JAX before constructing devices, arrays, or
meshes:

```python
import jaxmg

jaxmg.initialize_node_process()
mesh = jaxmg.make_cusolvermp_mesh(process_rows=2, process_cols=4)
```

Shard the input matrix with ordinary JAX ``NamedSharding``. ``syevd_mp``
infers the mesh and ``PartitionSpec`` from ``A`` by default:

```python
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

sharding = NamedSharding(mesh, P("pr", "pc"))
A = jax.device_put(jnp.asarray(A_host), sharding)

w, V = jaxmg.syevd_mp(A, T_A=128, eigvecs=True)
w_only = jaxmg.syevd_mp(A, T_A=128, eigvecs=False)
```

``syevd_mp`` returns replicated eigenvalues. When ``eigvecs=True``, the
eigenvector matrix is reverse-redistributed to the same JAX-facing
block-sharded layout as the input matrix.

::: jaxmg.syevd_mp
