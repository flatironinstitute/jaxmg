# jaxmg.syevd_mp

For multi-node runs, initialize JAX before constructing devices, arrays, or
meshes. ``syevd_mp`` requires one Python process per participating GPU:

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

Shard the input matrix with ordinary JAX ``NamedSharding``. ``syevd_mp``
infers the mesh and ``PartitionSpec`` from ``A`` by default:

```python
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

sharding = NamedSharding(mesh, P("pr", "pc"))
A = jax.device_put(jnp.asarray(A_host), sharding)

w, V = jaxmg.syevd_mp(A, T_A=128, eigvecs=True)
```

``syevd_mp`` returns replicated eigenvalues. When ``eigvecs=True``, the
eigenvector matrix is reverse-redistributed to the same JAX-facing
block-sharded layout as the input matrix.

The cuSOLVERMp no-vector path is still under validation for the 0.7.2 runtime
used in current CSD3 testing. Do not rely on ``eigvecs=False`` for this backend
until that path is explicitly marked supported.

::: jaxmg.syevd_mp
