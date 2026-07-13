# Cholesky solve

`jaxmg.potrs` solves $Ax=B$ for a symmetric or Hermitian positive-definite
matrix. The following example uses a degenerate 2D process grid with one Python
process per GPU.

```python
import jax

jax.config.update("jax_enable_x64", True)
jax.distributed.initialize()

import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from jaxmg import potrs


num_processes = jax.process_count()
mesh = jax.make_mesh((num_processes, 1), ("pr", "pc"))
matrix_specs = P("pr", "pc")

T_A = 64
N = T_A * num_processes
dtype = jnp.float64

a = jnp.diag(jnp.arange(1, N + 1, dtype=dtype))
b = jnp.ones((N, 1), dtype=dtype)

a = jax.device_put(a, NamedSharding(mesh, matrix_specs))
b = jax.device_put(b, NamedSharding(mesh, P("pr", None)))

x = potrs(
    a,
    b,
    T_A=T_A,
    mesh=mesh,
    matrix_specs=matrix_specs,
)
x.block_until_ready()

expected = 1.0 / jnp.arange(1, N + 1, dtype=dtype)
correct = jnp.allclose(x[:, 0], expected)
correct.block_until_ready()

if jax.process_index() == 0:
    print(correct)
```

The result is `True`. A one-dimensional `b` is also accepted; in that case
`potrs` returns a one-dimensional solution.

The matrix and right-hand side are ordinary JAX arrays. The local memory
conversion, edge-padding alignment, 2D block-cyclic redistribution, Cholesky
factorization, solve, and reverse redistribution occur inside the fused native
backend.

See [`potrs_shardmap_ctx`](../api/potrs.md) when this solve is embedded inside a
larger caller-owned `jax.jit` computation.
