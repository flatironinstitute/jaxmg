# LU solve

`jaxmg.lu_solve` solves $Ax=B$ for a general nonsingular matrix using pivoted LU
factorization. Its array, mesh, padding, and tile-size interface matches
`jaxmg.potrs`.

```python
import jax

jax.config.update("jax_enable_x64", True)
jax.distributed.initialize()

import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from jaxmg import lu_solve


num_processes = jax.process_count()
mesh = jax.make_mesh((num_processes, 1), ("pr", "pc"))
matrix_specs = P("pr", "pc")

T_A = 64
N = T_A * num_processes
dtype = jnp.float64

# A diagonally dominant, nonsymmetric matrix.
diagonal = jnp.arange(N, dtype=dtype) + 2 * N
a = jnp.diag(diagonal)
a = a + 0.01 * jnp.triu(jnp.ones((N, N), dtype=dtype), k=1)
expected_x = jnp.ones((N, 1), dtype=dtype)
b = a @ expected_x

a = jax.device_put(a, NamedSharding(mesh, matrix_specs))
b = jax.device_put(b, NamedSharding(mesh, P("pr", None)))

x = lu_solve(
    a,
    b,
    T_A=T_A,
    mesh=mesh,
    matrix_specs=matrix_specs,
)
x.block_until_ready()

correct = jnp.allclose(x, expected_x)
correct.block_until_ready()

if jax.process_index() == 0:
    print(correct)
```

The result is `True`. The check uses a known solution because the public solver
may donate `a` and `b`; callers should not use those input buffers after the
solve. LU factorization stores a distributed pivot vector internally, so users
do not need to construct or manage it.

Use `lu_solve` for general matrices. For symmetric or Hermitian
positive-definite matrices, `potrs` is normally faster and uses less workspace.

See [`lu_solve_shardmap_ctx`](../api/lu_solve.md) when an outer compiled
function controls donation and buffer lifetime.
