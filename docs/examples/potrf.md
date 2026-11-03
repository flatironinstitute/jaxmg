# Cholesky factorization

`jaxmg.potrf` computes $A=LL^{\dagger}$ for a symmetric or Hermitian
positive-definite matrix, returning the lower-triangular factor $L$ without
performing a solve.

## Common setup

This example uses four Python processes and a $2 \times 2$ process grid.
See [distributed execution](execution.md) for initialization and launch options.

```python
import jax

jax.config.update("jax_enable_x64", True)
jax.distributed.initialize()

import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from jaxmg import potrf


mesh = jax.make_mesh((2, 2), ("pr", "pc"))
jax.set_mesh(mesh)
matrix_sharding = NamedSharding(mesh, P("pr", "pc"))

T_A = 128
N = 512


@jax.jit
def make_matrix():
    expected = jnp.diag(jnp.linspace(1.0, 2.0, N, dtype=jnp.float64))
    expected += jnp.diag(jnp.full(N - 1, 0.5), k=-1)
    a = expected @ expected.T
    return jax.reshard(a, matrix_sharding), jax.reshard(expected, matrix_sharding)
```

## Compute the factor

```python
a, expected = make_matrix()
factor = potrf(a, T_A=T_A)

# Validate the factor against the known solution.
correct = jnp.allclose(factor, expected)
correct.block_until_ready()
if jax.process_index() == 0:
    print(correct)
```

!!! Warning

     The public wrapper donates `a` to the compiled factorization. Do not use
     the input array after the call. Use `donate=False` to preserve it. This is
     less memory efficient because the original input and working buffer must
     coexist.

## Advanced: control the outer `jax.jit`

`potrf_shardmap_ctx` leaves the outer compilation and donation boundary to the
caller. Its first result is the factor, which aliases the input work buffer.

```python
from functools import partial

from jaxmg import potrf_shardmap_ctx


@partial(jax.jit, donate_argnums=(0,))
def compiled_potrf(a):
    return potrf_shardmap_ctx(a, T_A=T_A)


a, _ = make_matrix()
factor, status = compiled_potrf(a)
factor.block_until_ready()
```

When `a` is created inside the compiled function, XLA owns its lifetime and
`donate_argnums` is unnecessary.

Ready-to-run scripts for both interfaces are provided in the
[examples directory](https://github.com/flatironinstitute/jaxmg/tree/main/examples).
See the [`potrf` API reference](../api/potrf.md) for the complete argument and
return-value documentation.
