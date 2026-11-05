# LU factorization

`jaxmg.getrf` computes $PA=LU$ for a square nonsingular matrix, returning the
packed factors and a replicated vector of zero-based pivot indices.

## Common setup

This example uses four Python processes and a $2 \times 2$ process grid.
See [distributed execution](execution.md) for initialization and launch options.

```python
import jax

jax.config.update("jax_enable_x64", True)
jax.distributed.initialize()

import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from jaxmg import getrf


mesh = jax.make_mesh((2, 2), ("pr", "pc"))
jax.set_mesh(mesh)
matrix_sharding = NamedSharding(mesh, P("pr", "pc"))

T_A = 128
N = 512


@jax.jit
def make_matrix():
    lower = jnp.eye(N, dtype=jnp.float64) + jnp.tril(jnp.full((N, N), 0.5), -1)
    upper = jnp.diag(jnp.linspace(1.0, 2.0, N)) + jnp.triu(jnp.full((N, N), 0.25), 1)
    a = (lower @ upper)[::-1]
    expected_lu = jnp.tril(lower, -1) + upper
    return jax.reshard(a, matrix_sharding), jax.reshard(expected_lu, matrix_sharding)
```

## Compute the factors and pivots

The constructed matrix requires a reversal of its rows. The packed result
stores $U$ on and above the diagonal and the off-diagonal entries of $L$ below
it; the diagonal of $L$ is implicitly one.

```python
a, expected_lu = make_matrix()
lu, piv = getrf(a, T_A=T_A)

# Validate the factors and permutation against the known solution.
permutation = jax.lax.linalg.lu_pivots_to_permutation(piv, N)
correct = jnp.allclose(lu, expected_lu) & jnp.array_equal(
    permutation, jnp.arange(N - 1, -1, -1, dtype=permutation.dtype)
)
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

`getrf_jit_ctx` leaves the outer compilation and donation boundary to the
caller. The packed factor aliases the input work buffer; the pivot vector is
a separate replicated result.

```python
from functools import partial

from jaxmg import getrf_jit_ctx


@partial(jax.jit, donate_argnums=(0,))
def compiled_getrf(a):
    return getrf_jit_ctx(a, T_A=T_A)


a, _ = make_matrix()
lu, piv, status = compiled_getrf(a)
lu.block_until_ready()
```

When `a` is created inside the compiled function, XLA owns its lifetime and
`donate_argnums` is unnecessary.

## Advanced: call from inside `jax.shard_map`

`getrf_shardmap_ctx` accepts the local matrix block. Its factor remains local,
while the pivot vector is identical on every rank and uses `P()` in the output
specification.

```python
from jaxmg import getrf_shardmap_ctx


@partial(jax.jit, donate_argnums=(0,))
@partial(jax.shard_map, mesh=mesh, in_specs=P("pr", "pc"),
         out_specs=(P("pr", "pc"), P(), P(("pr", "pc"))))
def local_factorize(a_block):
    return getrf_shardmap_ctx(a_block, T_A=T_A)


a, _ = make_matrix()
lu, piv, status = local_factorize(a)
lu.block_until_ready()
```

Ready-to-run scripts for the direct and caller-jitted interfaces are provided in the
[examples directory](https://github.com/flatironinstitute/jaxmg/tree/main/examples).
See the [`getrf` API reference](../api/getrf.md) for the complete argument and
return-value documentation.
