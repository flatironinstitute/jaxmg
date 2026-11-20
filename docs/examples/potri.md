# Cholesky inversion

`jaxmg.potri` computes the full inverse of a symmetric or Hermitian
positive-definite matrix using a distributed Cholesky factorization.
Pass the original matrix $A$, not precomputed factors. Use
[`potrs`](potrs.md) when only a linear solve $Ax=B$ is required.

## Common setup

This example uses four Python processes on a $2 \times 2$ process grid.
See [distributed execution](execution.md) for launch options.

```python
import jax

jax.config.update("jax_enable_x64", True)
jax.distributed.initialize()

import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from jaxmg import potri

mesh = jax.make_mesh((2, 2), ("pr", "pc"))
jax.set_mesh(mesh)
matrix_sharding = NamedSharding(mesh, P("pr", "pc"))
N, T_A = 512, 128


@jax.jit
def make_matrix():
    diagonal = jnp.linspace(1.0, 2.0, N)
    return jax.reshard(jnp.diag(diagonal), matrix_sharding)
```

## Invert the matrix

```python
a = make_matrix()
inverse = potri(a, T_A=T_A)
inverse.block_until_ready()

# Validate the result against the known inverse.
correct = jnp.allclose(inverse, jnp.diag(1 / jnp.linspace(1.0, 2.0, N)))
correct.block_until_ready()
if jax.process_index() == 0:
    print(correct)
```

!!! Warning

    The public routine donates A by default. Do not reuse the input after
    the call. Use `donate=False` to preserve it, at the cost of keeping the
    original matrix and working buffer in memory simultaneously.

## Advanced: control the outer `jax.jit`

`potri_jit_ctx` returns `(inverse, status)`. When A is supplied by the caller,
the outer JIT controls donation:

```python
from functools import partial
from jaxmg import potri_jit_ctx


@partial(jax.jit, donate_argnums=(0,))
def invert(a):
    return potri_jit_ctx(a, T_A=T_A)


inverse, status = invert(make_matrix())
inverse.block_until_ready()
```

When the matrix is constructed inside the compiled function, no donation
argument is needed:

```python
@jax.jit
def build_and_invert():
    return potri_jit_ctx(make_matrix(), T_A=T_A)


inverse, status = build_and_invert()
inverse.block_until_ready()
```

## Advanced: call from inside `jax.shard_map`

`potri_shardmap_ctx` accepts the local matrix block. The supplied
specification must match the enclosing shard map.

```python
from jaxmg import potri_shardmap_ctx


@partial(jax.jit, donate_argnums=(0,))
@partial(jax.shard_map, mesh=mesh, in_specs=P("pr", "pc"),
         out_specs=(P("pr", "pc"), P(("pr", "pc"))))
def local_inverse(a_block):
    return potri_shardmap_ctx(a_block, T_A=T_A, matrix_specs=P("pr", "pc"))


inverse, status = local_inverse(make_matrix())
inverse.block_until_ready()
```

Ready-to-run direct and caller-jitted scripts are provided in the
[examples directory](https://github.com/flatironinstitute/jaxmg/tree/main/examples).
See the [`potri` API reference](../api/potri.md) for the complete arguments
and return values.
