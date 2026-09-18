# Least-squares solve

`jaxmg.least_squares` solves
$\min_X\lVert AX-B\rVert_2$ for an $M\times N$ matrix with $M\geq N$ using
distributed QR factorization.

## Common setup

This example uses four Python processes and a $2\times2$ process grid. The
dimensions are chosen so the local matrix shards and the returned solution are
evenly divisible by the process grid and tile size.

```python
import jax

jax.config.update("jax_enable_x64", True)
jax.distributed.initialize()

import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from jaxmg import least_squares


mesh = jax.make_mesh((2, 2), ("pr", "pc"))
jax.set_mesh(mesh)
matrix_specs = P("pr", "pc")
matrix_sharding = NamedSharding(mesh, matrix_specs)
solve_sharding = NamedSharding(mesh, P("pr", None))

T_A = 128
M, N = 2048, 1024
dtype = jnp.float64
expected_x = jnp.ones((N,), dtype=dtype)


@jax.jit
def make_problem():
    diagonal = jnp.arange(1, N + 1, dtype=dtype)
    a = jnp.zeros((M, N), dtype=dtype)
    a = a.at[jnp.arange(N), jnp.arange(N)].set(diagonal)
    a = jax.reshard(a, matrix_sharding)
    b = jax.reshard(a @ expected_x, solve_sharding)
    return a, b
```

## Solve with `least_squares`

Pass the sharded matrix and solve input directly to `least_squares`:

```python
a, b = make_problem()

x = least_squares(
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

A matrix solve input of shape `(M, K)` returns a solution of shape `(N, K)`.

!!! Warning

     The public wrapper donates `a` and `b` to the compiled solve. Do not use
     those arrays after the call. Use `donate=False` to preserve them at the
     cost of retaining the original inputs alongside the native work buffers.

There is no need to apply `jax.jit` or specify `donate_argnums`:
`least_squares` uses an internally cached jitted wrapper. Use the context
interface below when the solve must be embedded in a larger compiled
calculation.

## Advanced: control the outer `jax.jit`

`least_squares_shardmap_ctx` runs the same distributed pipeline but leaves the
outer `jax.jit` boundary to the caller. It returns
`(a_work, b_work, x, status)`. The two work outputs provide shape-compatible
alias targets because cuSOLVERMp overwrites both inputs.

The advanced examples additionally use:

```python
from functools import partial

from jaxmg import least_squares_shardmap_ctx
```

### Case 1: `a` and `b` are arguments of the jitted function

Donate both arguments and retain both work outputs in the returned pytree:

```python
@partial(jax.jit, donate_argnums=(0, 1))
def compiled_solve(a, b):
    a_work, b_work, x, status = least_squares_shardmap_ctx(
        a,
        b,
        T_A=T_A,
        mesh=mesh,
        matrix_specs=matrix_specs,
    )
    return a_work, b_work, 2.0 * x, status


a, b = make_problem()
a_work, b_work, scaled_x, status = compiled_solve(a, b)
scaled_x.block_until_ready()
```

`a_work` and `b_work` are opaque overwritten solver storage. They should not
be interpreted as the original inputs.

### Case 2: `a` and `b` are created inside the jitted function

When the outer function constructs both inputs, XLA controls their lifetime
and the work outputs can remain internal:

```python
@jax.jit
def build_and_solve():
    diagonal = jnp.arange(1, N + 1, dtype=dtype)
    a = jnp.zeros((M, N), dtype=dtype)
    a = a.at[jnp.arange(N), jnp.arange(N)].set(diagonal)
    a = jax.reshard(a, matrix_sharding)
    b = jax.reshard(a @ expected_x, solve_sharding)

    _, _, x, status = least_squares_shardmap_ctx(
        a,
        b,
        T_A=T_A,
        mesh=mesh,
        matrix_specs=matrix_specs,
    )
    return x, status


x, status = build_and_solve()
x.block_until_ready()
```

See the [`least_squares` API reference](../api/least_squares.md) for the
complete argument and return-value documentation.
