# LU solve

`jaxmg.lu_solve` solves $Ax=B$ for a general nonsingular matrix using pivoted
LU factorization. Its array, mesh, padding, and tile-size interface matches
`jaxmg.potrs`. For a normal solve, use `lu_solve`. It provides the high-level
interface and internally handles JIT compilation, buffer donation,
input/output aliasing, padding, and distributed execution.

## Common setup

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
a_sharding = NamedSharding(mesh, matrix_specs)
b_sharding = NamedSharding(mesh, P("pr", None))

T_A = 64
N = T_A * num_processes
dtype = jnp.float64
expected_x = jnp.ones((N, 1), dtype=dtype)


def make_problem():
    # A diagonally dominant, nonsymmetric matrix.
    diagonal = jnp.arange(N, dtype=dtype) + 2 * N
    a = jnp.diag(diagonal)
    a = a + 0.01 * jnp.triu(jnp.ones((N, N), dtype=dtype), k=1)
    a = jax.device_put(a, a_sharding)

    # Form B from the globally sharded matrix. This is important in MPMD:
    # every process must not independently provide a different local B to
    # device_put as though it were a complete global array.
    x = jax.device_put(expected_x, NamedSharding(mesh, P(None, None)))
    b = jax.reshard(a @ x, b_sharding)
    return a, b
```

## Solve with `lu_solve`

Pass the sharded input matrix and right-hand side directly to `lu_solve`:

```python
a, b = make_problem()

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

A one-dimensional `b` is also accepted; in that case `lu_solve` returns a
one-dimensional solution.

!!! Warning

     The public wrapper will donate `a` and `b` to the compiled solve. Do not use
     those input arrays after the call.

LU factorization stores a distributed pivot vector internally; users do not
need to construct or manage it.

There is no need to apply `jax.jit` or specify `donate_argnums`: `lu_solve`
uses an internally cached jitted wrapper and manages donation and aliasing
itself. If the solve must be embedded inside a larger jitted calculation, use
the advanced interface below instead of wrapping `lu_solve` in another
`jax.jit`.

## Advanced: control the outer `jax.jit`

`lu_solve_shardmap_ctx` runs the same padding, redistribution, LU
factorization, solve, and reverse redistribution as `lu_solve`. The difference
is that it does not create an internal `jax.jit`. This allows the solve to
become one stage of a larger function compiled by the caller.

The context interface returns `(a_work, x, status)` rather than only `x`.
`a_work` contains the native LU work data and aliases the input matrix. Whether
it must be returned from the outer function depends on where the input matrix
was created.

The advanced examples additionally use:

```python
from functools import partial

from jaxmg import lu_solve_shardmap_ctx
```

### Case 1: `a` and `b` are arguments of the jitted function

When existing arrays enter the outer jitted function as arguments, donate them
with `donate_argnums`. Return `a_work` so the donated input matrix has an
$A$-sized output alias at the outer compiled boundary:

```python
@partial(jax.jit, donate_argnums=(0, 1))
def compiled_solve(a, b):
    a_work, x, status = lu_solve_shardmap_ctx(
        a,
        b,
        T_A=T_A,
        mesh=mesh,
        matrix_specs=matrix_specs,
    )

    # Further JAX operations on x can be part of this compiled function.
    scaled_x = 2.0 * x
    return a_work, scaled_x, status


a, b = make_problem()
a_work, scaled_x, status = compiled_solve(a, b)
scaled_x.block_until_ready()

correct = jnp.allclose(scaled_x, 2.0 * expected_x)
correct.block_until_ready()

if jax.process_index() == 0:
    print(correct)
```

Returning `a_work` from the outer jitted function is required when `a` is
donated. It contains native factorization work data, is not the original
input matrix, and should remain in the returned pytree until the compiled
computation has completed.

### Case 2: `a` and `b` are created inside the jitted function

When the outer compiled function constructs `a` and `b`, those arrays are
internal temporaries. XLA controls their lifetime, so `a_work` does not need to
be returned from the outer function:

```python
@jax.jit
def build_and_solve(diagonal):
    a = jnp.diag(diagonal)
    a = a + 0.01 * jnp.triu(jnp.ones((N, N), dtype=diagonal.dtype), k=1)
    a = jax.reshard(a, a_sharding)

    expected_x = jnp.ones((N, 1), dtype=diagonal.dtype)
    b = a @ expected_x
    b = jax.reshard(b, b_sharding)

    _, x, status = lu_solve_shardmap_ctx(
        a,
        b,
        T_A=T_A,
        mesh=mesh,
        matrix_specs=matrix_specs,
    )

    # a_work remains internal because a was created inside this function.
    return 2.0 * x, status


diagonal = jnp.arange(N, dtype=dtype) + 2 * N
diagonal = jax.device_put(diagonal, NamedSharding(mesh, P("pr")))
scaled_x, status = build_and_solve(diagonal)
scaled_x.block_until_ready()
```

`donate_argnums` is not needed for the internally created `a` and `b`. It only
applies to arguments of the outer jitted function. Donate an outer argument such
as `diagonal` only when the caller no longer needs it after the call.

Use `lu_solve` for general matrices. For symmetric or Hermitian
positive-definite matrices, `potrs` is normally faster and uses less workspace.

See the [`lu_solve` API reference](../api/lu_solve.md) for the complete argument
and return-value documentation.
