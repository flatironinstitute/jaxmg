# LU solve

`jaxmg.lu_solve` solves $Ax=B$ for a general nonsingular matrix using pivoted
LU factorization. Its array, mesh, padding, and tile-size interface matches
`jaxmg.potrs`. JAXMg provides two interfaces to the same native LU pipeline.

| Interface | Use it when |
|---|---|
| `lu_solve` | The solve is called directly from Python and JAXMg should manage the compiled boundary. |
| `lu_solve_shardmap_ctx` | A caller-owned `jax.jit` receives `a` and `b`, or constructs them internally as part of a larger calculation. |

Both interfaces perform the same validation, padding, shard mapping, fused
cuSOLVERMp call, and reverse redistribution. The difference is which function
owns the outer `jax.jit` boundary and the lifetime of the donated matrix buffer.
The caller does not construct `jax.shard_map` manually;
`lu_solve_shardmap_ctx` constructs the required shard map but leaves the
enclosing `jax.jit` to the caller.

## Common setup

```python
from functools import partial

import jax

jax.config.update("jax_enable_x64", True)
jax.distributed.initialize()

import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from jaxmg import lu_solve, lu_solve_shardmap_ctx


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
    b = a @ expected_x
    a = jax.device_put(a, a_sharding)
    b = jax.device_put(b, b_sharding)
    return a, b
```

## High-level interface

Use `lu_solve` for a standalone solve. It constructs the shard map and uses an
internally cached `jax.jit`-compiled wrapper:

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

The result is `True`. A one-dimensional right-hand side is also accepted. Pass
`return_status=True` to return the native diagnostic status alongside `x`.

The public wrapper may donate `a` and `b` to the compiled solve. Do not use
those input arrays after the call. LU factorization stores a distributed pivot
vector internally; users do not need to construct or manage it.

## Caller-owned `jax.jit` interface

Use `lu_solve_shardmap_ctx` when the solve is embedded in a larger compiled
calculation. It returns `(a_work, x, status)` so the caller-owned JIT can donate
`a` into an $A$-sized output:

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
coefficient matrix, and should remain in the returned pytree until the compiled
computation has completed.

## Constructing the matrix inside `jax.jit`

When the outer compiled function constructs `a` and `b`, those arrays are
internal temporaries. XLA controls their lifetime, so `a_work` does not need to
be returned from the outer function:

```python
@jax.jit
def build_and_solve(diagonal):
    a = jnp.diag(diagonal)
    a = a + 0.01 * jnp.triu(jnp.ones((N, N), dtype=diagonal.dtype), k=1)
    a = jax.lax.with_sharding_constraint(a, a_sharding)

    expected_x = jnp.ones((N, 1), dtype=diagonal.dtype)
    b = a @ expected_x
    b = jax.lax.with_sharding_constraint(b, b_sharding)

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
