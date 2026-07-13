# Cholesky solve

`jaxmg.potrs` solves $Ax=B$ for a symmetric or Hermitian positive-definite
matrix. JAXMg provides two interfaces to the same native Cholesky pipeline.

| Interface | Use it when |
|---|---|
| `potrs` | The solve is called directly from Python and JAXMg should manage the compiled boundary. |
| `potrs_shardmap_ctx` | The solve is part of a larger caller-owned `jax.jit` computation that controls donation and subsequent operations. |

Both interfaces perform the same validation, padding, shard mapping, fused
cuSOLVERMp call, and reverse redistribution. The difference is which function
owns the outer `jax.jit` boundary and the lifetime of the donated matrix buffer.
The caller does not construct `jax.shard_map` manually;
`potrs_shardmap_ctx` constructs the required shard map but leaves the enclosing
`jax.jit` to the caller.

## Common setup

The following setup uses a degenerate two-dimensional process grid with one
Python process per GPU:

```python
from functools import partial

import jax

jax.config.update("jax_enable_x64", True)
jax.distributed.initialize()

import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from jaxmg import potrs, potrs_shardmap_ctx


num_processes = jax.process_count()
mesh = jax.make_mesh((num_processes, 1), ("pr", "pc"))
matrix_specs = P("pr", "pc")

T_A = 64
N = T_A * num_processes
dtype = jnp.float64


def make_problem():
    a = jnp.diag(jnp.arange(1, N + 1, dtype=dtype))
    b = jnp.ones((N, 1), dtype=dtype)
    a = jax.device_put(a, NamedSharding(mesh, matrix_specs))
    b = jax.device_put(b, NamedSharding(mesh, P("pr", None)))
    return a, b


expected = 1.0 / jnp.arange(1, N + 1, dtype=dtype)
```

## High-level interface

Use `potrs` for a standalone solve. It constructs the shard map and uses an
internally cached `jax.jit`-compiled wrapper:

```python
a, b = make_problem()

x = potrs(
    a,
    b,
    T_A=T_A,
    mesh=mesh,
    matrix_specs=matrix_specs,
)
x.block_until_ready()

correct = jnp.allclose(x[:, 0], expected)
correct.block_until_ready()

if jax.process_index() == 0:
    print(correct)
```

The result is `True`. A one-dimensional `b` is also accepted; in that case
`potrs` returns a one-dimensional solution. Pass `return_status=True` to return
the native diagnostic status alongside `x`.

The public wrapper may donate `a` and `b` to the compiled solve. Do not use
those input arrays after the call.

## Caller-owned `jax.jit` interface

Use `potrs_shardmap_ctx` when the solve is one stage of a larger compiled
calculation. Unlike `potrs`, this interface does not create an internal
`jax.jit`. It returns `(a_work, x, status)` so the outer function has an
$A$-sized output that can alias the donated matrix input:

```python
@partial(jax.jit, donate_argnums=(0, 1))
def compiled_solve(a, b):
    a_work, x, status = potrs_shardmap_ctx(
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

correct = jnp.allclose(scaled_x[:, 0], 2.0 * expected)
correct.block_until_ready()

if jax.process_index() == 0:
    print(correct)
```

Returning `a_work` from the outer jitted function is required when `a` is
donated. It is an opaque native work buffer and should not be interpreted as
the original coefficient matrix. The caller should keep it in the returned
pytree until the compiled computation has completed.

In both interfaces, local memory conversion, edge-padding alignment, 2D
block-cyclic redistribution, Cholesky factorization, solve, and reverse
redistribution occur inside the same fused native backend.

See the [`potrs` API reference](../api/potrs.md) for the complete argument and
return-value documentation.
