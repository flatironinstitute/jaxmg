# Cholesky solve

`jaxmg.potrs` solves $Ax=B$ for a symmetric or Hermitian positive-definite
matrix. For a normal solve, use `potrs`. It provides the high-level interface
and internally handles JIT compilation, buffer donation, input/output aliasing,
padding, and distributed execution.

The same Cholesky factorization can also return the log determinant of the
input matrix:

$$
\log\det(A) = 2\sum_i \log |L_{ii}|,
\qquad A=LL^H.
$$

This quantity appears alongside the solve in Gaussian log likelihoods. For
example, Gaussian process regression requires both
$(K+\sigma^2I)^{-1}y$ and $\log\det(K+\sigma^2I)$, while analytical
marginalisation introduces equivalent quadratic and log-determinant terms.
JAXMg computes both from the same distributed factorization, avoiding a second
factorization or a separate determinant calculation.

## Common setup

The following setup uses a one-axis device mesh with one Python process per GPU:

```python
import jax

jax.config.update("jax_enable_x64", True)
jax.distributed.initialize()

import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from jaxmg import potrs


num_processes = jax.process_count()
mesh = jax.make_mesh((num_processes,), ("pr",))
jax.set_mesh(mesh)
matrix_specs = P("pr")
a_sharding = NamedSharding(mesh, matrix_specs)
b_sharding = NamedSharding(mesh, P("pr", None))

T_A = 64
N = T_A * num_processes
dtype = jnp.float64


def make_problem():
    a = jnp.diag(jnp.arange(1, N + 1, dtype=dtype))
    b = jnp.ones((N, 1), dtype=dtype)
    a = jax.device_put(a, a_sharding)
    b = jax.device_put(b, b_sharding)
    return a, b


expected = 1.0 / jnp.arange(1, N + 1, dtype=dtype)
```

## Solve with `potrs`

Pass the sharded matrix and solve input directly to `potrs`:

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

Set `return_logdet=True` to return the solution and log determinant together:

```python
a, b = make_problem()

x, logdet = potrs(
    a,
    b,
    T_A=T_A,
    mesh=mesh,
    matrix_specs=matrix_specs,
    return_logdet=True,
)

if jax.process_index() == 0:
    print(logdet)
```

A one-dimensional `b` is also accepted; in that case `potrs` returns a
one-dimensional solution.

!!! Warning

     The public wrapper will donate `a` and `b` to the compiled solve. Do not use
     those input arrays after the call.

There is no need to apply `jax.jit` or specify `donate_argnums`: `potrs` uses
an internally cached jitted wrapper and manages donation and aliasing itself.
If the solve must be embedded inside a larger jitted calculation, use the
advanced interface below instead of wrapping `potrs` in another `jax.jit`.

## Advanced: control the outer `jax.jit`

`potrs_shardmap_ctx` runs the same padding, redistribution, Cholesky
factorization, solve, and reverse redistribution as `potrs`. The difference is
that it does not create an internal `jax.jit`. This allows the solve to become
one stage of a larger function compiled by the caller.

The context interface returns `(a_work, x, status)` rather than only `x`.
`a_work` is the native work buffer that aliases the input matrix. Whether it
must be returned from the outer function depends on where the input matrix was
created.

The advanced examples additionally use:

```python
from functools import partial

from jaxmg import potrs_shardmap_ctx
```

### Case 1: `a` and `b` are arguments of the jitted function

When existing arrays enter the outer jitted function as arguments, donate them
with `donate_argnums`. Return `a_work` so the donated input matrix has an
$A$-sized output alias at the outer compiled boundary:

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
the original input matrix. The caller should keep it in the returned
pytree until the compiled computation has completed.

### Case 2: `a` and `b` are created inside the jitted function

If the outer compiled function constructs `a` and `b` itself, they are internal
temporaries rather than donated arguments. XLA controls their lifetime and can
reuse their storage after the solve. In this case, `a_work` can remain inside
the compiled function:

```python
@jax.jit
def build_and_solve(diagonal):
    a = jnp.diag(diagonal)
    b = jnp.ones((N, 1), dtype=diagonal.dtype)

    a = jax.reshard(a, a_sharding)
    b = jax.reshard(b, b_sharding)

    _, x, status = potrs_shardmap_ctx(
        a,
        b,
        T_A=T_A,
        mesh=mesh,
        matrix_specs=matrix_specs,
    )

    # a_work does not need to leave the function because a was created here.
    return 2.0 * x, status


diagonal = jnp.arange(1, N + 1, dtype=dtype)
diagonal = jax.device_put(diagonal, NamedSharding(mesh, P("pr")))
scaled_x, status = build_and_solve(diagonal)
scaled_x.block_until_ready()
```

`donate_argnums` is not needed for the internally created `a` and `b`: they are
not arguments of `build_and_solve`. Only donate an outer argument such as
`diagonal` if the caller no longer needs that input after the call.


See the [`potrs` API reference](../api/potrs.md) for the complete argument and
return-value documentation.
