# Singular-value decomposition

`jaxmg.gesvd` computes the singular-value decomposition of an $M \times N$ real
or complex matrix $A$ ($A = U \Sigma V^{\dagger}$). The default reduced
decomposition follows the JAX return order `(U, s, Vh)`.

## Common setup

This example uses four Python processes and a $2 \times 2$ process grid. The
matrix dimensions are chosen so A and the reduced U and Vh outputs are evenly
divisible by the process grid and local dimensions are divisible by `T_A`.

```python
import jax

jax.config.update("jax_enable_x64", True)
jax.distributed.initialize()

import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from jaxmg import gesvd


mesh = jax.make_mesh((2, 2), ("pr", "pc"))
jax.set_mesh(mesh)
matrix_specs = P("pr", "pc")
matrix_sharding = NamedSharding(mesh, matrix_specs)

T_A = 128
M = 1024
N = 512
K = min(M, N)
dtype = jnp.float64


@jax.jit
def make_matrix():
    singular_values = jnp.linspace(2.0, 1.0, K, dtype=dtype)
    a = jnp.zeros((M, N), dtype=dtype)
    a = a.at[jnp.arange(K), jnp.arange(K)].set(singular_values)
    return jax.reshard(a, matrix_sharding), singular_values
```

## Reduced decomposition

Pass the sharded matrix directly to `gesvd`:

```python
a, expected_singular_values = make_matrix()

u, singular_values, vh = gesvd(
    a,
    T_A=T_A,
    mesh=mesh,
    matrix_specs=matrix_specs,
)
vh.block_until_ready()

correct = jnp.allclose(singular_values, expected_singular_values)
correct.block_until_ready()

if jax.process_index() == 0:
    print(correct)
```

The reduced outputs have shapes `(M, K)`, `(K,)`, and `(K, N)`. Set
`full_matrices=True` to request U with shape `(M, M)` and Vh with shape
`(N, N)`.

## Select vector outputs

U and Vh can be selected independently:

```python
# Singular values only.
a, _ = make_matrix()
singular_values = gesvd(
    a,
    T_A=T_A,
    mesh=mesh,
    matrix_specs=matrix_specs,
    compute_u=False,
    compute_vh=False,
)

# U and singular values, without Vh.
a, _ = make_matrix()
u, singular_values = gesvd(
    a,
    T_A=T_A,
    mesh=mesh,
    matrix_specs=matrix_specs,
    compute_vh=False,
)
```

Disabling an output avoids allocating and restoring that distributed vector
matrix.

!!! Warning

     The public wrapper donates `a` to the compiled decomposition. Do not use
     the input array after the call.

There is no need to apply `jax.jit` or specify `donate_argnums`: `gesvd` uses
an internally cached jitted wrapper. Use the context interface below when the
decomposition must be embedded in a larger compiled calculation.

## Advanced: control the outer `jax.jit`

`gesvd_shardmap_ctx` runs the same padding, redistribution, and decomposition
as `gesvd`, but leaves the outer `jax.jit` boundary to the caller. This allows
the decomposition to become one stage of a larger compiled calculation.

The context interface returns `a_work` before the selected numerical outputs.
This is the overwritten input-matrix work buffer. Whether it must leave the
outer function depends on where the input matrix was created.

The advanced examples additionally use:

```python
from functools import partial

from jaxmg import gesvd_shardmap_ctx
```

### Case 1: `a` is an argument of the jitted function

When an existing matrix enters the outer jitted function as an argument, donate
it with `donate_argnums`. Return `a_work` so the donated input has an $A$-sized
output alias at the outer compiled boundary:

```python


@partial(jax.jit, donate_argnums=(0,))
def compiled_svd(a):
    a_work, u, singular_values, vh, status = gesvd_shardmap_ctx(
        a,
        T_A=T_A,
        mesh=mesh,
        matrix_specs=matrix_specs,
    )

    # Further JAX operations can be part of this compiled function.
    scaled_singular_values = 2.0 * singular_values
    return a_work, u, scaled_singular_values, vh, status


a, expected_singular_values = make_matrix()
a_work, u, scaled_singular_values, vh, status = compiled_svd(a)
vh.block_until_ready()

correct = jnp.allclose(scaled_singular_values, 2.0 * expected_singular_values)
correct.block_until_ready()

if jax.process_index() == 0:
    print(correct)
```

Returning `a_work` is required when `a` is donated. It is opaque overwritten
solver storage, not a numerical SVD output, and should remain in the returned
pytree until the compiled calculation has completed. Requested U and Vh
matrices occupy separate allocations because cuSOLVERMp does not permit them to
overlap A.

### Case 2: `a` is created inside the jitted function

If the outer compiled function constructs `a` itself, the matrix is an internal
temporary rather than a donated argument. XLA controls its lifetime, so
`a_work` can remain inside the compiled function:

```python
@jax.jit
def build_and_decompose(scale):
    expected_singular_values = scale * jnp.linspace(
        2.0, 1.0, K, dtype=dtype
    )
    a = jnp.zeros((M, N), dtype=dtype)
    a = a.at[jnp.arange(K), jnp.arange(K)].set(expected_singular_values)
    a = jax.reshard(a, matrix_sharding)

    _, u, singular_values, vh, status = gesvd_shardmap_ctx(
        a,
        T_A=T_A,
        mesh=mesh,
        matrix_specs=matrix_specs,
    )

    # a_work remains internal because a was created inside this function.
    return u, singular_values, vh, expected_singular_values, status


scale = jnp.asarray(2.0, dtype=dtype)
u, singular_values, vh, expected_singular_values, status = build_and_decompose(
    scale
)
vh.block_until_ready()

correct = jnp.allclose(singular_values, expected_singular_values)
correct.block_until_ready()

if jax.process_index() == 0:
    print(correct)
```

`donate_argnums` is not needed for the internally created `a`, because it is
not an argument of `build_and_decompose`. Only donate an outer argument such as
`scale` if the caller no longer needs it after the call.

See the [`gesvd` API reference](../api/gesvd.md) for the complete argument and
return-value documentation.
