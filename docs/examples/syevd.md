# Symmetric or Hermitian eigensolve

`jaxmg.syevd` computes the eigenvalues $\lambda_i$ and, optionally, eigenvectors
$v_i$ of a symmetric real or Hermitian complex matrix $A$, satisfying
$Av_i=\lambda_i v_i$. For a normal eigensolve, use `syevd`. It provides the
high-level interface and internally handles JIT compilation, buffer donation,
input/output aliasing, padding, and distributed execution.

## Common setup

The following setup uses an even number of processes arranged as a two-axis
device mesh, with one Python process per GPU:

```python
import jax

jax.config.update("jax_enable_x64", True)
jax.distributed.initialize()

import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from jaxmg import syevd


num_processes = jax.process_count()
if num_processes % 2:
    raise ValueError("This example requires an even number of processes.")
mesh = jax.make_mesh((num_processes // 2, 2), ("pr", "pc"))
jax.set_mesh(mesh)
matrix_specs = P("pr", "pc")
a_sharding = NamedSharding(mesh, matrix_specs)

T_A = 64
N = T_A * num_processes
dtype = jnp.float64


def make_problem():
    expected_eigenvalues = jnp.arange(1, N + 1, dtype=dtype)
    a = jnp.diag(expected_eigenvalues)
    a = jax.device_put(a, a_sharding)
    return a, expected_eigenvalues
```

## Solve with `syevd`

Pass the sharded matrix directly to `syevd`:

```python
a, expected_eigenvalues = make_problem()

eigenvalues, eigenvectors = syevd(
    a,
    T_A=T_A,
    mesh=mesh,
    matrix_specs=matrix_specs,
)
eigenvectors.block_until_ready()

correct = jnp.allclose(eigenvalues, expected_eigenvalues)
correct.block_until_ready()

if jax.process_index() == 0:
    print(correct)
```

The eigenvalues are returned as a real array. The eigenvectors have the input
dtype and are returned in the same JAX-facing matrix layout as `a`.

When eigenvectors are not required, select the shorter values-only workflow:

```python
a, expected_eigenvalues = make_problem()

eigenvalues = syevd(
    a,
    T_A=T_A,
    mesh=mesh,
    matrix_specs=matrix_specs,
    return_eigenvectors=False,
)
eigenvalues.block_until_ready()
```

This mode does not allocate, reverse-redistribute, or restore a matrix-sized
eigenvector result.

!!! Warning

     The public wrapper donates `a` to the compiled eigensolve. Do not use the
     input array after the call. Use `donate=False` to preserve it. This is less
     memory efficient because the original input and working buffer must coexist.

There is no need to apply `jax.jit` or specify `donate_argnums`: `syevd` uses
an internally cached jitted wrapper and manages donation and aliasing itself.
If the eigensolve must be embedded inside a larger jitted calculation, use the
advanced interface below instead of wrapping `syevd` in another `jax.jit`.

When eigenvectors are requested, SYEVD materializes the full distributed
eigenvector matrix and uses solver-specific workspace. It therefore reaches
the GPU memory limit at a smaller matrix size than `potrs` or `lu_solve` on the
same process grid.

## Advanced: control the outer `jax.jit`

`syevd_shardmap_ctx` runs the same selected eigensolver workflow as `syevd`.
The difference is that it does not create an internal `jax.jit`, allowing the
eigensolve to become one stage of a larger function compiled by the caller.

The context interface returns
`(a_work, eigenvalues, eigenvectors, status)`. cuSOLVERMp uses separate
distributed buffers for its overwritten input matrix and eigenvector output,
so `a_work` is returned to give the outer compiled function an $A$-sized alias
target for the donated input. The eigenvectors still require a separate
matrix-sized allocation.

With `return_eigenvectors=False`, the context interface instead returns
`(a_work, eigenvalues, status)`.

The advanced examples additionally use:

```python
from functools import partial

from jaxmg import syevd_shardmap_ctx
```

### Case 1: `a` is an argument of the jitted function

When an existing matrix enters the outer jitted function as an argument, donate
it with `donate_argnums`. Return `a_work` so the donated input has an
$A$-sized output alias at the outer compiled boundary:

```python
@partial(jax.jit, donate_argnums=(0,))
def compiled_eigensolve(a):
    a_work, eigenvalues, eigenvectors, status = syevd_shardmap_ctx(
        a,
        T_A=T_A,
        mesh=mesh,
        matrix_specs=matrix_specs,
    )

    # Further JAX operations can be part of this compiled function.
    shifted_eigenvalues = eigenvalues + 1.0
    return a_work, shifted_eigenvalues, eigenvectors, status


a, expected_eigenvalues = make_problem()
a_work, shifted_eigenvalues, eigenvectors, status = compiled_eigensolve(a)
eigenvectors.block_until_ready()

correct = jnp.allclose(shifted_eigenvalues, expected_eigenvalues + 1.0)
correct.block_until_ready()

if jax.process_index() == 0:
    print(correct)
```

Returning `a_work` from the outer jitted function is required when `a` is
donated. It is an opaque native work buffer and should remain in the returned
pytree until the compiled computation has completed. It is not the eigenvector
matrix.

### Case 2: `a` is created inside the jitted function

If the outer compiled function constructs `a` itself, the matrix is an internal
temporary. XLA controls its lifetime, so `a_work` does not need to be returned
from the outer function:

```python
@jax.jit
def build_and_eigensolve(diagonal):
    a = jnp.diag(diagonal)
    a = jax.reshard(a, a_sharding)

    _, eigenvalues, eigenvectors, status = syevd_shardmap_ctx(
        a,
        T_A=T_A,
        mesh=mesh,
        matrix_specs=matrix_specs,
    )

    # a_work remains internal because a was created inside this function.
    return eigenvalues + 1.0, eigenvectors, status


diagonal = jnp.arange(1, N + 1, dtype=dtype)
diagonal = jax.device_put(diagonal, NamedSharding(mesh, P("pr")))
shifted_eigenvalues, eigenvectors, status = build_and_eigensolve(diagonal)
eigenvectors.block_until_ready()
```

`donate_argnums` is not needed for the internally created `a`. It only applies
to arguments of the outer jitted function. Donate an outer argument such as
`diagonal` only when the caller no longer needs it after the call.

See the [`syevd` API reference](../api/syevd.md) for the complete argument and
return-value documentation.
