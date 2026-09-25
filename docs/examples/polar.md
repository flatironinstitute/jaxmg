# Polar decomposition

`jaxmg.polar` computes $A=U_pH$ for a tall or square matrix. By default it
returns both factors; set `compute_h=False` when only $U_p$ is required.

## Common setup

This example uses four Python processes and a $2 \times 2$ process grid.

```python
import jax

jax.config.update("jax_enable_x64", True)
jax.distributed.initialize()

import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from jaxmg import polar


mesh = jax.make_mesh((2, 2), ("pr", "pc"))
jax.set_mesh(mesh)
matrix_specs = P("pr", "pc")
matrix_sharding = NamedSharding(mesh, matrix_specs)

T_A = 128
M = 1024
N = 512
dtype = jnp.float64


@jax.jit
def make_matrix():
    diagonal = jnp.linspace(1.0, 2.0, N, dtype=dtype)
    a = jnp.zeros((M, N), dtype=dtype)
    a = a.at[jnp.arange(N), jnp.arange(N)].set(diagonal)
    return jax.reshard(a, matrix_sharding), diagonal
```

## Compute both factors

```python
a, expected_h_diagonal = make_matrix()

up, h = polar(
    a,
    T_A=T_A,
    mesh=mesh,
    matrix_specs=matrix_specs,
)
h.block_until_ready()

correct = jnp.allclose(jnp.diag(h), expected_h_diagonal)
correct.block_until_ready()

if jax.process_index() == 0:
    print(correct)
```

Set `compute_h=False` to return only `up` and avoid allocating and restoring the
$N \times N$ factor.

!!! Warning

     The public wrapper donates `a` to the compiled decomposition. Do not use
     the input array after the call. Use `donate=False` to preserve it. This is
     less memory efficient because the original input and working buffer must
     coexist.

## Advanced: control the outer `jax.jit`

`polar_shardmap_ctx` leaves the outer compilation and donation boundary to the
caller. Unlike the other decomposition context interfaces, its first result is
the numerical factor `up`: cuSOLVERMp overwrites the input matrix with this
factor, so no separate opaque work result is needed.

```python
from functools import partial

from jaxmg import polar_shardmap_ctx


@partial(jax.jit, donate_argnums=(0,))
def compiled_polar(a):
    up, h, status = polar_shardmap_ctx(
        a,
        T_A=T_A,
        mesh=mesh,
        matrix_specs=matrix_specs,
    )
    return up, h, status


a, _ = make_matrix()
up, h, status = compiled_polar(a)
h.block_until_ready()
```

When `a` is instead created inside the compiled function, XLA owns its lifetime
and `donate_argnums` is unnecessary.

See the [`polar` API reference](../api/polar.md) for the complete argument and
return-value documentation.
