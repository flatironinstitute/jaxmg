# Reduced QR decomposition

`jaxmg.qr` computes $A=QR$ for a tall or square matrix, returning the reduced
orthonormal factor $Q$ and upper-triangular factor $R$.

## Common setup

This example uses four Python processes and a $2 \times 2$ process grid.

```python
import jax

jax.config.update("jax_enable_x64", True)
jax.distributed.initialize()

import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from jaxmg import qr


mesh = jax.make_mesh((2, 2), ("pr", "pc"))
jax.set_mesh(mesh)
matrix_specs = P("pr", "pc")
matrix_sharding = NamedSharding(mesh, matrix_specs)

T_A = 128
M = 1024
N = 512


@jax.jit
def make_matrix():
    a = jnp.arange(M * N, dtype=jnp.float64).reshape(M, N) / (M * N)
    a = a.at[:N, :].add(jnp.eye(N, dtype=a.dtype))
    return jax.reshard(a, matrix_sharding)
```

## Compute the reduced factors

```python
a = make_matrix()
q, r = qr(a, T_A=T_A)
r.block_until_ready()

correct = jnp.allclose(q.conj().T @ q, jnp.eye(N))
correct.block_until_ready()

if jax.process_index() == 0:
    print(correct)
```

!!! Warning

     The public wrapper donates `a` to the compiled decomposition. Do not use
     the input array after the call. Use `donate=False` to preserve it. This is
     less memory efficient because the original input and working buffer must
     coexist.

## Advanced: control the outer `jax.jit`

`qr_shardmap_ctx` leaves the outer compilation and donation boundary to the
caller. Its first result is $Q$, which aliases the donated input matrix.

```python
from functools import partial

from jaxmg import qr_shardmap_ctx


@partial(jax.jit, donate_argnums=(0,))
def compiled_qr(a):
    return qr_shardmap_ctx(
        a,
        T_A=T_A,
    )


a = make_matrix()
q, r, status = compiled_qr(a)
r.block_until_ready()
```

When `a` is created inside the compiled function, XLA owns its lifetime and
`donate_argnums` is unnecessary.

See the [`qr` API reference](../api/qr.md) for the complete argument and
return-value documentation.
