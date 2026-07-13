# jaxmg.potrs

`potrs` is the high-level Cholesky solve interface. It validates the arrays and
mesh, applies tile-capacity padding when required, runs the internally compiled
fused backend, and returns the solution in the JAX-facing layout.

Use `potrs_shardmap_ctx` when the solve is part of a larger caller-owned
`jax.jit` computation. It exposes the donated matrix work buffer as
`(a_work, x, status)`, allowing the outer compiled function to keep an
`A`-sized output alive for input/output aliasing.

```python
from functools import partial

import jax
from jax.sharding import PartitionSpec as P
from jaxmg import potrs_shardmap_ctx

@partial(jax.jit, donate_argnums=(0, 1))
def solve(a, b):
    return potrs_shardmap_ctx(
        a,
        b,
        T_A=256,
        mesh=mesh,
        matrix_specs=P("pr", "pc"),
    )

a_work, x, status = solve(a, b)
```

`a_work` is an opaque native work result and should not be interpreted as the
original coefficient matrix. Keep it in the returned pytree when the outer JIT
donates `a`.

::: jaxmg.potrs

::: jaxmg.potrs_shardmap_ctx
