# jaxmg.potrs

`potrs` is the high-level Cholesky solve interface. It validates the arrays and
mesh, applies tile-capacity padding when required, runs the internally compiled
fused backend, and returns the solution in the JAX-facing layout.

Use `potrs` for a direct solve. Use `potrs_shardmap_ctx` only when the solve
must be embedded inside a larger caller-owned `jax.jit` computation. Both
interfaces run the same native solver pipeline; `_shardmap_ctx` means that the
caller owns the surrounding compilation context, not that it selects a
different solver or CUDA context.

The context interface returns `(a_work, x, status)`. The additional `a_work`
result allows an outer compiled function to preserve input/output aliasing when
the input matrix is donated.

Both interfaces accept `return_logdet=True`. Since the factorization produces
$A=LL^H$, the backend computes

$$
\log\det(A) = 2\sum_i \log |L_{ii}|
$$

directly from the distributed Cholesky factor. The high-level interface then
returns `(x, logdet)`, or `(x, logdet, status)` when `return_status=True`. The
context interface returns `(a_work, x, logdet, status)`. `logdet` is a
replicated real scalar.

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
original input matrix. Keep it in the returned pytree when the outer JIT
donates `a`.

::: jaxmg.potrs

::: jaxmg.potrs_shardmap_ctx
