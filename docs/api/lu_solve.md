# jaxmg.lu_solve

`lu_solve` is the high-level solver for general nonsingular matrices. It uses
cuSOLVERMp `Getrf` followed by `Getrs`, including distributed pivot storage and
the same fused memory-distribution path used by `potrs`.

Use `lu_solve_shardmap_ctx` when the solve is part of a larger caller-owned
`jax.jit` computation. It returns `(a_work, x, status)` so the outer compiled
function can preserve the donated matrix alias.

```python
from functools import partial

import jax
from jax.sharding import PartitionSpec as P
from jaxmg import lu_solve_shardmap_ctx

@partial(jax.jit, donate_argnums=(0, 1))
def solve(a, b):
    return lu_solve_shardmap_ctx(
        a,
        b,
        T_A=256,
        mesh=mesh,
        matrix_specs=P("pr", "pc"),
    )

a_work, x, status = solve(a, b)
```

`a_work` contains native factorization work data. It is returned to provide an
`A`-sized alias target and is not the original matrix.

::: jaxmg.lu_solve

::: jaxmg.lu_solve_shardmap_ctx
