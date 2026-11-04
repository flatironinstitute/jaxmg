# jaxmg.potrs

`potrs` is the high-level Cholesky solve interface. It validates the arrays and
mesh, applies tile-capacity padding when required, runs the internally compiled
fused backend, and returns the solution in the JAX-facing layout.

Use `potrs` for a direct solve, `potrs_jit_ctx` to include the solve within a
larger compiled calculation, or `potrs_shardmap_ctx` when working inside an
existing `jax.shard_map`. The first two interfaces accept global arrays;
`potrs_shardmap_ctx` accepts the local blocks supplied by `shard_map`. All three
use the same distributed Cholesky solver.

!!! Note "Context API migration"

    The previous `potrs_shardmap_ctx` interface for global arrays is now named
    `potrs_jit_ctx`, with the same arguments and outputs. Existing callers
    should use that name; `potrs_shardmap_ctx` now operates on local blocks
    inside `jax.shard_map`. The other routines retain their existing context
    interfaces for now.

All three interfaces accept `return_logdet=True`. Since the factorization
produces $A=LL^H$, the backend computes

$$
\log\det(A) = 2\sum_i \log |L_{ii}|
$$

directly from the distributed Cholesky factor.

::: jaxmg.potrs

::: jaxmg.potrs_jit_ctx

::: jaxmg.potrs_shardmap_ctx
