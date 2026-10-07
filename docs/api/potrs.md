# jaxmg.potrs

`potrs` is the high-level Cholesky solve interface. It validates the arrays and
mesh, applies tile-capacity padding when required, runs the internally compiled
fused backend, and returns the solution in the JAX-facing layout.

The solver comes in three layers, each wrapping the one below it:

| Function | Called on | Adds |
|---|---|---|
| `potrs_shardmap_ctx` | local blocks inside a caller's `jax.shard_map` | the per-shard solve: tile padding, the fused native call, and unpadding |
| `potrs_jit_ctx` | global sharded arrays inside a caller's `jax.jit` | `jax.shard_map`, placement of `a` and `b` in the matrix sharding, and restoration of the solution's layout |
| `potrs` | global sharded arrays | an internal `jax.jit` with buffer donation |

Use `potrs` for a direct solve. Use `potrs_jit_ctx` when the solve must be
embedded inside a larger caller-owned `jax.jit` computation, and
`potrs_shardmap_ctx` when it is one step of a caller-owned `jax.shard_map`. All
three run the same native solver pipeline.

Up to version 1.4.0, `potrs_shardmap_ctx` was the name of what is now
`potrs_jit_ctx`.

All three interfaces accept `return_logdet=True`. Since the factorization
produces $A=LL^H$, the backend computes

$$
\log\det(A) = 2\sum_i \log |L_{ii}|
$$

directly from the distributed Cholesky factor.

::: jaxmg.potrs

::: jaxmg.potrs_jit_ctx

::: jaxmg.potrs_shardmap_ctx
