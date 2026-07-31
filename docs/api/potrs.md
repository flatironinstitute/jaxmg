# jaxmg.potrs

`potrs` is the high-level Cholesky solve interface. It validates the arrays and
mesh, applies tile-capacity padding when required, runs the internally compiled
fused backend, and returns the solution in the JAX-facing layout.

Use `potrs` for a direct solve. Use `potrs_shardmap_ctx` only when the solve
must be embedded inside a larger caller-owned `jax.jit` computation. Both
interfaces run the same native solver pipeline; `_shardmap_ctx` means that the
caller owns the surrounding compilation context, not that it selects a
different solver or CUDA context.

Both interfaces accept `return_logdet=True`. Since the factorization produces
$A=LL^H$, the backend computes

$$
\log\det(A) = 2\sum_i \log |L_{ii}|
$$

directly from the distributed Cholesky factor.

::: jaxmg.potrs

::: jaxmg.potrs_shardmap_ctx
