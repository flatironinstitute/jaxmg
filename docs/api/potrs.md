# jaxmg.potrs

`potrs` is the high-level solver for symmetric or Hermitian positive-definite
matrices. It uses cuSOLVERMp `Potrf` followed by `Potrs`, with memory-layout
conversion, redistribution, and solver execution in one fused native call.

Use `potrs_jit_ctx` when the solve is part of a larger caller-owned
`jax.jit` computation. Inside an existing `jax.shard_map`, use
`potrs_shardmap_ctx` with local matrix and RHS blocks instead. Both use the
same distributed solver.

!!! Note "Context API migration"

    The previous global-array `potrs_shardmap_ctx` is now named
    `potrs_jit_ctx`, with the same arguments and outputs.

All three interfaces accept `return_logdet=True`. Since the factorization
produces $A=LL^H$, the backend computes

$$
\log\det(A) = 2\sum_i \log |L_{ii}|
$$

directly from the distributed Cholesky factor.

::: jaxmg.potrs

::: jaxmg.potrs_jit_ctx

::: jaxmg.potrs_shardmap_ctx
