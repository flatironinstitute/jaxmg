# jaxmg.potrf

`potrf` computes the lower Cholesky factor of a symmetric or Hermitian
positive-definite matrix $A$:

$$
A = L L^{\dagger}.
$$

Use `potrf` for a direct factorization. Use `potrf_jit_ctx` when the
factorization is part of a larger caller-owned `jax.jit`. The factor aliases
the input work buffer, so the context interface returns $L$ directly rather
than a separate opaque work result.

Inside an existing `jax.shard_map`, use `potrf_shardmap_ctx` with local matrix
blocks. It returns local factor blocks and per-rank status.

!!! Note "Context API migration"

    The previous global-array `potrf_shardmap_ctx` is now named
    `potrf_jit_ctx`, with the same arguments and outputs.

::: jaxmg.potrf

---

::: jaxmg.potrf_jit_ctx

::: jaxmg.potrf_shardmap_ctx
