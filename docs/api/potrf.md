# jaxmg.potrf

`potrf` computes the lower Cholesky factor of a symmetric or Hermitian
positive-definite matrix $A$:

$$
A = L L^{\dagger}.
$$

Use `potrf` for a direct factorization. Use `potrf_shardmap_ctx` when the
factorization is part of a larger caller-owned `jax.jit`. The factor aliases
the input work buffer, so the context interface returns $L$ directly rather
than a separate opaque work result.

::: jaxmg.potrf

---

::: jaxmg.potrf_shardmap_ctx
