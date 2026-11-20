# jaxmg.potri

`potri` computes the full inverse of a symmetric or Hermitian positive-definite matrix $A$ using a distributed Cholesky factorization:

$$
A=LL^{\dagger},\qquad A^{-1}=L^{-\dagger}L^{-1}.
$$

Pass the original matrix $A$, not precomputed factors. Factorization and
inversion run in one native call and reuse the same matrix buffer. Use
[`potrs`](potrs.md) when only a linear solve is needed.

Use `potri_jit_ctx` inside a caller-owned `jax.jit`, or
`potri_shardmap_ctx` for local matrix blocks inside an existing
`jax.shard_map`. Both return `(inverse, status)`; no separate work-matrix
result is needed.

::: jaxmg.potri

---

::: jaxmg.potri_jit_ctx

::: jaxmg.potri_shardmap_ctx
