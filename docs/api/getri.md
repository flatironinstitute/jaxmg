# jaxmg.getri

`getri` computes the full inverse of a general nonsingular matrix $A$ using a distributed LU factorization:

$$
PA=LU,\qquad A^{-1}=U^{-1}L^{-1}P.
$$

Pass the original matrix $A$, not precomputed factors. Factorization and
inversion run in one native call and reuse the same matrix buffer. Use
[`lu_solve`](lu_solve.md) when only a linear solve is needed.

Use `getri_jit_ctx` inside a caller-owned `jax.jit`, or
`getri_shardmap_ctx` for local matrix blocks inside an existing
`jax.shard_map`. Both return `(inverse, status)`; no separate work-matrix
result is needed.

::: jaxmg.getri

---

::: jaxmg.getri_jit_ctx

::: jaxmg.getri_shardmap_ctx
