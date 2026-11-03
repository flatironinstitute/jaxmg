# jaxmg.getrf

`getrf` computes the pivoted LU factorization of a square nonsingular matrix
$A$:

$$
P A = L U.
$$

The result `lu` stores the strictly lower part of the unit-lower factor $L$
and the upper-triangular factor $U$. The replicated vector `piv` records the
zero-based row interchanges.

Use `getrf` for a direct factorization. Use `getrf_shardmap_ctx` when the
factorization is part of a larger caller-owned `jax.jit`. The packed factor
aliases the input work buffer, so no separate opaque work result is needed.

::: jaxmg.getrf

---

::: jaxmg.getrf_shardmap_ctx
