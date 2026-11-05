# jaxmg.getrf

`getrf` computes the pivoted LU factorization of a square nonsingular matrix
$A$:

$$
P A = L U.
$$

The result `lu` stores the strictly lower part of the unit-lower factor $L$
and the upper-triangular factor $U$. The replicated vector `piv` records the
zero-based row interchanges.

Use `getrf` for a direct factorization. Use `getrf_jit_ctx` when the
factorization is part of a larger caller-owned `jax.jit`. The packed factor
aliases the input work buffer, so no separate opaque work result is needed.

::: jaxmg.getrf

---

::: jaxmg.getrf_jit_ctx

Use `getrf_shardmap_ctx` inside an existing `jax.shard_map`. It accepts a local
matrix block and returns a local factor block, replicated pivots and per-rank
status.

!!! Note "Context API migration"

    The previous global-array `getrf_shardmap_ctx` is now named `getrf_jit_ctx`,
    with the same arguments and outputs.

::: jaxmg.getrf_shardmap_ctx
