# jaxmg.least_squares

`least_squares` solves an overdetermined linear system using distributed QR
factorization:

$$
X = \underset{X}{\operatorname{argmin}}\;\lVert A X-B\rVert_2,
\qquad A\in\mathbb{F}^{M\times N},\quad M\geq N.
$$

The solve input $B$ may be a vector or a matrix containing multiple columns.
The returned solution has shape `(N,)` or `(N, K)`, respectively.
The RHS may have replicated or distributed columns. On grids with multiple
process columns, cuSOLVERMp requires $B$ to span at least as many column tiles
as $A$. Use a one-column process grid for a vector or a small number of RHS
columns; routing padding does not change this requirement.

Use `least_squares` for a direct solve. Use `least_squares_jit_ctx` when
the solve is part of a larger caller-owned `jax.jit`. Since cuSOLVERMp
overwrites both distributed inputs, the context interface returns `a_work` and
`b_work` as shape-compatible alias targets for donated arguments.

::: jaxmg.least_squares

---

::: jaxmg.least_squares_jit_ctx

---

Use `least_squares_shardmap_ctx` for local blocks inside your own `jax.shard_map`.
The previous global-array context interface is now named `least_squares_jit_ctx`.

::: jaxmg.least_squares_shardmap_ctx
