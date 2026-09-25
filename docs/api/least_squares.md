# jaxmg.least_squares

`least_squares` solves an overdetermined linear system using distributed QR
factorization:

$$
X = \underset{X}{\operatorname{argmin}}\;\lVert A X-B\rVert_2,
\qquad A\in\mathbb{F}^{M\times N},\quad M\geq N.
$$

The solve input $B$ may be a vector or a matrix containing multiple columns.
The returned solution has shape `(N,)` or `(N, K)`, respectively.
Each process-grid column must own at least one block-cyclic tile of $B$; a
vector input therefore requires a process grid with one column.

Use `least_squares` for a direct solve. Use `least_squares_shardmap_ctx` when
the solve is part of a larger caller-owned `jax.jit`. Since cuSOLVERMp
overwrites both distributed inputs, the context interface returns `a_work` and
`b_work` as shape-compatible alias targets for donated arguments.

::: jaxmg.least_squares

---

::: jaxmg.least_squares_shardmap_ctx
