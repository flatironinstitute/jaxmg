# jaxmg.gesvd

`gesvd` computes the singular-value decomposition of an $M \times N$ real or
complex matrix $A$:

$$
A = U \Sigma V^{\dagger}.
$$

By default, `gesvd` returns the reduced decomposition `(U, s, Vh)`, following
`jax.numpy.linalg.svd`. Set `full_matrices=True` to return full-sized
singular-vector matrices. The left and right singular vectors can be requested
independently with `compute_u` and `compute_vh`; outputs that are not requested
are neither allocated nor redistributed.

Use `gesvd` for a direct decomposition. Use `gesvd_shardmap_ctx` when the SVD
is part of a larger caller-owned `jax.jit`. The context interface returns the
overwritten input-matrix work buffer so an outer compiled function can preserve
the donated input alias. cuSOLVERMp requires separate storage for A, U, and Vh,
so requested singular-vector matrices cannot alias the donated input.

::: jaxmg.gesvd

---

::: jaxmg.gesvd_shardmap_ctx
