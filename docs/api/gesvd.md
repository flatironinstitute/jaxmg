# jaxmg.gesvd

`gesvd` computes the singular values and optional singular vectors of a real or
complex rectangular matrix:

$$
A = U \Sigma V^{\dagger}.
$$

The default reduced decomposition returns `U`, `s`, and `Vh`, matching the
ordering used by `jax.numpy.linalg.svd`. Set `full_matrices=True` for complete
square vector matrices. `compute_u` and `compute_vh` select the two vector
outputs independently; disabling an output avoids its matrix allocation and
reverse redistribution.

Use `gesvd` for a direct decomposition. Use `gesvd_shardmap_ctx` when the SVD
is part of a larger caller-owned `jax.jit`. The context interface returns the
overwritten input-matrix work buffer so an outer compiled function can preserve
the donated input alias. cuSOLVERMp requires separate storage for A, U, and Vh,
so requested singular-vector matrices cannot alias the donated input.

::: jaxmg.gesvd

---

::: jaxmg.gesvd_shardmap_ctx
