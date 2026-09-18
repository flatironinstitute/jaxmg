# jaxmg.polar

`polar` computes the polar decomposition of a tall or square matrix $A$:

$$
A = U_p H.
$$

For a full-rank $M \times N$ matrix with $M \geq N$, the columns of $U_p$ are
orthonormal and $H$ is an $N \times N$ Hermitian positive-definite matrix. Set
`compute_h=False` when only $U_p$ is required; the $H$ output and its reverse
redistribution are then omitted.

Use `polar` for a direct decomposition. Use `polar_shardmap_ctx` when the
decomposition is part of a larger caller-owned `jax.jit`. cuSOLVERMp overwrites
the input matrix with $U_p$, so the context interface returns $U_p$ directly as
the donated input alias rather than returning a separate work buffer.

::: jaxmg.polar

---

::: jaxmg.polar_shardmap_ctx
