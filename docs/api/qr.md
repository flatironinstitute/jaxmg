# jaxmg.qr

`qr` computes the reduced QR decomposition of a tall or square matrix $A$:

$$
A = Q R, \qquad Q^{\dagger}Q=I,
$$

where an $M\times N$ input with $M\geq N$ produces an $M\times N$ matrix $Q$
and an $N\times N$ upper-triangular matrix $R$.

Use `qr` for a direct decomposition. Use `qr_shardmap_ctx` when the
decomposition is part of a larger caller-owned `jax.jit`. cuSOLVERMp overwrites
the input matrix with $Q$, so the context interface returns $Q$ directly as the
donated input alias.

::: jaxmg.qr

---

::: jaxmg.qr_shardmap_ctx
