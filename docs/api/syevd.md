# jaxmg.syevd

`syevd` computes both eigenvalues and eigenvectors. The input matrix is
symmetric for real dtypes and Hermitian for complex dtypes. Eigenvalues are
real, while eigenvectors use the input dtype and are returned in the original
JAX-facing matrix layout.

SYEVD materializes a distributed eigenvector matrix and requests
solver-specific cuSOLVERMp workspace. Its peak memory requirement can therefore
be substantially larger than a linear solve at the same matrix size.

The eigenvalues are returned as a real array. The eigenvectors use the input
dtype and are returned with the same JAX-facing matrix layout as `a`.

Use `syevd` for a direct eigensolve. Use `syevd_shardmap_ctx` when the
eigensolve is part of a larger caller-owned `jax.jit`. The context interface
returns the native input-matrix work buffer so the outer compiled function can
preserve the donated input alias. cuSOLVERMp stores the overwritten input
matrix and eigenvectors in separate distributed buffers, so `a_work` is the
alias target rather than the eigenvector output.

::: jaxmg.syevd

---

::: jaxmg.syevd_shardmap_ctx
