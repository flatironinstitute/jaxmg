# jaxmg.syevd

`syevd` computes the eigenvalues and, by default, eigenvectors of a symmetric
real matrix or Hermitian complex matrix. Eigenvalues are real, while
eigenvectors use the input dtype and are returned in the original JAX-facing
matrix layout.

Set `return_eigenvectors=False` when only the spectrum is required. This selects
the eigenvalues-only mode and reduces memory use by avoiding the full
distributed eigenvector matrix.

Use `syevd` for a direct eigensolve. Use `syevd_shardmap_ctx` when the
eigensolve is part of a larger caller-owned `jax.jit`. The context interface
returns the native input-matrix work buffer so the outer compiled function can
preserve the donated input alias. cuSOLVERMp stores the overwritten input
matrix and eigenvectors in separate distributed buffers, so `a_work` is the
alias target rather than the eigenvector output when vectors are requested.

::: jaxmg.syevd

---

::: jaxmg.syevd_shardmap_ctx
