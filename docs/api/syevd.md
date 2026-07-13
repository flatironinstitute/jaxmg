# jaxmg.syevd

`syevd` computes both eigenvalues and eigenvectors. The input matrix is
symmetric for real dtypes and Hermitian for complex dtypes. Eigenvalues are
real, while eigenvectors use the input dtype and are returned in the original
JAX-facing matrix layout.

SYEVD materializes a distributed eigenvector matrix and requests
solver-specific cuSOLVERMp workspace. Its peak memory requirement can therefore
be substantially larger than a linear solve at the same matrix size.

::: jaxmg.syevd
