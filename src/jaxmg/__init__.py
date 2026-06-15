"""JAXMg: Multi-GPU Numerical Solvers for JAX.

JAXMg provides high-performance multi-GPU implementations of common numerical
routines, such as Cholesky solves (``potrs``) and symmetric/Hermitian
eigensolvers (``syevd``). It leverages NVIDIA's cuSOLVERMp library and XLA's
native FFI to achieve scalable performance on distributed GPU clusters.

Main Entry Points:
    - :func:`jaxmg.potrs`: Solve ``A x = B`` for positive-definite ``A``.
    - :func:`jaxmg.syevd`: Compute eigenvalues and eigenvectors.
"""

from ._potrs import potrs
from ._syevd import syevd

__all__ = [
    "potrs",
    "syevd",
]
