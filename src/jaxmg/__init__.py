"""JAXMg: Multi-GPU Numerical Solvers for JAX.

JAXMg provides high-performance multi-GPU implementations of common numerical
routines, including Cholesky and LU solves, symmetric/Hermitian eigensolvers,
and singular-value decompositions. It leverages NVIDIA's cuSOLVERMp library and
XLA's native FFI to achieve scalable performance on distributed GPU clusters.

Main Entry Points:
    - :func:`jaxmg.potrs`: Solve ``A x = B`` for positive-definite ``A``.
    - :func:`jaxmg.lu_solve`: Solve ``A x = B`` for general nonsingular ``A``.
    - :func:`jaxmg.syevd`: Compute eigenvalues and optional eigenvectors.
    - :func:`jaxmg.gesvd`: Compute singular values and optional singular vectors.
"""

from importlib.metadata import version

from ._gesvd import gesvd, gesvd_shardmap_ctx
from ._lu_solve import lu_solve, lu_solve_shardmap_ctx
from ._potrs import potrs, potrs_shardmap_ctx
from ._syevd import syevd, syevd_shardmap_ctx
from ._device import device_supports_vmm

__version__ = version("jaxmg")

__all__ = [
    "__version__",
    "gesvd",
    "gesvd_shardmap_ctx",
    "lu_solve",
    "lu_solve_shardmap_ctx",
    "potrs",
    "potrs_shardmap_ctx",
    "syevd",
    "syevd_shardmap_ctx",
    "device_supports_vmm",
]
