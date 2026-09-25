"""JAXMg: Multi-GPU Numerical Solvers for JAX.

JAXMg provides high-performance multi-GPU implementations of common numerical
routines, including Cholesky and LU solves, symmetric/Hermitian eigensolvers,
least-squares solves, QR, singular-value, and polar decompositions.
It leverages NVIDIA's
cuSOLVERMp library and XLA's native FFI to achieve scalable performance on
distributed GPU clusters.

Main Entry Points:
    - :func:`jaxmg.potrs`: Solve ``A x = B`` for positive-definite ``A``.
    - :func:`jaxmg.lu_solve`: Solve ``A x = B`` for general nonsingular ``A``.
    - :func:`jaxmg.least_squares`: Solve ``min_X ||A X - B||_2``.
    - :func:`jaxmg.syevd`: Compute eigenvalues and optional eigenvectors.
    - :func:`jaxmg.gesvd`: Compute singular values and optional singular vectors.
    - :func:`jaxmg.polar`: Compute a polar decomposition.
    - :func:`jaxmg.qr`: Compute a reduced QR decomposition.
"""

from importlib.metadata import version

from ._gesvd import gesvd, gesvd_shardmap_ctx
from ._least_squares import least_squares, least_squares_shardmap_ctx
from ._lu_solve import lu_solve, lu_solve_shardmap_ctx
from ._potrs import potrs, potrs_shardmap_ctx
from ._polar import polar, polar_shardmap_ctx
from ._qr import qr, qr_shardmap_ctx
from ._syevd import syevd, syevd_shardmap_ctx
from ._device import device_supports_vmm

__version__ = version("jaxmg")

__all__ = [
    "__version__",
    "gesvd",
    "gesvd_shardmap_ctx",
    "lu_solve",
    "lu_solve_shardmap_ctx",
    "least_squares",
    "least_squares_shardmap_ctx",
    "potrs",
    "potrs_shardmap_ctx",
    "polar",
    "polar_shardmap_ctx",
    "qr",
    "qr_shardmap_ctx",
    "syevd",
    "syevd_shardmap_ctx",
    "device_supports_vmm",
]
