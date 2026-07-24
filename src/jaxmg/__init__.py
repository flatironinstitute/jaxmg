"""JAXMg: Multi-GPU Numerical Solvers for JAX.

JAXMg provides high-performance multi-GPU implementations of common numerical
routines, such as Cholesky solves (``potrs``) and symmetric/Hermitian
eigensolvers (``syevd``). It leverages NVIDIA's cuSOLVERMp library and XLA's
native FFI to achieve scalable performance on distributed GPU clusters.

Main Entry Points:
    - :func:`jaxmg.potrs`: Solve ``A x = B`` for positive-definite ``A``.
    - :func:`jaxmg.lu_solve`: Solve ``A x = B`` for general nonsingular ``A``.
    - :func:`jaxmg.syevd`: Compute eigenvalues and eigenvectors.
"""

from importlib.metadata import version

from ._lu_solve import lu_solve, lu_solve_shardmap_ctx
from ._potrs import potrs, potrs_shardmap_ctx
from ._syevd import syevd
from ._device import device_supports_vmm

__version__ = version("jaxmg")

__all__ = [
    "__version__",
    "lu_solve",
    "lu_solve_shardmap_ctx",
    "potrs",
    "potrs_shardmap_ctx",
    "syevd",
    "device_supports_vmm",
]
