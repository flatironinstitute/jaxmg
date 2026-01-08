import importlib
import pathlib
import ctypes
import warnings
import sys
import os
import re

from .utils import JaxMgWarning

if not sys.platform.startswith("linux"):
    warnings.warn(
        f"Unsupported platform {sys.platform}, only Linux is supported. Non-Linux only works for docs.",
        JaxMgWarning,
        stacklevel=2,
    )


def _load(module, libraries):
    try:
        m = importlib.import_module(f"nvidia.{module}")
    except ImportError:
        m = None

    for lib in libraries:
        if m is not None:
            path = pathlib.Path(m.__path__[0]) / "lib" / lib
            try:
                ctypes.cdll.LoadLibrary(path)
                continue
            except OSError as e:
                raise OSError(
                    f"Unable to load CUDA library {lib}, make sure you have a version of JAX that is "
                    "GPU compatible: jax[cuda12], jax[cuda12-local] (>=0.6.2) or jax[cuda13], jax[cuda13-local] (>=0.7.2)."
                    "This is guaranteed if you install JAXMg as: jaxmg[cuda12], jaxmg[cuda12-local], jaxmg[cuda13] or jaxmg[cuda13-local]"
                ) from e

import jax
# Only import libraries for GPU compatible JAX
if any("gpu" == d.platform for d in jax.devices()):
    import jax.extend
    # Determine CUDA backend
    backend = jax.extend.backend.get_backend()
    m = re.search(r"cuda[^0-9]*([0-9]+(?:\.[0-9]+)*)", backend.platform_version, re.I)
    cuda_major = ""
    if m:
        cuda_major = m.group(1)[:2]
        print(f"CUDA major: {cuda_major}")
    else:
        raise OSError("Unable to parse CUDA version")
    bin_dir = f"cu{cuda_major}"
    # Load Cusolver
    _load("cusolver", ["libcusolverMg.so.11"])
    _load("cu13", ["libcusolverMg.so.12"])

    jax.config.update("jax_enable_x64", True)

    from .utils import determine_distributed_setup

    import jax.extend
    # Determine CUDA backend
    backend = jax.extend.backend.get_backend()
    m = re.search(r"cuda[^0-9]*([0-9]+(?:\.[0-9]+)*)", backend.platform_version, re.I)
    cuda_major = ""
    if m:
        cuda_major = m.group(1)[:2]
        print(f"CUDA major: {cuda_major}")
    else:
        raise OSError("Unable to parse CUDA version")
    bin_dir = f"cu{cuda_major}"
    # Load Cusolver
    _load("cusolver", ["libcusolverMg.so.11"])
    _load("cu13", ["libcusolverMg.so.12"])

    jax.config.update("jax_enable_x64", True)

    from .utils import determine_distributed_setup

    n_machines, n_devices_per_node, n_devices_per_process, mode = (
        determine_distributed_setup()
    )
    os.environ["JAXMG_NUMBER_OF_DEVICES"] = str(n_devices_per_node)
    if n_machines > 1:
        warnings.warn(
            f"Computation seems to be running on multiple machines.\n"
            "Ensure that jaxmg is only called over a local device mesh, otherwise process might hang.\n"
            "See examples for how this can be safely achieved.",
            JaxMgWarning,
            stacklevel=2,
        )
    if mode == "SPMD":
        # Load the shared libraries

        SHARED_LIBRARY_CYCLIC = os.path.join(
            os.path.dirname(__file__), f"{bin_dir}/libcyclic.so"
        )
        library_cyclic = ctypes.cdll.LoadLibrary(SHARED_LIBRARY_CYCLIC)
        SHARED_LIBRARY_POTRS = os.path.join(
            os.path.dirname(__file__), f"{bin_dir}/libpotrs.so"
        )
        library_potrs = ctypes.cdll.LoadLibrary(SHARED_LIBRARY_POTRS)
        SHARED_LIBRARY_POTRI = os.path.join(
            os.path.dirname(__file__), f"{bin_dir}/libpotri.so"
        )
        library_potri = ctypes.cdll.LoadLibrary(SHARED_LIBRARY_POTRI)
        SHARED_LIBRARY_SYEVD = os.path.join(
            os.path.dirname(__file__), f"{bin_dir}/libsyevd.so"
        )
        library_syevd = ctypes.cdll.LoadLibrary(SHARED_LIBRARY_SYEVD)
        SHARED_LIBRARY_SYEVD_NO_V = os.path.join(
            os.path.dirname(__file__), f"{bin_dir}/libsyevd_no_V.so"
        )
        library_syevd_no_V = ctypes.cdll.LoadLibrary(SHARED_LIBRARY_SYEVD_NO_V)
        # Register FFI targets
        jax.ffi.register_ffi_target(
            "cyclic_mg", jax.ffi.pycapsule(library_cyclic.CyclicMgFFI), platform="CUDA"
        )
        jax.ffi.register_ffi_target(
            "potrs_mg", jax.ffi.pycapsule(library_potrs.PotrsMgFFI), platform="CUDA"
        )
        jax.ffi.register_ffi_target(
            "potri_mg", jax.ffi.pycapsule(library_potri.PotriMgFFI), platform="CUDA"
        )
        jax.ffi.register_ffi_target(
            "syevd_mg", jax.ffi.pycapsule(library_syevd.SyevdMgFFI), platform="CUDA"
        )
        jax.ffi.register_ffi_target(
            "syevd_no_V_mg",
            jax.ffi.pycapsule(library_syevd_no_V.SyevdMgFFI),
            platform="CUDA",
        )

    else:
        # Load the shared library
        SHARED_LIBRARY_POTRS_MP = os.path.join(
            os.path.dirname(__file__), f"{bin_dir}/libpotrs_mp.so"
        )
        library_potrs_mp = ctypes.cdll.LoadLibrary(SHARED_LIBRARY_POTRS_MP)
        # Register FFI targets
        jax.ffi.register_ffi_target(
            "potrs_mg",
            jax.ffi.pycapsule(library_potrs_mp.PotrsMgMpFFI),
            platform="CUDA",
        )
    from ._potrs import potrs, potrs_shardmap_ctx
    from ._potri import potri, potri_shardmap_ctx, potri_symmetrize
    from ._syevd import syevd, syevd_shardmap_ctx

else:
    warnings.warn(
        f"No GPUs found, only use this mode for testing or generating documentation.",
        JaxMgWarning,
        stacklevel=2,
    )
    from ._potrs import potrs, potrs_shardmap_ctx
    from ._potri import potri, potri_shardmap_ctx, potri_symmetrize
    from ._syevd import syevd, syevd_shardmap_ctx

    os.environ["JAXMG_NUMBER_OF_DEVICES"] = str(jax.device_count())

from ._cyclic_1d import (
    cyclic_1d,
    calculate_padding,
    pad_rows,
    unpad_rows,
    verify_cyclic,
    get_cols_cyclic,
    plot_block_to_cyclic,
)

__all__ = [
    "potrs",
    "potrs_shardmap_ctx",
    "potri",
    "potri_shardmap_ctx",
    "potri_symmetrize",
    "syevd",
    "syevd_shardmap_ctx",
    "cyclic_1d",
    "pad_rows",
    "unpad_rows",
    "verify_cyclic",
    "calculate_padding",
    "get_cols_cyclic",
    "plot_block_to_cyclic",
    "determine_distributed_setup",
]
