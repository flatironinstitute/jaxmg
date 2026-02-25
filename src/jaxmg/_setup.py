import importlib
import pathlib
import ctypes
import warnings
import sys
import os
import re

import jax
import jax.extend

from .utils import JaxMgWarning
from .utils import determine_distributed_setup

_lib_dir = os.path.dirname(__file__)
_initialized = False

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


def _initialize():
    if any("gpu" == d.platform for d in jax.devices()):
        # Determine CUDA backend
        backend = jax.extend.backend.get_backend()
        m = re.search(r"cuda[^0-9]*([0-9]+(?:\.[0-9]+)*)", backend.platform_version, re.I)
        if m:
            cuda_major = m.group(1)[:2]
        else:
            raise OSError("Unable to parse CUDA version")
        bin_dir = f"cu{cuda_major}"

        # Load Cusolver
        _load("cusolver", ["libcusolverMg.so.11"])
        _load("cu13", ["libcusolverMg.so.12"])

        jax.config.update("jax_enable_x64", True)

        if not jax.distributed.is_initialized():
            n_machines = 1
            n_devices_per_node = jax.local_device_count()
            mode = "SPMD"
        else:
            # Necessary to ensure jaxmg can be imported during compile time.
            with jax.ensure_compile_time_eval():
                n_machines, n_devices_per_node, _, mode = determine_distributed_setup()
            if n_machines > 1:
                warnings.warn(
                    "Computation seems to be running on multiple machines.\n"
                    "Ensure that jaxmg is only called over a local device mesh, otherwise process might hang.\n"
                    "See examples for how this can be safely achieved.",
                    JaxMgWarning,
                    stacklevel=4,  # _initialize -> ensure_init_jaxmg_backend -> public fn -> user code
                )

        os.environ["JAXMG_NUMBER_OF_DEVICES"] = str(n_devices_per_node)

        if mode == "SPMD":
            library_cyclic = ctypes.cdll.LoadLibrary(os.path.join(_lib_dir, f"{bin_dir}/libcyclic.so"))
            library_potrs = ctypes.cdll.LoadLibrary(os.path.join(_lib_dir, f"{bin_dir}/libpotrs.so"))
            library_potri = ctypes.cdll.LoadLibrary(os.path.join(_lib_dir, f"{bin_dir}/libpotri.so"))
            library_syevd = ctypes.cdll.LoadLibrary(os.path.join(_lib_dir, f"{bin_dir}/libsyevd.so"))
            library_syevd_no_V = ctypes.cdll.LoadLibrary(os.path.join(_lib_dir, f"{bin_dir}/libsyevd_no_V.so"))

            jax.ffi.register_ffi_target("cyclic_mg", jax.ffi.pycapsule(library_cyclic.CyclicMgFFI), platform="CUDA")
            jax.ffi.register_ffi_target("potrs_mg", jax.ffi.pycapsule(library_potrs.PotrsMgFFI), platform="CUDA")
            jax.ffi.register_ffi_target("potri_mg", jax.ffi.pycapsule(library_potri.PotriMgFFI), platform="CUDA")
            jax.ffi.register_ffi_target("syevd_mg", jax.ffi.pycapsule(library_syevd.SyevdMgFFI), platform="CUDA")
            jax.ffi.register_ffi_target(
                "syevd_no_V_mg", jax.ffi.pycapsule(library_syevd_no_V.SyevdMgFFI), platform="CUDA"
            )

        else:
            library_potrs_mp = ctypes.cdll.LoadLibrary(os.path.join(_lib_dir, f"{bin_dir}/libpotrs_mp.so"))
            library_potri_mp = ctypes.cdll.LoadLibrary(os.path.join(_lib_dir, f"{bin_dir}/libpotri_mp.so"))
            library_syevd_mp = ctypes.cdll.LoadLibrary(os.path.join(_lib_dir, f"{bin_dir}/libsyevd_mp.so"))
            library_syevd_no_V_mp = ctypes.cdll.LoadLibrary(os.path.join(_lib_dir, f"{bin_dir}/libsyevd_no_V_mp.so"))

            jax.ffi.register_ffi_target("potrs_mg", jax.ffi.pycapsule(library_potrs_mp.PotrsMgMpFFI), platform="CUDA")
            jax.ffi.register_ffi_target("potri_mg", jax.ffi.pycapsule(library_potri_mp.PotriMgMpFFI), platform="CUDA")
            jax.ffi.register_ffi_target("syevd_mg", jax.ffi.pycapsule(library_syevd_mp.SyevdMgMpFFI), platform="CUDA")
            jax.ffi.register_ffi_target(
                "syevd_no_V_mg", jax.ffi.pycapsule(library_syevd_no_V_mp.SyevdNoVMgMpFFI), platform="CUDA"
            )

    else:
        warnings.warn(
            "No GPUs found, only use this mode for testing or generating documentation.",
            JaxMgWarning,
            stacklevel=4,  # _initialize -> ensure_init_jaxmg_backend -> public fn -> user code
        )
        os.environ["JAXMG_NUMBER_OF_DEVICES"] = str(jax.device_count())


def ensure_init_jaxmg_backend():
    global _initialized
    if _initialized:
        return
    _initialized = True
    _initialize()
