"""Backend initialization and native FFI target registration for JAXMg.

This module locates the packaged native backend for the active CUDA runtime,
loads it with ``ctypes``, and registers the exported XLA FFI targets with JAX.
The native backend is responsible for loading cuSOLVERMp/NCCL-compatible
runtime libraries and for borrowing XLA's communicator during each FFI call.

The setup path supports both single-process multi-GPU execution and JAX
distributed runs with one Python process per local GPU.
"""

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

_lib_dir = os.path.dirname(__file__)
_initialized = False
_diagnostics_initialized = False
_runtime_mode = None
_cuda_bin_dir = None
_xla_comm_backend_library = "libjaxmg_xla_comm_backend.so"

_PRODUCTION_FFI_TARGETS = (
    (
        "cusolvermp_potrs",
        {
            "prepare": "XlaCusolverMpPotrsPrepareFFI",
            "execute": "XlaCusolverMpPotrsFFI",
        },
    ),
    (
        "cusolvermp_syevd",
        {
            "prepare": "XlaCusolverMpSyevdPrepareFFI",
            "execute": "XlaCusolverMpSyevdFFI",
        },
    ),
)

_DIAGNOSTIC_FFI_TARGETS = (
    (
        "cusolvermp_init_probe",
        {
            "prepare": "XlaCusolverMpInitProbePrepareFFI",
            "execute": "XlaCusolverMpInitProbeFFI",
        },
    ),
    (
        "cusolvermp_scatter_layout_probe",
        {
            "prepare": "XlaCusolverMpScatterLayoutProbePrepareFFI",
            "execute": "XlaCusolverMpScatterLayoutProbeFFI",
        },
    ),
    (
        "cusolvermp_potrs_probe",
        {
            "prepare": "XlaCusolverMpPotrsProbePrepareFFI",
            "execute": "XlaCusolverMpPotrsProbeFFI",
        },
    ),
    (
        "cusolvermp_distributed_potrs_probe",
        {
            "prepare": "XlaCusolverMpDistributedPotrsProbePrepareFFI",
            "execute": "XlaCusolverMpDistributedPotrsProbeFFI",
        },
    ),
    (
        "cusolvermp_syevd_probe",
        {
            "prepare": "XlaCusolverMpSyevdProbePrepareFFI",
            "execute": "XlaCusolverMpSyevdProbeFFI",
        },
    ),
)

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


def _register_optional_cuda_target_bundle(bin_dir, library_name, ffi_name, symbols):
    path = os.path.join(_lib_dir, f"{bin_dir}/{library_name}")
    if not os.path.exists(path):
        return
    library = ctypes.cdll.LoadLibrary(path)
    try:
        bundle = {
            stage: jax.ffi.pycapsule(getattr(library, symbol_name))
            for stage, symbol_name in symbols.items()
        }
    except AttributeError:
        return
    jax.ffi.register_ffi_target(ffi_name, bundle, platform="CUDA")


def _register_cuda_target_bundle(bin_dir, library_name, ffi_name, symbols):
    path = os.path.join(_lib_dir, f"{bin_dir}/{library_name}")
    if not os.path.exists(path):
        raise OSError(
            f"Required JAXMg CUDA library is missing: {path}. "
            "Build the XLA communicator backend before using the CUDA package."
        )
    library = ctypes.cdll.LoadLibrary(path)
    bundle = {
        stage: jax.ffi.pycapsule(getattr(library, symbol_name))
        for stage, symbol_name in symbols.items()
    }
    jax.ffi.register_ffi_target(ffi_name, bundle, platform="CUDA")


def _detect_runtime_mode():
    """Return ``(mode, devices_per_node)`` for the current JAX runtime.

    JAXMg has two native orchestration modes:

    - SPMD: one Python process controls all local GPUs.
    - MPMD: one Python process participates per GPU/rank through
      ``jax.distributed``.  In this mode the number of participating local
      ranks must be supplied by ``JAXMG_NUMBER_OF_DEVICES`` when it cannot be
      inferred from the local device set.

    JAXMg does not initialize distributed JAX itself.  Users must call
    ``jax.distributed.initialize(...)`` before device discovery in multi-node
    programs; this function only records the resulting runtime shape for native
    diagnostics and FFI handlers.
    """
    requested_mode = os.environ.get("JAXMG_EXECUTION_MODE", "").upper()
    if requested_mode:
        if requested_mode not in {"SPMD", "MPMD"}:
            raise ValueError("JAXMG_EXECUTION_MODE must be either SPMD or MPMD.")
        if "JAXMG_NUMBER_OF_DEVICES" in os.environ:
            devices_per_node = int(os.environ["JAXMG_NUMBER_OF_DEVICES"])
        else:
            devices_per_node = jax.local_device_count()
        if devices_per_node <= 0:
            raise ValueError("JAXMG_NUMBER_OF_DEVICES must be positive.")
        return requested_mode, devices_per_node

    if not jax.distributed.is_initialized():
        return "SPMD", jax.local_device_count()

    local_device_count = jax.local_device_count()
    if local_device_count > 1:
        return "SPMD", local_device_count

    if "JAXMG_NUMBER_OF_DEVICES" in os.environ:
        devices_per_node = int(os.environ["JAXMG_NUMBER_OF_DEVICES"])
        if devices_per_node <= 0:
            raise ValueError("JAXMG_NUMBER_OF_DEVICES must be positive.")
        return "MPMD", devices_per_node

    # In a multi-process JAX runtime, local_device_count is commonly one. Keep
    # the old package's conservative default but warn loudly because a wrong
    # value can make collective clique construction hang.
    return "MPMD", jax.device_count()


def _register_diagnostic_targets():
    global _diagnostics_initialized
    if _diagnostics_initialized or _cuda_bin_dir is None:
        return
    for ffi_name, symbols in _DIAGNOSTIC_FFI_TARGETS:
        _register_optional_cuda_target_bundle(
            _cuda_bin_dir,
            _xla_comm_backend_library,
            ffi_name,
            symbols,
        )
    _diagnostics_initialized = True


def _initialize():
    global _runtime_mode, _cuda_bin_dir
    if any("gpu" == d.platform for d in jax.devices()):
        # Determine CUDA backend
        backend = jax.extend.backend.get_backend()
        m = re.search(r"cuda[^0-9]*([0-9]+(?:\.[0-9]+)*)", backend.platform_version, re.I)
        if m:
            cuda_major = m.group(1)[:2]
        else:
            raise OSError("Unable to parse CUDA version")
        bin_dir = f"cu{cuda_major}"

        jax.config.update("jax_enable_x64", True)

        mode, n_devices_per_node = _detect_runtime_mode()
        if mode == "MPMD":
            warnings.warn(
                "Running the XLA communicator backend in experimental MPMD "
                f"mode with JAXMG_NUMBER_OF_DEVICES={n_devices_per_node}. "
                "Use rank-per-GPU launch tests before relying on this mode in "
                "production.",
                JaxMgWarning,
                stacklevel=4,
            )
        _runtime_mode = mode
                
        # set if not set already
        os.environ.setdefault("JAXMG_NUMBER_OF_DEVICES", str(n_devices_per_node))
        os.environ.setdefault("JAXMG_EXECUTION_MODE", mode)
        _cuda_bin_dir = bin_dir

        for ffi_name, symbols in _PRODUCTION_FFI_TARGETS:
            _register_cuda_target_bundle(
                bin_dir,
                _xla_comm_backend_library,
                ffi_name,
                symbols,
            )
    else:
        warnings.warn(
            "No GPUs found, only use this mode for testing or generating documentation.",
            JaxMgWarning,
            stacklevel=4,  # _initialize -> ensure_init_jaxmg_backend -> public fn -> user code
        )
        os.environ["JAXMG_NUMBER_OF_DEVICES"] = str(jax.device_count())


def ensure_init_jaxmg_backend(*, include_diagnostics: bool = False):
    """Ensure that the JAXMg native backend and FFI targets are initialized.

    This function should be called by every public JAXMg entry point before
    executing any native FFI calls. It performs one-time initialization:

    1. identifies the CUDA version from the JAX backend;
    2. detects the JAX runtime mode;
    3. loads the packaged native backend library; and
    4. registers production JAX FFI targets such as ``cusolvermp_potrs``.

    Args:
        include_diagnostics: If True, also register cuSOLVERMp diagnostic FFI
            targets. Default is False.
    """
    global _initialized
    if not _initialized:
        _initialized = True
        _initialize()
    if include_diagnostics:
        _register_diagnostic_targets()
