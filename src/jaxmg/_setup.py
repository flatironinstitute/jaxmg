"""Backend initialization and native FFI target registration for JAXMg.

This module locates the packaged native backend for the active CUDA runtime,
preloads the required NVIDIA wheel libraries, loads the backend with
``ctypes``, and registers the exported XLA FFI targets with JAX. The native
backend borrows XLA's communicator during each FFI call.

The setup path supports both single-process multi-GPU execution and JAX
distributed runs with one Python process per local GPU.
"""

import pathlib
import ctypes
import site
import warnings
import sys
import os
import re

import jax
import jax.extend

from .utils import JaxMgWarning

_lib_dir = os.path.dirname(__file__)
_initialized = False
_runtime_mode = None
_xla_comm_backend_library = "libjaxmg_xla_comm_backend.so"
_preloaded_cuda_libraries = {}

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

if not sys.platform.startswith("linux"):
    warnings.warn(
        f"Unsupported platform {sys.platform}, only Linux is supported. Non-Linux only works for docs.",
        JaxMgWarning,
        stacklevel=2,
    )


def _candidate_python_roots():
    """Yield Python installation roots that may contain NVIDIA wheel payloads."""
    roots = [pathlib.Path(path) for path in sys.path if path]
    try:
        roots.extend(pathlib.Path(path) for path in site.getsitepackages())
    except AttributeError:
        pass
    try:
        roots.append(pathlib.Path(site.getusersitepackages()))
    except AttributeError:
        pass
    seen = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            resolved = root
        if resolved in seen:
            continue
        seen.add(resolved)
        yield root


def _find_nvidia_cuda_library(cuda_major, library_name):
    """Find a CUDA-library wheel payload such as nvidia/cu12/lib/*.so."""
    relative_path = pathlib.Path("nvidia") / f"cu{cuda_major}" / "lib" / library_name
    for root in _candidate_python_roots():
        path = root / relative_path
        if path.exists():
            return path
    return None


def _preload_cusolvermp_runtime(cuda_major):
    """Load libcusolverMp from the nvidia-cusolvermp-cuXX dependency wheel."""
    library_name = "libcusolverMp.so.0"
    cache_key = (cuda_major, library_name)
    if cache_key in _preloaded_cuda_libraries:
        return

    path = _find_nvidia_cuda_library(cuda_major, library_name)
    if path is None:
        raise OSError(
            f"Unable to find {library_name} from nvidia-cusolvermp-cu{cuda_major}. "
            f"Install JAXMg with the matching CUDA extra, e.g. jaxmg[cuda{cuda_major}]."
        )
    try:
        _preloaded_cuda_libraries[cache_key] = ctypes.CDLL(
            str(path), mode=ctypes.RTLD_GLOBAL
        )
    except OSError as e:
        raise OSError(
            f"Unable to load cuSOLVERMp runtime library at {path}. "
            "Make sure the matching NVIDIA CUDA runtime dependency wheels are installed."
        ) from e


def _register_cuda_target_bundle(bin_dir, library_name, ffi_name, symbols):
    """Load one packaged CUDA backend and register its staged FFI target."""
    path = os.path.join(_lib_dir, f"{bin_dir}/{library_name}")
    if not os.path.exists(path):
        raise OSError(
            f"Required JAXMg CUDA library is missing: {path}. "
            "Build the XLA communicator backend before using the CUDA package."
        )
    cuda_major = bin_dir.removeprefix("cu")
    _preload_cusolvermp_runtime(cuda_major)
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
    FFI handlers.
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


def _initialize():
    """Initialize native CUDA FFI targets for the active JAX runtime."""
    global _runtime_mode
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


def ensure_init_jaxmg_backend():
    """Ensure that the JAXMg native backend and FFI targets are initialized.

    This function should be called by every public JAXMg entry point before
    executing any native FFI calls. It performs one-time initialization:

    1. identifies the CUDA version from the JAX backend;
    2. detects the JAX runtime mode;
    3. loads the packaged native backend library; and
    4. registers production JAX FFI targets such as ``cusolvermp_potrs``.
    """
    global _initialized
    if not _initialized:
        _initialized = True
        _initialize()
