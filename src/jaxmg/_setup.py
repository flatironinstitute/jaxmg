"""Backend initialization and native FFI target registration for JAXMg.

This module locates the packaged native backend for the active CUDA runtime,
preloads the required NVIDIA wheel libraries, loads the backend with
``ctypes``, and registers the exported XLA FFI targets with JAX. The native
backend borrows XLA's communicator during each FFI call.

The cuSOLVERMp backend requires rank-per-GPU execution: each Python process
must own exactly one local GPU.  Multi-GPU runs should therefore be launched
with ``jax.distributed.initialize(...)`` and one process per GPU.
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
        "cusolvermp_potrs_logdet",
        {
            "prepare": "XlaCusolverMpPotrsLogdetPrepareFFI",
            "execute": "XlaCusolverMpPotrsLogdetFFI",
        },
    ),
    (
        "cusolvermp_lu_solve",
        {
            "prepare": "XlaCusolverMpLuSolvePrepareFFI",
            "execute": "XlaCusolverMpLuSolveFFI",
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


def _validate_rank_per_gpu_runtime():
    """Validate that the current JAX process owns exactly one local GPU.

    cuSOLVERMp is a distributed runtime: every participating GPU is represented
    by one process/rank.  JAXMg therefore rejects the old single-process,
    multi-device regime before native FFI targets are registered.  Users remain
    responsible for calling ``jax.distributed.initialize(...)`` before device
    discovery in multi-node programs.
    """
    local_device_count = jax.local_device_count()
    if local_device_count != 1:
        raise RuntimeError(
            "JAXMg's cuSOLVERMp backend supports only rank-per-GPU execution: "
            "each Python process must see exactly one local GPU. This process "
            f"sees {local_device_count} local GPUs. Launch one process per GPU "
            "with jax.distributed.initialize(..., local_device_ids=[...]) or "
            "restrict CUDA_VISIBLE_DEVICES to a single GPU per process."
        )


def _initialize():
    """Initialize native CUDA FFI targets for the active JAX runtime."""
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

        _validate_rank_per_gpu_runtime()

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


def ensure_init_jaxmg_backend():
    """Ensure that the JAXMg native backend and FFI targets are initialized.

    This function should be called by every public JAXMg entry point before
    executing any native FFI calls. It performs one-time initialization:

    1. identifies the CUDA version from the JAX backend;
    2. verifies that the process owns exactly one local GPU;
    3. loads the packaged native backend library; and
    4. registers production JAX FFI targets such as ``cusolvermp_potrs``.
    """
    global _initialized
    if not _initialized:
        _initialized = True
        _initialize()
