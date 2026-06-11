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
_runtime_mode = None
_xla_comm_backend_library = "libjaxmg_xla_comm_backend.so"

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


def _register_xla_comm_cusolvermg_targets(bin_dir):
    _register_cuda_target_bundle(
        bin_dir,
        _xla_comm_backend_library,
        "xla_comm_matrix_column_native_plan",
        {
            "prepare": "XlaCommMatrixColumnNativePlanPrepareFFI",
            "execute": "XlaCommMatrixColumnNativePlanFFI",
        },
    )
    _register_cuda_target_bundle(
        bin_dir,
        _xla_comm_backend_library,
        "potrs_mg",
        {
            "prepare": "XlaCommPotrsMgNativePlanPrepareFFI",
            "execute": "XlaCommPotrsMgNativePlanFFI",
        },
    )
    _register_cuda_target_bundle(
        bin_dir,
        _xla_comm_backend_library,
        "potri_mg",
        {
            "prepare": "XlaCommPotriMgNativePlanPrepareFFI",
            "execute": "XlaCommPotriMgNativePlanFFI",
        },
    )
    _register_cuda_target_bundle(
        bin_dir,
        _xla_comm_backend_library,
        "syevd_mg",
        {
            "prepare": "XlaCommSyevdMgNativePlanPrepareFFI",
            "execute": "XlaCommSyevdMgNativePlanFFI",
        },
    )
    _register_cuda_target_bundle(
        bin_dir,
        _xla_comm_backend_library,
        "syevd_no_V_mg",
        {
            "prepare": "XlaCommSyevdNoVMgNativePlanPrepareFFI",
            "execute": "XlaCommSyevdNoVMgNativePlanFFI",
        },
    )


def _detect_runtime_mode():
    """Return ``(mode, devices_per_node)`` for the current JAX runtime.

    JAXMg has two single-node cuSolverMg orchestration modes:

    - SPMD: one Python process controls all local GPUs.
    - MPMD: one Python process participates per GPU/rank through
      ``jax.distributed``. cuSolverMg is still single-node, so the number of
      participating local ranks must be supplied by ``JAXMG_NUMBER_OF_DEVICES``
      when it cannot be inferred from a non-distributed local device set.
    """
    if not jax.distributed.is_initialized():
        return "SPMD", jax.local_device_count()

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

        # Load Cusolver
        _load("cusolver", ["libcusolverMg.so.11"])
        _load("cu13", ["libcusolverMg.so.12"])

        jax.config.update("jax_enable_x64", True)

        mode, n_devices_per_node = _detect_runtime_mode()
        if mode == "MPMD":
            warnings.warn(
                "Running the XLA communicator backend in experimental MPMD "
                f"mode with JAXMG_NUMBER_OF_DEVICES={n_devices_per_node}. "
                "The XLA communicator probes are expected to work first; "
                "fused cuSolverMg solvers still require the MPMD IPC pointer "
                "sharing layer to be enabled.",
                JaxMgWarning,
                stacklevel=4,
            )
        _runtime_mode = mode
                
        # set if not set already
        os.environ.setdefault("JAXMG_NUMBER_OF_DEVICES", str(n_devices_per_node))
        os.environ.setdefault("JAXMG_EXECUTION_MODE", mode)

        _register_xla_comm_cusolvermg_targets(bin_dir)
        _register_optional_cuda_target_bundle(
            bin_dir,
            _xla_comm_backend_library,
            "xla_comm_collective_probe",
            {
                "prepare": "XlaCommCollectiveProbePrepareFFI",
                "execute": "XlaCommCollectiveProbeFFI",
            },
        )
        _register_optional_cuda_target_bundle(
            bin_dir,
            _xla_comm_backend_library,
            "xla_comm_allreduce_probe",
            {
                "prepare": "XlaCommAllReduceProbePrepareFFI",
                "execute": "XlaCommAllReduceProbeFFI",
            },
        )
        _register_optional_cuda_target_bundle(
            bin_dir,
            _xla_comm_backend_library,
            "xla_comm_ring_permute_probe",
            {
                "prepare": "XlaCommRingPermuteProbePrepareFFI",
                "execute": "XlaCommRingPermuteProbeFFI",
            },
        )
        _register_optional_cuda_target_bundle(
            bin_dir,
            _xla_comm_backend_library,
            "xla_comm_shift_permute_probe",
            {
                "prepare": "XlaCommShiftPermuteProbePrepareFFI",
                "execute": "XlaCommShiftPermuteProbeFFI",
            },
        )
        _register_optional_cuda_target_bundle(
            bin_dir,
            _xla_comm_backend_library,
            "xla_comm_permute_probe",
            {
                "prepare": "XlaCommPermuteProbePrepareFFI",
                "execute": "XlaCommPermuteProbeFFI",
            },
        )
        _register_optional_cuda_target_bundle(
            bin_dir,
            _xla_comm_backend_library,
            "xla_comm_chunk_permute_probe",
            {
                "prepare": "XlaCommChunkPermuteProbePrepareFFI",
                "execute": "XlaCommChunkPermuteProbeFFI",
            },
        )
        _register_optional_cuda_target_bundle(
            bin_dir,
            _xla_comm_backend_library,
            "xla_rect_pack_unpack_probe",
            {
                "prepare": "XlaRectPackUnpackProbePrepareFFI",
                "execute": "XlaRectPackUnpackProbeFFI",
            },
        )
        _register_optional_cuda_target_bundle(
            bin_dir,
            _xla_comm_backend_library,
            "xla_rect_transfer_probe",
            {
                "prepare": "XlaRectTransferProbePrepareFFI",
                "execute": "XlaRectTransferProbeFFI",
            },
        )
        _register_optional_cuda_target_bundle(
            bin_dir,
            _xla_comm_backend_library,
            "xla_rect_2d_native_plan",
            {
                "prepare": "XlaRect2DNativePlanPrepareFFI",
                "execute": "XlaRect2DNativePlanFFI",
            },
        )
        _register_optional_cuda_target_bundle(
            bin_dir,
            _xla_comm_backend_library,
            "xla_rect_padded_2d_native_plan",
            {
                "prepare": "XlaRectPadded2DNativePlanPrepareFFI",
                "execute": "XlaRectPadded2DNativePlanFFI",
            },
        )
        _register_optional_cuda_target_bundle(
            bin_dir,
            _xla_comm_backend_library,
            "cusolvermp_init_probe",
            {
                "prepare": "XlaCusolverMpInitProbePrepareFFI",
                "execute": "XlaCusolverMpInitProbeFFI",
            },
        )
        _register_optional_cuda_target_bundle(
            bin_dir,
            _xla_comm_backend_library,
            "cusolvermp_scatter_layout_probe",
            {
                "prepare": "XlaCusolverMpScatterLayoutProbePrepareFFI",
                "execute": "XlaCusolverMpScatterLayoutProbeFFI",
            },
        )
        _register_optional_cuda_target_bundle(
            bin_dir,
            _xla_comm_backend_library,
            "cusolvermp_potrs_probe",
            {
                "prepare": "XlaCusolverMpPotrsProbePrepareFFI",
                "execute": "XlaCusolverMpPotrsProbeFFI",
            },
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
