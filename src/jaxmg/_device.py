"""Device capability queries via the CUDA driver (libcuda).

These helpers query the CUDA driver API directly with ``ctypes`` and do not
initialize JAX or load the jaxmg native backend. They are therefore safe to call
*before* JAX initializes its GPU allocator — e.g. to decide
``XLA_PYTHON_CLIENT_ALLOCATOR``.
"""

import ctypes

# enum CUdevice_attribute 102 corresponds to
# CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED. 
_CU_DEVICE_ATTRIBUTE_VMM_SUPPORTED = 102

_libcuda = None


def _load_libcuda():
    """Return a cached handle to the CUDA driver library."""
    global _libcuda
    if _libcuda is None:
        try:
            _libcuda = ctypes.CDLL("libcuda.so.1")
        except OSError as e:
            raise RuntimeError(
                "CUDA driver library (libcuda.so.1) is not available; "
                "cannot query CUDA device capabilities."
            ) from e
    return _libcuda


def device_supports_vmm(device: int = 0) -> bool:
    """Return True if the CUDA device supports Virtual Memory Management.

    VMM support is required by JAX's ``vmm`` GPU allocator. ``device`` is a CUDA
    device ordinal (default 0). Queries the CUDA driver directly, independent of
    JAX, so it is safe to call before JAX GPU initialization.

    Raises:
        RuntimeError: if the CUDA driver is unavailable or a driver query fails.
    """
    lib = _load_libcuda()
    if lib.cuInit(0) != 0:
        raise RuntimeError("cuInit failed; no usable CUDA driver or GPU present.")
    dev = ctypes.c_int()
    if lib.cuDeviceGet(ctypes.byref(dev), int(device)) != 0:
        raise RuntimeError(f"cuDeviceGet failed for device ordinal {device}.")
    value = ctypes.c_int()
    if (
        lib.cuDeviceGetAttribute(
            ctypes.byref(value), _CU_DEVICE_ATTRIBUTE_VMM_SUPPORTED, dev
        )
        != 0
    ):
        raise RuntimeError(
            f"cuDeviceGetAttribute failed querying VMM support for device {device}."
        )
    return bool(value.value)
