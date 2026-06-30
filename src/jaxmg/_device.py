"""Device capability queries backed by the native JAXMg backend."""

import ctypes

from ._setup import get_backend_library


def device_supports_vmm(device: int = 0) -> bool:
    """Return True if the given CUDA device supports Virtual Memory Management.

    VMM support is required by JAX's platform/async GPU allocator. ``device`` is
    a CUDA device ordinal (default 0, the single GPU owned by this process).

    Raises:
        RuntimeError: if the native backend is not loaded (no GPU detected) or
            the underlying ``cuDeviceGetAttribute`` driver query fails.
    """
    lib = get_backend_library()
    if lib is None:
        raise RuntimeError(
            "JAXMg native backend is not loaded (no GPU detected); "
            "cannot query CUDA VMM support."
        )
    fn = lib.jaxmg_device_supports_vmm
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_int]
    result = fn(int(device))
    if result < 0:
        raise RuntimeError(
            f"cuDeviceGetAttribute failed querying VMM support for device {device}."
        )
    return bool(result)
