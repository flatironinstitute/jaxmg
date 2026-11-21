/* Linker version script for libjaxmg_xla_comm_backend.so.
 *
 * Export only the XLA FFI handlers that src/jaxmg/_setup.py resolves through
 * ctypes. Everything else stays local: the toolchain links libc++, libc++abi
 * and libunwind statically, and if those symbols were exported, the
 * libstdc++ that cuSOLVERMp brings in with RTLD_GLOBAL would take over our
 * own calls into the C++ runtime (e.g. __dynamic_cast).
 */
{
  global:
    XlaCusolverMp*;
  local:
    *;
};
