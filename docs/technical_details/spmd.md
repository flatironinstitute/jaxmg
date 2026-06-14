# Single Process Multiple Devices (SPMD)

This migration branch supports the single-node SPMD execution path. A public
solver call builds a `jax.shard_map` over the local device mesh, so each GPU
invocation receives its local JAX-owned shard as an FFI buffer.

The native backend is `libjaxmg_xla_comm_backend.so`, built with Bazel
against the pinned OpenXLA source for the selected JAX/JAXLIB version. During
the FFI prepare phase, the handler requests a node-scoped collective clique.
In ordinary SPMD execution this is the same as the local-device clique. During
execution, it looks up the XLA-owned GPU communicator for the current rank and
uses that communicator for GPU-to-GPU column movement.

The production single-node flow is:

1. JAX pads the row-sharded matrix if the per-device shard is not tile-aligned.
2. The fused native handler receives the donated JAX buffers.
3. The handler computes the 1D block-cyclic column permutation natively.
4. The XLA communicator executes the permutation cycles directly between GPU
   buffers.
5. The CUDA stream is synchronized before the cuSOLVERMg call, because
   cuSOLVERMg has no stream setter equivalent to cuSolverDN.
6. The designated host invocation calls cuSOLVERMg with the per-device matrix
   pointers.
7. Any solver outputs that must be replicated are broadcast with the same XLA
   communicator.

The old shared-memory and `cudaMemcpyPeerAsync` shuffler is intentionally not
part of the SPMD path. Multi-process execution is being reintroduced through
the XLA communicator probes first; fused cuSOLVERMg solver support still needs
the CUDA IPC pointer-sharing layer described in `mpmd.md`.

The native source is split by responsibility under `src/cuda`: shared XLA and
cuSolverMg declarations live in `include/xla_comm_backend.h`, common helper
code in `utils/xla_comm_common.cc`, the 1D redistribution in `cyclic_1d.cc`,
solver handlers in `potrs.cc` and `syevd.cc`, diagnostic
communicator probes in `collective_diagnostics.cc`, and exported FFI bindings in
`ffi_handlers.cc`.
