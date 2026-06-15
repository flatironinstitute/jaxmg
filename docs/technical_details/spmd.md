# Single-Process Launches

Single-process execution is the local version of the same fused cuSOLVERMp
pipeline. A public solver call builds a `jax.shard_map` over the local mesh, so
each GPU invocation receives its local JAX-owned shard as an FFI buffer.

The native backend is `libjaxmg_xla_comm_backend.so`, built with Bazel against
the pinned OpenXLA source for the selected JAX/JAXLIB version. During FFI
prepare, the handler requests the XLA collective clique spanning the process
grid. During execution, it borrows the XLA-owned GPU communicator for the
duration of the call.

The production single-process flow is:

1. JAX pads local shards if the per-device capacity is not tile-aligned.
2. The fused native handler receives donated JAX buffers.
3. Native code compacts edge padding and redistributes into cuSOLVERMp's 2D
   block-cyclic layout.
4. Native code creates cuSOLVERMp handles, grid, descriptors, and workspaces.
5. The cuSOLVERMp routine runs on the redistributed buffers.
6. Native code reverse-redistributes outputs back to the JAX-facing layout.
7. Python removes local padding from the returned arrays.

The native source is split by responsibility under `src/cuda`: shared XLA and
cuSOLVERMp declarations live in `include/xla_comm_backend.h`, common helper
code in `utils/xla_comm_common.cc`, 2D redistribution in
`edge_padding_2d.cc`, `block_cyclic_2d.cc`, and `rectangle_pack.cc`,
cuSOLVERMp solver integration in `cusolvermp.cc`, diagnostic communicator
probes in `collective_diagnostics.cc`, and exported FFI bindings in
`ffi_handlers.cc`.
