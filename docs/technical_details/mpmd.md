# Multiple Process Multiple Devices (MPMD)

The original JAXMg package had separate `_mp` CUDA libraries for multi-process
execution. In that mode each Python process owns one GPU, so ordinary
process-local C++ state cannot be used to share cuSOLVERMg pointer tables or
host barriers.

The XLA communicator migration reintroduces MPMD in stages. The first stage is
communicator-level support: when `jax.distributed` is initialized, JAXMg detects
MPMD mode, reads `JAXMG_NUMBER_OF_DEVICES`, and requests a node-scoped XLA
collective clique. The node scope keeps cuSOLVERMg single-node while still
allowing the communicator probes and redistribution handlers to use XLA/NCCL
between participating ranks.

This solves the communication side of the old MPMD design:

1. build the participating rank group from XLA's global device map;
2. restrict the group to one node-sized chunk;
3. borrow the XLA-owned communicator during FFI execution;
4. use that communicator for all-reduce and collective-permute probes;
5. use the same clique shape for the 1D cyclic reshuffle.

It does not by itself solve cuSOLVERMg pointer ownership. cuSOLVERMg still
expects one host invocation to pass arrays of device pointers:

```cpp
void* a_ptrs[num_devices];
void* work_ptrs[num_devices];
```

In SPMD those pointers are process-local. In MPMD they belong to different
processes, so the solver layer still needs a cleaned-up replacement for the old
CUDA IPC handle exchange:

1. each rank exports its JAX/XLA buffer base pointer and byte offset;
2. the node-local coordinator opens peer handles with CUDA IPC;
3. the coordinator builds the cuSOLVERMg pointer tables;
4. all ranks synchronize workspace allocation and solver completion;
5. all opened IPC handles are closed before returning to JAX.

The intended MPMD checkpoint order is therefore:

1. validate XLA communicator probes in a `jax.distributed` one-process-per-GPU
   launch;
2. validate the XLA communicator 1D cyclic reshuffle in the same launch mode;
3. add the CUDA IPC pointer-sharing layer for `potrs`;
4. extend the pointer-sharing layer to `potri` and `syevd`;
5. only then consider this backend a replacement for the original `_mp`
   libraries.

This remains distinct from the future cuSOLVERMp migration. cuSOLVERMp should
eventually own the true multi-node 2D block-cyclic solver path; MPMD cuSOLVERMg
is still a single-node compatibility layer.
