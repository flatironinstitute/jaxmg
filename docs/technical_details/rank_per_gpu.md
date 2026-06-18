# Rank-Per-GPU Execution

The cuSOLVERMp backend requires one Python process per participating GPU.  This
matches cuSOLVERMp's distributed execution model: every GPU is represented by a
process/rank, and the native JAXMg backend borrows the NCCL communicator that
XLA has already created for that distributed JAX program.

JAXMg does not ask the user to declare an SPMD or MPMD mode.  Instead, during
backend initialization it inspects the JAX runtime:

- if the current process owns exactly one local GPU, the backend can be used;
- if the current process owns more than one local GPU, initialization raises an
  error before any native FFI target is registered.

For multi-GPU and multi-node runs, users should initialize distributed JAX in
the normal way before constructing arrays or calling JAXMg routines.  A typical
rank-per-GPU launch gives each process one local device, for example by using
``jax.distributed.initialize(..., local_device_ids=[local_rank])`` or an
equivalent launcher setup.

The public JAXMg routines then operate on ordinary JAX-sharded arrays.  The
Python wrapper validates the mesh and sharding, applies local tile padding when
needed, and enters one fused native FFI call.  Native code performs the local
layout conversion, 2D redistribution, cuSOLVERMp call, and reverse
redistribution.
