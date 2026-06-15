# cuSOLVERMp Production Migration Plan

This document records the target production shape for the cuSOLVERMp migration.
It is intentionally stricter than the current development branch: diagnostic
probes, removed single-node solver paths, and Python-orchestrated redistribution stages are
useful while developing, but should not remain in the final release backend.

## End State

The production backend should be cuSOLVERMp-only:

```text
JAX block-sharded input
  -> JAX-side local padding, if needed
  -> one fused native FFI call
       -> native 2D redistribution to cuSOLVERMp block-cyclic layout
       -> cuSOLVERMp routine
       -> native reverse redistribution of outputs
  -> JAX-side unpadding
  -> JAX-facing output
```

Python still owns padding and unpadding because padding changes the JAX-visible
array shape and sharding contract. Native code owns the internal redistribution,
communication, cuSOLVERMp object creation, solver call, and temporary workspace.

## Supported Public Routines

The final public API should keep the original JAXMg function names:

```python
jaxmg.potrs(...)
jaxmg.syevd(...)
```

`potrs` should run the Cholesky factor/solve sequence:

```text
potrf
potrs
```

`syevd` should compute eigenvalues and eigenvectors. The eigvals-only path is
not part of the production migration:

- remove all no-vector eigensolver Python, C++, FFI, docs, and build references;
- remove any public input that selects eigenvalue-only SYEVD;
- do not leave a public `NotImplementedError` branch for this mode.

Explicit inverse support should also be removed from the production
surface unless a future cuSOLVERMp-backed implementation is intentionally
designed and tested.

## Removed Backend Surface

The production branch should remove the old single-node solver backend and
the 1D cyclic redistribution backend:

- remove old solver FFI targets and Python wrappers;
- remove 1D cyclic redistribution code;
- remove CUDA peer/shared-memory transfer machinery;
- remove public API/docs for routines that no longer exist;
- remove old `*_mp` public wrappers once their behavior is folded into the
  original `potrs` and `syevd` entry points.

The old branch is useful as a structural reference, especially
because it used one fused FFI handler per solver. It should not remain as a
second production backend in this migration.

## Python Responsibilities

The Python layer should be thin and close in style to the original JAXMg
functions. It should:

1. validate ranks, shapes, dtypes, tile size, and mesh/spec compatibility;
2. infer the JAX mesh and `PartitionSpec` from the input sharding when possible;
3. validate that the mesh device order is representable by cuSOLVERMp's
   row-major or column-major grid mapping;
4. pad local shards with JAX when local shapes are not tile-aligned;
5. construct output shape/dtype declarations for a single fused FFI call;
6. call the fused native target under `jax.jit`/`jax.shard_map`; and
7. unpad JAX-facing outputs.

The Python layer should not:

- execute redistribution phases one call at a time;
- pass Python-built tile movement schedules into production FFI calls;
- expose redistribution scratch as a user-visible JAX array;
- own distributed initialization.

Users remain responsible for calling `jax.distributed.initialize(...)` in
multi-node programs before JAX initializes the GPU backend. JAXMg validates the
resulting runtime and mesh.

## Native Production FFI Targets

The target production FFI calls are:

```text
cusolvermp_potrs
cusolvermp_syevd
```

`cusolvermp_potrs` should:

1. receive padded JAX block-sharded `A` and `B` buffers;
2. allocate internal native scratch for redistribution;
3. redistribute `A` into cuSOLVERMp 2D block-cyclic layout;
4. redistribute `B` into the matching 2D block-cyclic layout;
5. create cuSOLVERMp handle, grid, descriptors, and workspaces;
6. call `potrf` and `potrs`;
7. reverse-redistribute solved `B`;
8. return padded solved `B` and a status buffer.

`cusolvermp_syevd` should:

1. receive a padded JAX block-sharded `A` buffer;
2. allocate internal native scratch for redistribution;
3. redistribute `A` into cuSOLVERMp 2D block-cyclic layout;
4. create cuSOLVERMp handle, grid, descriptors, and workspaces;
5. call `syevd` in the eigenvector-producing mode;
6. reverse-redistribute eigenvectors into the JAX-facing padded layout;
7. return eigenvalues, padded eigenvectors, and a status buffer.

The native FFI handlers should borrow the XLA-owned NCCL communicator only for
the duration of the FFI call.

## Internal Scratch Allocation

Redistribution scratch and solver workspace should be allocated inside the FFI
handler using the XLA FFI scratch allocator, following the original fused JAXMg
style. The user should not see or pass a scratch array.

Native scratch includes:

- saved-cycle payload storage;
- send staging storage;
- receive staging storage;
- cuSOLVERMp workspace;
- temporary local buffers needed by the fused routine.

The native handler should calculate and validate the required scratch size from:

```text
logical shape
padded local shape
tile size
process grid
dtype
routine
```

The current development path exposes scratch through JAX because it made the
redistribution stages easier to build and test. Production should remove that
visible scratch and allocate it internally.

## Native Code Organisation

The production native code is grouped by ownership:

```text
src/cuda/
  ffi_handlers.cc
  include/
    xla_comm_backend.h

  cusolvermp_routines/
    cusolvermp.cc
    cusolvermp_potrs.cc
    cusolvermp_syevd.cc

  memory_redist/
    block_cyclic_2d.cc
    edge_padding_2d.cc
    rectangle_pack.cc

  diagnostics/
    collective_diagnostics.cc

  utils/
    xla_comm_common.cc
```

`ffi_handlers.cc` remains top-level because it is the one exported FFI
registration surface. The solver-specific files in `cusolvermp_routines/`
still compile into one shared library and still enter one native FFI call per
public solver.

## Comment and Documentation Style

Every C++/CUDA file under `src/cuda` should have a high-level file comment near
the top, matching the style already introduced in the branch. That comment
should explain:

- what subsystem the file belongs to;
- whether it is production or diagnostic code;
- the main workflow implemented by the file;
- the ownership/lifetime assumptions for buffers, streams, and communicators;
- the role of the file in the fused call path.

Within each C++/CUDA file, comments should explain the major stages of the
algorithm, especially:

- communicator lookup and validation;
- rank/process-grid mapping;
- local padding assumptions;
- redistribution planning;
- pack/send/recv/unpack staging;
- cuSOLVERMp descriptor setup;
- solver workspace allocation;
- synchronization and stream-ordering assumptions;
- status/error propagation.

Python files under `src/jaxmg` should stay close in structure and tone to the
original Flatiron JAXMg code. Prefer preserving old function names, signatures,
and unchanged code where possible. When wording has to change because the
backend changed, keep the same concise, practical docstring style rather than
turning the public API files into design documents.

Use `git mv` and focused edits where possible so history remains readable and
`git blame` still maps unchanged code back to the original project.

## Launch Modes

The rank-per-GPU launch mode has passed large 8-GPU, 2-node validation for both
redistribution and solver status in the current development tests.

The one-process-per-node launch mode must still be validated. If it passes, the
production code should support both:

```text
one Python process per GPU
one Python process per node with multiple local GPUs
```

If one-process-per-node exposes stream, communicator, or cuSOLVERMp behavior
that differs from rank-per-GPU, production should either add explicit native
handling for that mode or reject it with a clear error. The supported contract
must be documented before release.

## Production Guards

The production backend should include these guards:

- require or clearly validate cuSOLVERMp `>= 0.8.0`;
- reject unsupported dtypes/routines early;
- reject unsupported process-grid mappings early;
- add a guard or warning for very small local dimensions in `syevd`, because
  tiny 8-rank cases failed while larger `n=512/640` cases passed;
- validate `MB_A == NB_A == T_A` until non-square tile support is implemented;
- return or raise clear decoded cuSOLVERMp/XLA/NCCL status information.

## Suggested Implementation Order

1. Validate the one-process-per-node diagnostic job and decide the launch-mode
   contract.
2. Add fused `potrs` FFI with internal scratch allocation while keeping the
   existing staged path for comparison.
3. Switch Python `potrs` to the fused target and remove the staged scratch
   argument from the public production path.
4. Add fused `syevd` FFI with eigenvectors only.
5. Switch Python `syevd` to the fused target.
6. Fold `_potrs.py` and `_syevd.py` behavior into `_potrs.py` and
   `_syevd.py`.
7. Remove old solver paths, 1D cyclic code, no-vector eigensolver code, explicit
   inverse code, and unused diagnostic
   production registrations.
8. Reorganize `src/cuda` into production subdirectories.
9. Run the full correctness suite in supported launch modes.
10. Update user documentation and packaging notes.
