# cuSOLVERMp Routine Coverage

This note records which cuSOLVERMp routines matter for migrating the current
JAXMg API. It is intentionally separate from the redistribution notes: a correct
2D block-cyclic layout is necessary, but the solver layer also has its own
routine coverage, descriptor, stream, workspace, and communicator constraints.

## Required cuSOLVERMp Objects

The cuSOLVERMp solver path needs these native objects before any factorization
or solve can run:

```text
cusolverMpHandle_t
cusolverMpGrid_t
cusolverMpMatrixDescriptor_t
```

The handle is tied to one CUDA device and stream. The grid is created from an
already-initialized `ncclComm_t` plus the process-grid dimensions. This is the
reason the current XLA-communicator work matters for cuSOLVERMp: the future
solver backend must either borrow the XLA-owned NCCL communicator safely or
create a compatible communicator that does not conflict with XLA's collective
ordering.

The matrix descriptor carries the global matrix shape, tile shape
`MB_A x NB_A`, source process row/column, and local leading dimension. Current
cuSOLVERMp documentation only supports `RSRC_A = 0` and `CSRC_A = 0`, which
matches the layout the current 2D redistribution prototype targets.

## Current JAXMg API Mapping

| JAXMg API | Current backend | cuSOLVERMp status | Migration note |
| --- | --- | --- | --- |
| `potrs` | `cusolverMgPotrf` + `cusolverMgPotrs` | Directly supported by `cusolverMpPotrf` + `cusolverMpPotrs` | First real solver target. Requires descriptors for `A` and `B`, square `MB_A == NB_A`, and aligned `A`/`B` row blocking. |
| `syevd` | `cusolverMgSyevd` | Directly supported by `cusolverMpSyevd` | Good second target after `potrs`. `jobz = "V"` maps to eigenvalues plus eigenvectors. |
| `syevd_no_V` | `cusolverMgSyevd` with no eigenvectors | Directly supported by `cusolverMpSyevd` | `jobz = "N"` maps to eigenvalues only. |
| `potri` | `cusolverMgPotrf` + `cusolverMgPotri` | No direct `cusolverMpPotri` entry in the current C API documentation | Not supported by the first cuSOLVERMp backend. It should remain out of scope until there is a separate design for inverse support. |

The practical migration order should therefore be:

1. `potrs`, because it tests the complete factor/solve path and matches the
   user's main Cholesky-solve workflow.
2. `syevd_no_V` and `syevd`, because `cusolverMpSyevd` exists and maps onto the
   current API shape.
3. Do not migrate `potri` in the first cuSOLVERMp backend. It should fail
   clearly, or remain on a separate legacy path, rather than silently emulating
   an inverse with a large distributed identity solve.

## Other cuSOLVERMp Routines Worth Knowing

The current cuSOLVERMp dense API also includes:

```text
Getrf / Getrs
Geqrf / Ormqr / Orgqr / Gels
Sytrd / Stedc / Ormtr / Sygst / Sygvd
Laset
NewtonSchulz
```

These are useful context but are not immediate replacements for the current
JAXMg public API. `Laset` may become useful for future utilities, but it should
not be used to hide a `potri` emulation path in the first cuSOLVERMp backend.
`Getrf/Getrs` could support a future general linear solve API, but that is not
part of the current JAXMg interface. `NewtonSchulz` is also not a general
replacement for `potri`: the documented routine is an orthogonalization/polar
factor iteration with its own process-grid and data-type limitations.

## Constraints to Probe Next

Before adding production cuSOLVERMp solver calls, the next native diagnostic
should prove the following on a tiny single-node case:

1. The borrowed XLA communicator handle is a valid NCCL communicator for
   `cusolverMpCreateDeviceGrid`.
2. `cusolverMpCreate`, `cusolverMpCreateDeviceGrid`, and
   `cusolverMpCreateMatrixDesc` work on the XLA callback stream.
3. `cusolverMpPotrf_bufferSize` and `cusolverMpPotrs_bufferSize` return sensible
   host/device workspace sizes for the redistributed local buffers.
4. `cusolverMpPotrf` followed by `cusolverMpPotrs` succeeds for a very small
   matrix whose layout has already been checked against
   `cusolverMpMatrixScatterH2D`.
5. Repeated calls do not leak handles, descriptors, grids, host workspaces, or
   device workspaces.

Only after this diagnostic passes should the branch wire cuSOLVERMp into the
public `jaxmg.potrs` path.

The next diagnostic is `cusolvermp_potrs_probe`. It still does not use JAXMg's
GPU-to-GPU redistribution output. Instead it isolates the cuSOLVERMp solver
boundary:

1. borrow XLA's NCCL communicator from the FFI collective context;
2. create a cuSOLVERMp handle, row-major process grid, and descriptors;
3. scatter a tiny deterministic diagonal SPD matrix and one RHS from rank 0
   with `cusolverMpMatrixScatterH2D`;
4. query `cusolverMpPotrf_bufferSize` and `cusolverMpPotrs_bufferSize`;
5. run `cusolverMpPotrf` followed by `cusolverMpPotrs`;
6. gather the solution on rank 0 with `cusolverMpMatrixGatherD2H` and check the
   known diagonal solve residual.

Passing this probe means cuSOLVERMp can consume the borrowed communicator for a
real Cholesky solve. It does not yet prove the production path, because the
input layout still comes from NVIDIA's host scatter helper instead of JAXMg's
native GPU-to-GPU 2D redistribution.

## CSD3 cuSOLVERMp SDK Environment

CSD3's standard CUDA 12.1/cuDNN module stack provides cuSolverMg but does not
provide the current cuSOLVERMp shared library. The single-node cuSOLVERMp
diagnostics therefore use the NVIDIA HPC SDK 26.3 installation under:

```text
/rds/user/jlt67/hpc-work/PhD/NVHPC/Linux_x86_64/26.3
```

The durable paths used by the verification jobs are:

```text
NVHPC_ROOT=/rds/user/jlt67/hpc-work/PhD/NVHPC/Linux_x86_64/26.3
CUSOLVERMP_INC=${NVHPC_ROOT}/math_libs/12.9/targets/x86_64-linux/include
CUSOLVERMP_LIB=${NVHPC_ROOT}/math_libs/12.9/targets/x86_64-linux/lib
NVHPC_NCCL_LIB=${NVHPC_ROOT}/comm_libs/12.9/nccl/lib
```

The loader path should put the JAX wheel's NCCL library before the HPC SDK NCCL
library when the diagnostic borrows XLA's communicator:

```text
${CONDA_ENV}/lib/python3.11/site-packages/nvidia/nccl/lib
```

This keeps `libcusolverMp.so` and the XLA/JAX CUDA plugin using a compatible
NCCL runtime in the same process. The cuSOLVERMp math libraries still come from
the SDK path. This is a CSD3 test environment detail; the package code itself
continues to load cuSOLVERMp dynamically from the user's runtime library path.

## References

- NVIDIA cuSOLVERMp initialization: `https://docs.nvidia.com/cuda/cusolvermp/usage/initialization/index.html`
- NVIDIA cuSOLVERMp C API: `https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html`
- NVIDIA cuSOLVERMp data types and grid mapping: `https://docs.nvidia.com/cuda/cusolvermp/usage/types.html`
