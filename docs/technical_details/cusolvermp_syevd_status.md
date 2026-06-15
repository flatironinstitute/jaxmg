# cuSOLVERMp SYEVD Investigation Status

This note records the current state of the cuSOLVERMp SYEVD migration after
the rank-per-GPU, native NCCL, and cuSOLVERMp 0.8.0 diagnostic work. It is a
development decision record: it should make clear what has been proved, what
has failed, and which experiment should decide the next code direction.

## Current Position

The cuSOLVERMp Cholesky solve path is in good shape for small rank-per-GPU
tests:

```text
JAX block-sharded A and B
  -> local shard padding
  -> native edge-padding and 2D block-cyclic redistribution
  -> borrowed XLA-owned ncclComm_t
  -> cusolverMpPotrf + cusolverMpPotrs
  -> reverse 2D redistribution
  -> JAX-facing output
```

The SYEVD migration should now focus only on the eigenvector-producing path:

```text
jaxmg.syevd
  -> local shard padding
  -> native edge-padding and 2D block-cyclic redistribution
  -> borrowed XLA-owned ncclComm_t
  -> cusolverMpSyevd(compz = "Z")
  -> reverse 2D redistribution of eigenvectors
  -> JAX-facing eigenvalues and eigenvectors
```

The true no-vector path is not part of the current release target. Both
cuSOLVERMp 0.7.2 and the staged cuSOLVERMp 0.8.0 runtime reject
`compz = "N"` in the current tests. JAXMg therefore does not expose a cheap
eigenvalue-only public mode.

## cuSOLVERMp Runtime Status

CSD3's installed NVIDIA HPC SDK 26.3 currently provides:

```text
/rds/user/jlt67/hpc-work/PhD/NVHPC/Linux_x86_64/26.3
libcusolverMp.so.0.7.2.0
```

A cuSOLVERMp 0.8.0 redistributable has also been staged for probing:

```text
/rds/user/jlt67/hpc-work/PhD/JAXMG/cusolvermp_0.8.0_probe/
  libcusolvermp-linux-x86_64-0.8.0.3126_cuda12-archive/
```

The 0.8.0 release notes matter for this work because they mention:

1. `cusolverMpSyevd()` performance improvements;
2. `cusolverMpSetStream()`;
3. a fix for a previous restriction around passing the default CUDA stream to
   `cusolverMpCreate()`;
4. a fix for `potrf`/`potrs` on non-square process grids in versions up to and
   including 0.7.2.

Reference:
`https://docs.nvidia.com/cuda/cusolvermp/release_notes/index.html#cusolvermp-v0-8-0`

## Important API Detail: `compz`

The C API exposed by the installed headers accepts a `compz` argument:

```c
cusolverMpSyevd(... char* compz, ..., cudaDataType_t computeType, ...)
```

NVIDIA's current `mp_syevd.c` sample uses:

```c
char compz = 'Z';
```

for the eigenvector-producing path. Earlier JAXMg experiments used
`compz = 'V'`, which matches the public-doc language around `jobz = 'V'` but
does not match the sample for this C API. The branch now uses `compz = 'Z'`
when eigenvectors are requested.

Eigenvalue-only mode is not supported in the tested runtimes. Passing
`compz = 'N'` reaches the library but the runtime reports:

```text
SYEVD does not support eigenvalue only, compz=N
```

JAXMg should not emulate this by computing vectors and discarding them unless
that behavior is explicitly requested in a future API as an expensive fallback.

## Test Record

| Job | What it tested | Result | Conclusion |
| --- | --- | --- | --- |
| `30574045` | Standalone NVIDIA-style raw `mp_syevd.c` binary launched outside JAX/JAXMg with the site 0.7.2 runtime. | `COMPLETED`, 21 s. | CSD3 can run SYEVD in the standalone sample mode. This used MPI/Hydra and cuSOLVERMp's own communicator path, not the XLA-owned communicator. |
| `30580251` | Rank-per-GPU production `potrs` matrix tests before moving on to SYEVD. | Slurm job failed later, but all `potrs` cases reported status code `0`. | The Cholesky path works for 4-rank single-node cases including `2x2`, `1x4`, `4x1`, all supported dtypes, padding, and repeated calls. |
| `30572566` | Small SYEVD sample-layout probe using cuSOLVERMp scatter input, before the `Z` correction. | Failed with cuSOLVER status `7`. | The failure was not caused by JAXMg's 2D redistribution alone, because this probe used cuSOLVERMp's own scatter layout. |
| `30572754` | SYEVD sample-layout probe comparing XLA stream and private nonblocking CUDA stream, before the `Z` correction. | `compz=N` failed; `compz=V` vector case hung/cancelled. | A private stream alone did not fix the old `V`/`N` behavior. |
| `30580388` | Rank-per-GPU 4-rank SYEVD sample-layout probe with allocation-visible launch, before the `Z` correction. | `compz=N` failed; `compz=V` failed inside `mp_stedc` with NCCL/CUDA errors. | The corrected rank-per-GPU launch is not enough if the old `V` flag is used. Failure occurs even without production redistribution. |
| `30580702` | Rebuild after changing SYEVD vector mode to `compz='Z'`. | `COMPLETED`, 5 min 47 s, commit `8cd7626`. | The installed native backend contains the `Z` correction. |
| `30582053` | Single-process cuSOLVERMp 0.8.0 version sanity probe. | `COMPLETED`; `cusolverMpGetVersion` returned `800`. | The staged 0.8.0 library can be loaded and used by a basic non-MPI process on CSD3. |
| `30582082` | Tiny 2-rank MPI + cuSOLVERMp 0.8.0 version probe. | Failed with `SIGILL` before useful rank output. | The standalone MPI launcher path has an environment/runtime problem with the staged 0.8.0 payload. This does not invalidate the JAX/XLA-rank path. |
| `30582194` | 2-GPU rank-per-GPU JAXMg sample-layout SYEVD probe with cuSOLVERMp 0.8.0. | Slurm job marked failed only because values-only subtests failed. Vector subtests passed on both XLA and private streams. | `compz='Z'` vector SYEVD works through the JAX/XLA borrowed communicator path with cuSOLVERMp 0.8.0 in the sample-layout probe. `compz='N'` remains unsupported. |
| `30582168` | 4-GPU cuSOLVERMp 0.8.0 JAXMg probe. | Stuck in Slurm configuring/running on one node with no useful output at last check. | Treat as a scheduler/node result, not solver evidence. |

## Interpretation

The strongest positive SYEVD evidence is job `30582194`. It proves that, with
cuSOLVERMp 0.8.0:

1. JAXMg can load the staged cuSOLVERMp library;
2. the borrowed XLA-owned communicator can be consumed by cuSOLVERMp SYEVD for
   the eigenvector-producing path;
3. the vector path works on both the XLA FFI stream and a private nonblocking
   CUDA stream in the sample-layout diagnostic;
4. the old `compz='V'` path was not the correct interface for this C API; and
5. `compz='N'` should remain disabled.

This does not yet prove the production `syevd` path. The passing diagnostic
uses NVIDIA's host scatter/gather style sample layout, not JAXMg's native
GPU-to-GPU 2D redistribution output. The next production checkpoint is
therefore:

```text
ordinary JAX-sharded matrix
  -> JAXMg native 2D redistribution
  -> cusolverMpSyevd(compz = "Z", cuSOLVERMp 0.8.0)
  -> reverse JAXMg redistribution of eigenvectors
  -> eigenvalue/eigenvector residual check
```

If that production vector test fails while the sample-layout vector diagnostic
passes, the likely issue is in descriptor wiring, redistributed local layout,
workspace sizing, or reverse redistribution, not in the basic borrowed
communicator concept.

## Current Release Guidance

The clean user-facing position is:

```text
potrs: supported on the validated rank-per-GPU path.
syevd: active target; sample-layout probe passes with cuSOLVERMp 0.8.0,
       production redistributed-buffer validation still required.
explicit inverse: unsupported in the cuSOLVERMp backend.
```

The immediate engineering target is not to debug eigenvalue-only SYEVD. It is
to make the vector-producing production wrapper pass with the staged
cuSOLVERMp 0.8.0 runtime.
