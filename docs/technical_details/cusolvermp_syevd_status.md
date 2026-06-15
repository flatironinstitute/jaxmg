# cuSOLVERMp SYEVD Investigation Status

This note records the current state of the cuSOLVERMp migration after the
rank-per-GPU, native NCCL, and SYEVD diagnostic work. It is a development
decision record: it should make clear what has been proved, what has failed,
and which experiment should decide the next code direction.

## Current Position

The cuSOLVERMp Cholesky solve path is in good shape for small single-node
rank-per-GPU tests:

```text
JAX block-sharded A and B
  -> local shard padding
  -> native edge-padding and 2D block-cyclic redistribution
  -> borrowed XLA-owned ncclComm_t
  -> cusolverMpPotrf + cusolverMpPotrs
  -> reverse 2D redistribution
  -> JAX-facing output
```

The SYEVD path is not yet in the same state. The standalone NVIDIA
cuSOLVERMp sample works on CSD3, but the JAXMg FFI path has not yet been shown
to run `cusolverMpSyevd` successfully. The current active question is whether
the failure is caused by:

1. the old `compz` value used by JAXMg;
2. the XLA/JAX stream used inside FFI;
3. the borrowed XLA-owned NCCL communicator;
4. the way the JAXMg production path wires descriptors and redistributed
   buffers; or
5. a cuSOLVERMp 0.7.2 runtime limitation/bug.

## Installed cuSOLVERMp Runtime

CSD3 testing currently uses NVIDIA HPC SDK 26.3:

```text
/rds/user/jlt67/hpc-work/PhD/NVHPC/Linux_x86_64/26.3
```

The installed cuSOLVERMp library is:

```text
libcusolverMp.so.0.7.2.0
```

The installed headers define:

```c
#define CUSOLVERMP_VER_MAJOR 0
#define CUSOLVERMP_VER_MINOR 7
#define CUSOLVERMP_VER_PATCH 2
#define CUSOLVERMP_VER_BUILD 0
#define CUSOLVERMP_VERSION 702
```

The v0.8.0 release notes are relevant because they mention fixes in areas that
overlap our symptoms:

1. `cusolverMpSyevd()` performance was improved.
2. `cusolverMpSetStream()` was added.
3. the previous restriction on passing the default `NULL`/`0` CUDA stream to
   `cusolverMpCreate()` was fixed;
4. a `potrf`/`potrs` issue on non-square process grids was fixed for versions
   up to and including 0.7.2.

These notes do not prove that the current SYEVD issue is a cuSOLVERMp bug, but
they are enough reason to prefer testing against 0.8.0 before treating a
0.7.2-only failure as a JAXMg design failure.

Reference:
`https://docs.nvidia.com/cuda/cusolvermp/release_notes/index.html#cusolvermp-v0-8-0`

## Important API Detail: `compz`

The installed CSD3 header exposes the generic SYEVD API:

```c
cusolverMpSyevd(... char* compz, ..., cudaDataType_t computeType, ...)
```

The current NVIDIA `mp_syevd.c` sample uses:

```c
char compz = 'Z';
```

for the eigenvector path. Earlier JAXMg code used `compz = 'V'`, which matches
the public-doc language around `jobz = 'V'` but not the sample path for this
installed API. The branch now uses `compz = 'Z'` when eigenvectors are
requested.

Eigenvalue-only mode is also not settled. Passing `compz = 'N'` reaches the
library but the 0.7.2 runtime reports:

```text
SYEVD does not support eigenvalue only, compz=N
```

Until this changes, `syevd_mp(eigvecs=False)` should either be disabled for the
cuSOLVERMp backend or implemented as an explicitly documented expensive
fallback that computes eigenvectors and discards them. It should not silently
pretend to be a cheap no-vector path.

## Test Record

The table below records the recent CSD3 jobs that matter for the current
decision.

| Job | What it tested | Result | Conclusion |
| --- | --- | --- | --- |
| `30574045` | Standalone NVIDIA-style raw `mp_syevd.c` binary launched outside JAX/JAXMg. | `COMPLETED`, 21 s. | CSD3's cuSOLVERMp installation can run SYEVD in its normal sample launch mode. This used MPI/Hydra and cuSOLVERMp's own communicator path, not the XLA-owned communicator. |
| `30580251` | Rank-per-GPU production `potrs_mp` matrix tests before moving on to SYEVD. | Slurm job failed later, but all `potrs_mp` cases reported status code `0`. | The Cholesky path works for 4-rank single-node cases including `2x2`, `1x4`, `4x1`, `float32`, `float64`, `complex64`, `complex128`, padding, and repeated calls. |
| `30572566` | Small SYEVD sample-layout probe using cuSOLVERMp scatter input, before the `Z` correction. | Failed with cuSOLVER status `7`. | The failure is not caused by JAXMg's 2D redistribution alone, because this probe used cuSOLVERMp's own scatter layout. |
| `30572754` | SYEVD sample-layout probe comparing XLA stream and private nonblocking CUDA stream, before the `Z` correction. | `compz=N` failed; `compz=V` vector case hung/cancelled. | A private stream alone did not fix the old `V`/`N` behavior. This does not yet decide the new `Z` path. |
| `30580388` | Rank-per-GPU 4-rank SYEVD sample-layout probe with allocation-visible launch, before the `Z` correction. | `compz=N` failed; `compz=V` failed inside `mp_stedc` with NCCL/CUDA errors. | The corrected rank-per-GPU launch is not enough if the old `V` flag is used. Failure occurs even without production redistribution. |
| `30580702` | Rebuild after changing SYEVD vector mode to `compz='Z'`. | `COMPLETED`, 5 min 47 s, commit `8cd7626`. | The installed native backend now contains the `Z` correction. |
| `30581230` | Active focused diagnostic: sample-layout SYEVD on XLA stream and private stream after the `Z` rebuild. | Pending at time of writing. | This is the current decision point. The vector cases decide whether the `Z` correction is enough. |

## Interpretation So Far

The strongest positive result is the production `potrs_mp` path. It proves that
these pieces work together for Cholesky solves:

1. rank-per-GPU JAX distributed launch;
2. borrowed XLA-owned NCCL communicator;
3. native NCCL send/recv redistribution;
4. cuSOLVERMp handle, grid, and descriptors;
5. forward and reverse JAXMg redistribution;
6. repeated calls without obvious immediate lifetime failure.

The strongest negative result is the SYEVD sample-layout probe. Because that
probe uses cuSOLVERMp's own `MatrixScatterH2D` style layout creation, it
removes JAXMg's 2D redistribution as the first suspect. The remaining suspects
are the stream/communicator/runtime/SYEVD interaction.

The private-stream result is informative but not decisive. It failed before the
`Z` correction, so it only proves:

```text
private stream + compz='V' did not work
```

It does not prove:

```text
private stream + compz='Z' cannot work
```

## Decision Tree After Job 30581230

The next direction should be chosen from the rebuilt `compz='Z'` diagnostic.

### Case A: sample vectors pass on both XLA and private streams

Conclusion: the old `V` flag was the main blocker. Then:

1. promote `compz='Z'` into the production `syevd_mp(eigvecs=True)` path;
2. run production `syevd_mp(eigvecs=True)` on the same tiny rank-per-GPU case;
3. if production fails while sample-layout passes, focus on JAXMg descriptor,
   output, and reverse redistribution wiring;
4. keep `eigvecs=False` disabled unless `N` is separately proven.

### Case B: sample vectors pass only on private stream

Conclusion: `cusolverMpSyevd` is sensitive to the XLA FFI stream. Then:

1. make the SYEVD production path use a private nonblocking CUDA stream;
2. add explicit stream synchronization before and after the cuSOLVERMp call;
3. keep the Cholesky path unchanged unless benchmarks or correctness require
   the same stream policy;
4. retest production `syevd_mp(eigvecs=True)`.

This would be consistent with the v0.8.0 stream-related release-note changes,
although it would not prove the exact same default-stream bug.

### Case C: sample vectors fail on both XLA and private streams

Conclusion: the issue is probably not just the CUDA stream. Then:

1. build a diagnostic FFI path that creates a private NCCL communicator instead
   of borrowing XLA's communicator;
2. run the same sample-layout SYEVD probe with that private communicator;
3. if private NCCL works, treat borrowed-communicator compatibility as the
   blocker for SYEVD and either use private CAL/NCCL for cuSOLVERMp or defer
   SYEVD;
4. if private NCCL also fails, test cuSOLVERMp 0.8.0 before making larger
   JAXMg changes.

### Case D: `compz='N'` still fails

This is expected for 0.7.2 from the current logs. The release branch should not
advertise true no-vector cuSOLVERMp SYEVD unless it is proven on the target
runtime.

## Recommended Direction

The release branch should be split conceptually:

1. **Production-ready path:** keep hardening `potrs_mp`.
2. **Experimental path:** keep `syevd_mp(eigvecs=True)` behind validation until
   the `Z` diagnostic and production test pass.
3. **Unsupported for now:** remove or guard `syevd_mp(eigvecs=False)` for
   cuSOLVERMp 0.7.2, because `compz=N` has not worked.

The next engineering checkpoints are:

1. finish job `30581230`;
2. decide stream vs communicator vs runtime based on its vector cases;
3. run one production `syevd_mp(eigvecs=True)` case only if the sample-layout
   vector probe passes;
4. otherwise implement a private-communicator diagnostic before touching the
   production redistribution code again;
5. test cuSOLVERMp 0.8.0 if CSD3 or another system can provide it.

## Practical Release Guidance

Until SYEVD is resolved, the cleanest user-facing position is:

```text
potrs_mp: supported on the validated rank-per-GPU path.
syevd_mp(eigvecs=True): experimental, pending the Z/private-stream diagnostic.
syevd_mp(eigvecs=False): unsupported on cuSOLVERMp 0.7.2 unless implemented as
                         a documented compute-vectors-and-discard fallback.
potri / explicit inverse: unsupported in the cuSOLVERMp backend.
```

This keeps the codebase honest: the Cholesky migration has real evidence
behind it, while the eigensolver path remains under active investigation.
