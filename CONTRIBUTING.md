# Contributing

Please help us improve JAXMg. You can contribute by
[reporting a bug or suggesting a feature](https://github.com/flatironinstitute/jaxmg/issues),
improving the documentation, or
[opening a pull request](https://github.com/flatironinstitute/jaxmg/pulls)
with a code change. Contributions of any size are welcome.

A typical contribution follows five steps:

1. pull a clean copy of the repository and create a branch;
2. set up a development environment;
3. make a focused change;
4. run the checks relevant to that change;
5. open a pull request describing the change and its validation.

## Current development priority

1. Some native failures still use `std::runtime_error`. Contributions that
   express these consistently through XLA FFI error handling are particularly
   useful.

2. We would like to make the JAXMg solver routines differentiable. This will
   require custom JVP and VJP rules that tell JAX how to propagate derivatives
   through the FFI-backed distributed calculations.

## Pull a clean copy

Fork the repository first if you do not have write access. Then clone your fork
or the main repository, update `main`, and create a branch:

```bash
git clone https://github.com/flatironinstitute/jaxmg.git
cd jaxmg
git switch main
git pull --ff-only
git switch -c my-change
```

## Set up an environment

Create a virtual environment and install the development dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

For documentation-only work on a system without CUDA, install `.[docs]` and
pytest instead.

## Make your changes

The main areas of the repository are:

- `src/jaxmg/` for the public Python interface and JAX integration;
- `src/cuda/` for memory redistribution and cuSOLVERMp routines;
- `tests/` for interface and distributed numerical tests;
- `docs/` for user guides, examples, and technical documentation;
- `.github/`, `.jenkins/`, and `pyproject.toml` for building and packaging.

Changes under `src/cuda/`, `bazel/`, or `build_native_backend.sh` require a
native rebuild. Follow
[Building the native backend](https://flatironinstitute.github.io/jaxmg/latest/technical_details/building_from_source/)
before running the GPU tests.

Public behavior changes should include corresponding tests and documentation.

## Test your changes

Run the checks relevant to the change:

| Change | Check |
|---|---|
| Python API, validation, or layout metadata | `python -m pytest -q -m "not gpu" tests` |
| Documentation | `python -m mkdocs build --strict` |
| Native solver or redistribution code | Rebuild the backend and run the GPU tests |
| Packaging or dependencies | Build and install a wheel through the matching CUDA extra |

The GPU suite distinguishes between `single_gpu` and `multi_gpu` tests. The
multi-GPU tests launch one Python process per GPU. On a workstation with two
GPUs, run the complete smoke suite with:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
JAXMG_GPU_TEST_LAUNCHER=subprocess \
python -m pytest -q -m "gpu and not slow" tests/gpu
```

The two smoke groups can also be selected independently with
`-m "single_gpu and not slow"` or `-m "multi_gpu and not slow"`. Tests
requiring more GPUs than are visible are skipped. On Slurm systems, the test
helper can use `srun` across the current allocation.

## Open a pull request

Push your branch and
[open a pull request](https://github.com/flatironinstitute/jaxmg/compare):

```bash
git push -u origin my-change
```

In the pull-request description, explain:

- what changed and why;
- any user-visible or compatibility effects;
- which tests were run;
- any hardware-dependent checks that could not be run locally.

Generated build outputs should not be committed.

## Continuous integration

The release build and GPU tests run on separate systems:

1. **GitHub Actions builds** CUDA 12 and CUDA 13 backends for `x86_64` and
   `aarch64`.
2. **GitHub Actions packages** Python 3.11–3.14 `manylinux_2_28` wheels and
   checks their metadata and native libraries.
3. **Jenkins downloads those wheels** and installs the normal
   `jaxmg[cuda12]` dependency path.
4. **Jenkins tests them** on two physical A100 GPUs using the interface and GPU
   smoke suites.

This ensures that Jenkins tests the wheel produced by GitHub Actions rather
than rebuilding it. AArch64 and CUDA 13 are currently build-validated but are
not runtime-tested by Jenkins.

The implementation is in:

- `.github/workflows/build-wheels.yml`;
- `.github/workflows/ci.yml`;
- `.github/workflows/jenkins-test.yml`;
- `.jenkins/Jenkinsfile`.

## Versioned documentation

The site is published per release with [mike](https://github.com/jimporter/mike)
on the `gh-pages` branch, so documentation for an older release keeps matching
the wheel that shipped it:

| Trigger | Published as |
|---|---|
| Pull request | nothing — `mkdocs build --strict` runs as a check only |
| Push to `main` | `dev`, a mutable build of the development tree |
| Release tag `vX.Y.Z` | `X.Y.Z`, and takes over the `latest` alias |

The bare site URL redirects to whatever `latest` points at, so links in the
README and in `pyproject.toml` should use `/latest/…` rather than a version
number. `0.0.9` is archived because it wrapped the older single-node cuSolverMg
API; do not redeploy it.

Numbered versions are written only by `release.yml`. Never run
`mike deploy X.Y.Z` by hand for a version that has already shipped — that would
silently replace the documentation for a released wheel.

## Updating JAX or CUDA

JAXMg pins `jax==0.10.1` and builds against its matching internal XLA revision.
Updating JAX therefore requires auditing the `@xla` targets, updating the JAX
source tag and dependencies, and rebuilding both CUDA backends on both
architectures.

The same care is required when changing CUDA, cuSOLVERMp, or the supported GPU
architecture list.

## Release process

This section is for maintainers.

Before a final release, run `.github/workflows/release.yml` manually with a
release candidate such as `1.0.0rc1`. The workflow builds all wheels, runs the
Jenkins GPU tests, and publishes to TestPyPI.

After checking the TestPyPI installation, create the final tag:

```bash
git tag v1.0.0
git push origin v1.0.0
```

The tagged workflow repeats the build and test process, publishes to TestPyPI,
waits for approval, then publishes to PyPI and creates the GitHub release.

Publishing requires the Jenkins secrets, PyPI Trusted Publishers, and
`testpypi` and `pypi` GitHub environments described in
`.github/workflows/release.yml`.
