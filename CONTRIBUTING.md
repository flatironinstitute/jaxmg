# Contributing

## Development priorities

- **Better error handling**. There are parts of the code that simply throw
  `std::runtime_error`. We need to make the error handling compatible with XLA
  FFI error handlers such as `FFI_ASSIGN_OR_RETURN` and
  `JAX_FFI_RETURN_IF_GPU_ERROR`.

## Build from source

The native backend must be compiled against the XLA revision associated with
the pinned JAX release. It is built with Bazel from a matching JAX source
checkout inside the official JAX CI container. Separate CUDA 12 and CUDA 13
libraries are installed under `src/jaxmg/cu12/` and `src/jaxmg/cu13/`.

See
[Building the native backend](https://flatironinstitute.github.io/jaxmg/technical_details/building_from_source/)
for the complete procedure and build configuration.

## JAX and CUDA

The Python package pins `jax==0.10.1` (see `pyproject.toml`). The native backend
uses internal OpenXLA APIs. Changing JAX therefore requires more than updating
a Python dependency:

1. Audit the `@xla` targets in `bazel/jaxmg_backend.BUILD.bazel` against the XLA
   revision used by the new JAX release.
2. Update `JAX_GIT_TAG` in `.github/workflows/build-wheels.yml`.
3. Rebuild and test both CUDA backends.

The `cuda12`, `cuda12-local`, `cuda13`, and `cuda13-local` package extras select
the matching JAX and cuSOLVERMp runtime dependencies.

## Continuous integration

The continuous-integration pipeline separates building the package from testing
it on physical GPUs.

GitHub Actions builds the release artifacts in the official JAX CI
environments. It compiles the CUDA 12 and CUDA 13 backends natively for
`x86_64` and `aarch64`, packages `manylinux_2_28` wheels for Python 3.11–3.14,
and checks that each wheel contains both CUDA backends with valid metadata.

The completed x86 wheels are then passed to Jenkins, which provides the NVIDIA
hardware needed for runtime validation. Jenkins installs those exact wheels and
runs the CUDA 12 test suite on two A100 GPUs for every supported Python version.
This ensures that the artifacts produced by GitHub Actions are tested without
being rebuilt in a different environment.

The AArch64 and CUDA 13 wheels currently receive build and packaging validation
in GitHub Actions, but are not exercised by the Jenkins GPU tests.

See `.github/workflows/build-wheels.yml`, `.github/workflows/ci.yml`, and
`.jenkins/Jenkinsfile` for the exact configurations.

## Documentation setup

Install the documentation dependencies and build the site strictly before
submitting documentation changes:

```bash
python -m pip install -e ".[docs]"
python -m mkdocs build --strict
```

Use `python -m mkdocs serve` for a local preview. Documentation changes are
validated in pull requests and deployed from `main` by
`.github/workflows/deploy-docs.yml`.

## Publish package

Releases are automated by `.github/workflows/release.yml`. Publishing uses PyPI
Trusted Publishing through GitHub's OIDC identity, so no PyPI token is stored in
the repository.

The git tag is the single source of truth for the version. To cut a release:

1. Make sure `main` is green and the intended version is reflected in
   `pyproject.toml`.
2. Push a version tag:

   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

3. The release workflow then:
   - builds x86_64 and AArch64 wheels for Python 3.11–3.14;
   - runs the Jenkins A100 tests against the x86 wheels;
   - publishes all wheels to TestPyPI;
   - waits for approval on the `pypi` environment;
   - publishes to PyPI and attaches the wheels to the GitHub release.
4. Before approving the PyPI step, sanity-check the TestPyPI upload:

   ```bash
   python -m pip install \
     --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ \
     "jaxmg[cuda12]==1.0.0"
   ```

For a release candidate, run the release workflow manually with a pre-release
version such as `1.0.0rc1`. This exercises the same build, test, and TestPyPI
path. Do not approve the `pypi` environment unless that version should also be
published to PyPI.

### One-time setup

- Configure the `JENKINS_JOB_URL`, `JENKINS_USER`, and `JENKINS_API_TOKEN`
  repository secrets.
- Configure Trusted Publishers on PyPI and TestPyPI for this repository and
  `.github/workflows/release.yml`.
- Create GitHub environments named `testpypi` and `pypi`, with a required
  reviewer on `pypi`.
