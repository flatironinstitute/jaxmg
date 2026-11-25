# Building from source

The native backend is built with CMake against the public FFI headers shipped
with `jaxlib`. It borrows XLA's NCCL communicator through the Collectives C
extension; a JAX/XLA source checkout and Bazel are not required.

## Requirements

- Linux on `x86_64` or `aarch64`;
- Python 3.12 or later;
- a CUDA 12.9 or CUDA 13 development toolkit, including `nvcc`;
- CMake 3.26 or later, Ninja, Git, and a C++17 compiler;
- the matching cuSOLVERMp and NCCL development headers and libraries.

CMake fetches a pinned Abseil release for internal status handling. Abseil and
JAXMg's C++ symbols remain private to the backend; only the C FFI handlers are
exported. Neither XLA nor `jaxlib` is linked into the library.

## Build the backend

From the JAXMg repository, prepare a Python environment and install the
minimum supported FFI headers and the CUDA 12 dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install cmake ninja 'jax==0.11.2' \
  'nvidia-cusolvermp-cu12==0.10.0.27368' 'nvidia-nccl-cu12==2.32.6'
CUDA_HOME=/usr/local/cuda-12.9 ./build_native_backend.sh
```

For CUDA 13, install the corresponding `cu13` packages and point `CUDA_HOME`
at that toolkit. The script discovers the installed FFI and NVIDIA wheel
headers and writes the backend to
`src/jaxmg/cu12/libjaxmg_xla_comm_backend.so` or the corresponding `cu13`
directory.

The default build includes the supported GPU architectures and PTX for forward
compatibility. For a local build targeting only an H100, for example:

```bash
JAXMG_CUDA_ARCHITECTURES='90' JAXMG_BUILD_JOBS=8 ./build_native_backend.sh
```

If the NVIDIA libraries are installed outside the Python environment, pass
`-DCUSOLVERMP_INCLUDE_DIR=...`, `-DCUSOLVERMP_LIBRARY=...`,
`-DNCCL_INCLUDE_DIR=...`, and `-DNCCL_LIBRARY=...` to the build script. Library
arguments must name the shared-library files, not their directories.

## Package and test

Install the source checkout after building the backend:

```bash
python -m pip install -e '.[cuda12]'
```

The cache and FFI contract tests use fake library calls and run without a GPU:

```bash
JAXMG_RUN_NATIVE_TESTS=ON ./build_native_backend.sh
```

Numerical validation still requires GPUs. GitHub Actions builds both CUDA
backends and packages them into one Python-independent wheel per architecture;
Jenkins tests the same x86_64 wheel with the supported JAX runtimes. See
[CONTRIBUTING.md](https://github.com/flatironinstitute/jaxmg/blob/main/CONTRIBUTING.md)
for the test and release workflows.
