# Installation
The package is available on PyPi and can be installed with

```bash
pip install jaxmg[cuda12]
```

This will install a GPU compatible version of JAX. 

1. `pip install "jaxmg[cuda12]"`: Use CUDA 12 (only works for `jax>=0.6.2`).

2. `pip install "jaxmg[cuda12-local]"`: Use locally available CUDA 12 installation.

3. `pip install "jaxmg[cuda13]"`: Use CUDA 13 (only works for `jax>=0.7.2`).

4. `pip install "jaxmg[cuda13-local]"`: Use locally available CUDA 13 installation.

The provided binaries are compiled with

|**JAXMg** | **CUDA** | **cuDNN** |
|---|---|---| 
| `cuda12`,`cuda12-local` | 12.8.0 | 9.17.1.4|
| `cuda13`,`cuda13-local` | 13.0.0 | 9.17.1.4|

> **_Note:_** `pip install jaxmg` will install a CPU-only version of JAX. Since `jaxmg` is a GPU-only package you will receive a warning to install a GPU-compatible version of jax. 