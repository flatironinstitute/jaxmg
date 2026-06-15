# Multi-Process Launches

JAXMg no longer exposes separate `_mp` solver wrappers or a separate CUDA IPC
pointer-sharing backend. Multi-process execution is handled through ordinary
JAX distributed setup, and the public functions remain:

```python
jaxmg.potrs(...)
jaxmg.syevd(...)
```

The fused cuSOLVERMp FFI handlers use the same native path regardless of
whether a job is single-node or multi-node:

```text
JAX block-sharded input
  -> JAX-side local padding
  -> fused native FFI call
       -> native 2D redistribution
       -> borrowed XLA-owned NCCL communicator
       -> cuSOLVERMp routine
       -> reverse native redistribution
  -> JAX-side unpadding
```

The supported multi-process contract is rank-per-GPU: one Python process owns
one participating GPU. User code should call `jax.distributed.initialize()`
before any operation that can initialize the JAX backend, then build a normal
JAX mesh over the participating devices.

```python
import jax
from jax.sharding import NamedSharding, PartitionSpec as P

import jaxmg

jax.distributed.initialize(...)

mesh = jax.make_mesh((2, 4), ("pr", "pc"))
sharding = NamedSharding(mesh, P("pr", "pc"))
```

JAXMg inspects the resulting mesh and accepts row-major or column-major process
rank mappings that cuSOLVERMp can represent directly. Exotic mesh permutations
are rejected early instead of being silently remapped inside native code.

One-process-per-node launches may be useful for diagnostics, but they are not
the primary cuSOLVERMp contract unless explicitly validated for the target
routine and runtime. The package does not create or manage distributed JAX
processes itself.
