# Symmetric or Hermitian eigensolve

`jaxmg.syevd` computes the eigenvalues and eigenvectors of a symmetric real
matrix or Hermitian complex matrix.

```python
import jax

jax.config.update("jax_enable_x64", True)
jax.distributed.initialize()

import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from jaxmg import syevd


num_processes = jax.process_count()
mesh = jax.make_mesh((num_processes, 1), ("pr", "pc"))
matrix_specs = P("pr", "pc")

T_A = 64
N = T_A * num_processes
dtype = jnp.float64

expected_eigenvalues = jnp.arange(1, N + 1, dtype=dtype)
a = jnp.diag(expected_eigenvalues)
a = jax.device_put(a, NamedSharding(mesh, matrix_specs))

eigenvalues, eigenvectors = syevd(
    a,
    T_A=T_A,
    mesh=mesh,
    matrix_specs=matrix_specs,
)
eigenvectors.block_until_ready()

correct = jnp.allclose(eigenvalues, expected_eigenvalues)
correct.block_until_ready()

if jax.process_index() == 0:
    print(correct)
```

The eigenvalues are returned as a real array. The eigenvectors have the input
dtype and are returned in the same JAX-facing matrix layout as `a`.

SYEVD materializes the full distributed eigenvector matrix and uses
solver-specific workspace. It therefore reaches the GPU memory limit at a
smaller matrix size than `potrs` or `lu_solve` on the same process grid.
