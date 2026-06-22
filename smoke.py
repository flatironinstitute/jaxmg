import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding
from jaxmg import potrs

jax.distributed.initialize(coordinator_address="localhost:12399", num_processes=1, process_id=0)

T_A = 3
N = T_A * jax.process_count()
A = jnp.diag(jnp.arange(N, dtype=jnp.float64) + 1)
b = jnp.ones((N, 1), dtype=jnp.float64)
mesh = jax.make_mesh((jax.process_count(), 1), ("pr", "pc"))
sh = NamedSharding(mesh, P("pr", "pc"))
out = potrs(jax.device_put(A, sh), jax.device_put(b, sh), T_A=T_A)
out.block_until_ready()
print(out.flatten())   # -> [1. 0.5 0.333...]
