# In file gpu_example.py...
import os

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"
# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
# os.environ["JAXMG_CUSOLVER_UTILS_VERBOSE"] = "1"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import sys



def random_psd(n, dtype, seed):
    """
    Generate a random n x n positive semidefinite matrix.
    """
    key = jax.random.key(seed)
    A = jax.random.normal(key, (n, n), dtype=dtype) / jnp.sqrt(n)
    return A @ A.T.conj() + jnp.eye(n, dtype=dtype) * 1e-5  # symmetric PSD

#
N = 8
print("N=", N)
NRHS = 1
T_A = 1  # From 2120 onwards it gives NaN


print("shards", (N // 2) / T_A)

# Get the coordinator_address, process_id, and num_processes from the command line.
coord_addr = sys.argv[1]
proc_id = int(sys.argv[2])
num_procs = int(sys.argv[3])

# Initialize the GPU machines.
jax.distributed.initialize(
    coordinator_address=coord_addr,
    num_processes=num_procs,
    process_id=proc_id,
    local_device_ids=proc_id,
)
print("process id =", jax.process_index())
print("global devices =", jax.devices())
print("local devices =", jax.local_devices())
print("visible devices", os.environ["CUDA_VISIBLE_DEVICES"])
import jax.numpy as jnp

from jax.sharding import NamedSharding, PartitionSpec as P
from jaxmg import potri, syevd
from jaxmg import determine_distributed_setup
from jaxmg import calculate_padding
from jaxmg.utils import random_psd
from jax.experimental.multihost_utils import process_allgather
print(determine_distributed_setup())
dtype = jnp.complex64
ndev = int(os.environ["JAXMG_NUMBER_OF_DEVICES"])
chunk_size = N // ndev
mesh = jax.make_mesh((ndev,), ("x",))


@jax.jit
def run_once():
    _A = jax.lax.with_sharding_constraint(
        # random_psd(N, dtype=dtype, seed=100),
        jnp.diag(jnp.arange(N, dtype=dtype) + 1), 
        NamedSharding(mesh, P("x", None))
    )
    return _A


A = run_once()

out, status = potri(A, T_A, mesh, (P("x", None), ), return_status=True, pad=True)
print(process_allgather(out, tiled=True))
print(status)
exit()