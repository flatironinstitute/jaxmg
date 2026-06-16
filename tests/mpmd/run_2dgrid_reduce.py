import sys

proc_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
num_procs = int(sys.argv[2]) if len(sys.argv) > 2 else 1

# initialize the distributed system
import jax

jax.config.update("jax_platform_name", "cpu")
jax.distributed.initialize(
    coordinator_address="localhost:6000",
    num_processes=num_procs,
    process_id=proc_id,
    coordinator_bind_address="localhost:6000" if proc_id == 0 else None,
)

from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import jax.numpy as jnp
import numpy as np


def get_device_grid():
    by_proc = {}
    for d in jax.devices():
        by_proc.setdefault(d.process_index, []).append(d)
    hosts = sorted(by_proc)
    return np.array(
        [[by_proc[h][x] for x in range(jax.local_device_count())] for h in hosts]
    )


def create_2d_mesh():
    dev_grid = get_device_grid()
    return Mesh(dev_grid, ("x", "y"))


def create_1d_mesh():
    dev_grid = get_device_grid()
    return Mesh(dev_grid.flatten(), ("y",))


def main():
    print(f"Rank {proc_id}")
    print(f"Local devices {jax.local_device_count()}")
    print(f"Global devices {jax.device_count()}")
    print(f"World size {num_procs}")
    print(f"Device grid{get_device_grid()}")

    mesh2d = create_2d_mesh()
    dtype = jnp.float32
    A = jax.device_put(
        jnp.diag(jnp.arange(1, jax.device_count() + 1, dtype=dtype)),
        NamedSharding(mesh2d, P(None, ("x", "y"))),
    )

    shard_size = jax.device_count() // jax.local_device_count()

    for i, shard in enumerate(A.addressable_shards):
        print(f"shard\n {shard.data}")
        assert jnp.isclose(
            jnp.sum(shard.data), i + 1 + jax.process_index() * shard_size
        )

    # Gather over the number of hosts
    A = jax.lax.with_sharding_constraint(A, NamedSharding(mesh2d, P(None, "y")))
    print(f"Device: {jax.process_index()}")
    for i, shard in enumerate(A.addressable_shards):
        start = i * shard_size
        end = (i + 1) * shard_size
        assert jnp.allclose(
            shard.data[start:end, :],
            jnp.diag(jnp.arange(1, shard_size + 1, dtype=dtype) + i * shard_size),
        )


if __name__ == "__main__":
    main()
