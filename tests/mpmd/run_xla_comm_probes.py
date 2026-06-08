import json
import sys
import traceback

coord_addr = sys.argv[1]
proc_id = int(sys.argv[2])
num_procs = int(sys.argv[3])
name = sys.argv[4]

import jax

jax.distributed.initialize(
    coordinator_address=coord_addr,
    num_processes=num_procs,
    process_id=proc_id,
    local_device_ids=[proc_id],
    coordinator_bind_address=coord_addr if proc_id == 0 else None,
)

import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from jaxmg import (
    xla_comm_allreduce_probe_shardmap,
    xla_comm_ring_permute_probe_shardmap,
)


def _emit(status, **payload):
    print(
        "MPMD_RESULT "
        + json.dumps({"proc": proc_id, "name": name, "status": status, **payload}),
        flush=True,
    )


def _addressable_values(array):
    """Returns the flattened process-local shards of a global JAX array."""
    array.block_until_ready()
    return np.concatenate(
        [np.asarray(shard.data).reshape(-1) for shard in array.addressable_shards]
    )


def main():
    try:
        devices = jax.devices("gpu")
        mesh = Mesh(np.asarray(devices, dtype=object), ("x",))
        sharding = NamedSharding(mesh, P("x"))
        token = jax.device_put(jnp.arange(num_procs, dtype=jnp.uint32), sharding)

        allreduce = xla_comm_allreduce_probe_shardmap(token, mesh, P("x"))
        expected_sum = np.full(
            (1,), num_procs * (num_procs - 1) // 2, dtype=np.uint32
        )
        allreduce_local = _addressable_values(allreduce)
        if not np.array_equal(allreduce_local, expected_sum):
            _emit(
                "fail",
                check="allreduce",
                got=allreduce_local.tolist(),
                expected=expected_sum.tolist(),
            )
            return

        ring = xla_comm_ring_permute_probe_shardmap(token, mesh, P("x"))
        expected_ring = np.roll(np.arange(num_procs, dtype=np.uint32), 1)[
            proc_id : proc_id + 1
        ]
        ring_local = _addressable_values(ring)
        if not np.array_equal(ring_local, expected_ring):
            _emit(
                "fail",
                check="ring_permute",
                got=ring_local.tolist(),
                expected=expected_ring.tolist(),
            )
            return

        _emit(
            "ok",
            devices=[str(device) for device in devices],
            local_devices=[str(device) for device in jax.local_devices()],
        )
    except Exception:
        _emit("fail", traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
