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


def main():
    try:
        devices = jax.devices("gpu")
        mesh = Mesh(np.asarray(devices, dtype=object), ("x",))
        sharding = NamedSharding(mesh, P("x"))
        token = jax.device_put(jnp.arange(num_procs, dtype=jnp.uint32), sharding)

        allreduce = xla_comm_allreduce_probe_shardmap(token, mesh, P("x"))
        allreduce.block_until_ready()
        expected_sum = jnp.full((num_procs,), num_procs * (num_procs - 1) // 2, jnp.uint32)
        if not jnp.array_equal(jax.device_get(allreduce), expected_sum):
            _emit(
                "fail",
                check="allreduce",
                got=jax.device_get(allreduce).tolist(),
                expected=expected_sum.tolist(),
            )
            return

        ring = xla_comm_ring_permute_probe_shardmap(token, mesh, P("x"))
        ring.block_until_ready()
        expected_ring = jnp.roll(jnp.arange(num_procs, dtype=jnp.uint32), 1)
        if not jnp.array_equal(jax.device_get(ring), expected_ring):
            _emit(
                "fail",
                check="ring_permute",
                got=jax.device_get(ring).tolist(),
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
