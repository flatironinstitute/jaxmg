import json
import sys
import traceback

coord_addr = sys.argv[1]
proc_id = int(sys.argv[2])
num_procs = int(sys.argv[3])
name = sys.argv[4]

import jax

jax.config.update("jax_enable_x64", True)
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

from jaxmg import potrs


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


def _run_case(dtype, n, tile_size):
    devices = jax.devices("gpu")
    mesh = Mesh(np.asarray(devices, dtype=object), ("x",))
    a_sharding = NamedSharding(mesh, P("x", None))
    b_sharding = NamedSharding(mesh, P(None, None))

    diag = jnp.arange(1, n + 1, dtype=dtype)
    a = jnp.diag(diag)
    b = jnp.ones((n, 1), dtype=dtype)
    expected = np.asarray((1 / diag).reshape(n, 1))

    a = jax.device_put(a, a_sharding)
    b = jax.device_put(b, b_sharding)
    out, status = potrs(
        a.copy(),
        b.copy(),
        tile_size,
        mesh=mesh,
        in_specs=(P("x", None),),
        return_status=True,
    )
    out_local = _addressable_values(out).reshape(n, 1)
    status_local = _addressable_values(status.reshape((1,)))

    if not np.array_equal(status_local, np.zeros((1,), dtype=np.int32)):
        _emit(
            "fail",
            check="potrs_status",
            dtype=str(dtype),
            got=status_local.tolist(),
            expected=[0],
        )
        return False

    if not np.allclose(out_local, expected, rtol=2e-4, atol=2e-4):
        _emit(
            "fail",
            check="potrs_solution",
            dtype=str(dtype),
            got=out_local.tolist(),
            expected=expected.tolist(),
        )
        return False

    return True


def main():
    try:
        n = num_procs * 2
        for dtype in (jnp.float32, jnp.float64, jnp.complex64):
            if not _run_case(dtype, n=n, tile_size=1):
                return

        _emit(
            "ok",
            devices=[str(device) for device in jax.devices("gpu")],
            local_devices=[str(device) for device in jax.local_devices()],
        )
    except Exception:
        _emit("fail", traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
