import json
import os
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


def _shm_snapshot():
    if not os.path.isdir("/dev/shm"):
        return set()
    return {name for name in os.listdir("/dev/shm") if name.startswith("jaxmg_")}


def _spd_matrix(n, dtype):
    values = np.arange(1, n * n + 1, dtype=np.float64).reshape(n, n)
    a = values @ values.T
    a += np.eye(n, dtype=np.float64) * (n * n)
    return a.astype(dtype)


def main():
    try:
        iterations = int(os.environ.get("JAXMG_MPMD_STRESS_ITERS", "25"))
        n = int(os.environ.get("JAXMG_MPMD_STRESS_N", str(num_procs * 16)))
        nrhs = int(os.environ.get("JAXMG_MPMD_STRESS_NRHS", "3"))
        tile_size = int(os.environ.get("JAXMG_MPMD_STRESS_TILE", "4"))
        dtype = np.float64

        devices = jax.devices("gpu")
        mesh = Mesh(np.asarray(devices, dtype=object), ("x",))
        a_sharding = NamedSharding(mesh, P("x", None))
        b_sharding = NamedSharding(mesh, P(None, None))

        a_host = _spd_matrix(n, dtype)
        b_host = (
            np.arange(n * nrhs, dtype=dtype).reshape(n, nrhs) / n
            + np.ones((n, nrhs), dtype=dtype)
        )
        expected = np.linalg.solve(a_host, b_host)
        a = jax.device_put(jnp.asarray(a_host), a_sharding)
        b = jax.device_put(jnp.asarray(b_host), b_sharding)

        fd_start = len(os.listdir("/proc/self/fd")) if os.path.isdir("/proc/self/fd") else -1
        shm_start = _shm_snapshot()
        last_out = None
        last_status = None

        for _ in range(iterations):
            last_out, last_status = potrs(
                a.copy(),
                b.copy(),
                tile_size,
                mesh=mesh,
                in_specs=(P("x", None),),
                return_status=True,
            )
            last_out.block_until_ready()
            last_status.block_until_ready()

        out_local = _addressable_values(last_out).reshape(b_host.shape)
        status_local = _addressable_values(last_status.reshape((1,)))
        fd_end = len(os.listdir("/proc/self/fd")) if os.path.isdir("/proc/self/fd") else -1
        shm_end = _shm_snapshot()

        if not np.array_equal(status_local, np.zeros((1,), dtype=np.int32)):
            _emit("fail", check="potrs_stress_status", got=status_local.tolist())
            return
        if not np.allclose(out_local, expected, rtol=5e-4, atol=5e-4):
            _emit(
                "fail",
                check="potrs_stress_solution",
                got=out_local.tolist(),
                expected=expected.tolist(),
            )
            return
        if fd_start >= 0 and fd_end - fd_start > 32:
            _emit(
                "fail",
                check="potrs_stress_fd_leak",
                fd_start=int(fd_start),
                fd_end=int(fd_end),
            )
            return
        leaked_shm = sorted(shm_end - shm_start)
        if leaked_shm:
            _emit("fail", check="potrs_stress_shm_leak", leaked_shm=leaked_shm)
            return

        _emit(
            "ok",
            iterations=int(iterations),
            n=int(n),
            nrhs=int(nrhs),
            tile_size=int(tile_size),
            fd_delta=int(fd_end - fd_start) if fd_start >= 0 else None,
        )
    except Exception:
        _emit("fail", traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
