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


def _spd_matrix(n, dtype):
    values = np.arange(1, n * n + 1, dtype=np.float64).reshape(n, n)
    a = values @ values.T
    a += np.eye(n, dtype=np.float64) * (n * n)
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        imag = np.tril(values / (n * n * 10), k=-1)
        skew = imag - imag.T
        a = a.astype(np.complex128) + 1j * skew
    return a.astype(dtype)


def _run_case(case_name, dtype, a_host, b_host, tile_size):
    devices = jax.devices("gpu")
    mesh = Mesh(np.asarray(devices, dtype=object), ("x",))
    a_sharding = NamedSharding(mesh, P("x", None))
    b_sharding = NamedSharding(mesh, P(None, None))
    expected = np.linalg.solve(a_host, b_host)

    a = jax.device_put(jnp.asarray(a_host), a_sharding)
    b = jax.device_put(jnp.asarray(b_host), b_sharding)
    out, status = potrs(
        a.copy(),
        b.copy(),
        tile_size,
        mesh=mesh,
        in_specs=(P("x", None),),
        return_status=True,
    )
    out_local = _addressable_values(out).reshape(b_host.shape)
    status_local = _addressable_values(status.reshape((1,)))

    if not np.array_equal(status_local, np.zeros((1,), dtype=np.int32)):
        _emit(
            "fail",
            check="potrs_status",
            case=case_name,
            dtype=str(dtype),
            n=int(a_host.shape[0]),
            nrhs=int(b_host.shape[1]),
            tile_size=int(tile_size),
            got=status_local.tolist(),
            expected=[0],
        )
        return False

    if not np.allclose(out_local, expected, rtol=5e-4, atol=5e-4):
        _emit(
            "fail",
            check="potrs_solution",
            case=case_name,
            dtype=str(dtype),
            n=int(a_host.shape[0]),
            nrhs=int(b_host.shape[1]),
            tile_size=int(tile_size),
            got=out_local.tolist(),
            expected=expected.tolist(),
        )
        return False

    return True


def main():
    try:
        dtypes = (np.float32, np.float64, np.complex64)
        for dtype in dtypes:
            n = num_procs * 2
            diag = np.arange(1, n + 1, dtype=dtype)
            if not _run_case(
                "diagonal_no_padding",
                dtype,
                np.diag(diag),
                np.ones((n, 1), dtype=dtype),
                tile_size=1,
            ):
                return

            padded_n = num_procs * 3
            padded_diag = np.arange(1, padded_n + 1, dtype=dtype)
            if not _run_case(
                "diagonal_with_padding",
                dtype,
                np.diag(padded_diag),
                np.ones((padded_n, 1), dtype=dtype),
                tile_size=2,
            ):
                return

        multi_rhs_n = num_procs * 2
        if not _run_case(
            "multiple_rhs",
            np.float64,
            _spd_matrix(multi_rhs_n, np.float64),
            (np.arange(multi_rhs_n * 3, dtype=np.float64).reshape(multi_rhs_n, 3)
             + 1),
            tile_size=1,
        ):
            return

        spd_n = num_procs * 3
        if not _run_case(
            "non_diagonal_spd",
            np.float64,
            _spd_matrix(spd_n, np.float64),
            np.ones((spd_n, 1), dtype=np.float64),
            tile_size=1,
        ):
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
