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

from jaxmg import potri


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


def _check_addressable_shards(array, expected, rtol, atol):
    array.block_until_ready()
    for shard in array.addressable_shards:
        got = np.asarray(shard.data)
        want = expected[shard.index]
        if got.shape != want.shape:
            return False, got, want, shard.index
        if not np.allclose(got, want, rtol=rtol, atol=atol):
            return False, got, want, shard.index
    return True, None, None, None


def _spd_matrix(n, dtype):
    values = np.arange(1, n * n + 1, dtype=np.float64).reshape(n, n)
    a = values @ values.T
    a += np.eye(n, dtype=np.float64) * (n * n)
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        imag = np.tril(values / (n * n * 10), k=-1)
        skew = imag - imag.T
        a = a.astype(np.complex128) + 1j * skew
    return a.astype(dtype)


def _run_case(case_name, dtype, a_host, tile_size):
    devices = jax.devices("gpu")
    mesh = Mesh(np.asarray(devices, dtype=object), ("x",))
    a_sharding = NamedSharding(mesh, P("x", None))
    expected = np.linalg.inv(a_host)

    a = jax.device_put(jnp.asarray(a_host), a_sharding)
    out, status = potri(
        a.copy(),
        tile_size,
        mesh=mesh,
        in_specs=(P("x", None),),
        return_status=True,
    )
    status_local = _addressable_values(status.reshape((1,)))

    if not np.array_equal(status_local, np.zeros((1,), dtype=np.int32)):
        _emit(
            "fail",
            check="potri_status",
            case=case_name,
            dtype=str(dtype),
            n=int(a_host.shape[0]),
            tile_size=int(tile_size),
            got=status_local.tolist(),
            expected=[0],
        )
        return False

    ok, got, want, index = _check_addressable_shards(out, expected, 2e-3, 2e-3)
    if not ok:
        _emit(
            "fail",
            check="potri_inverse",
            case=case_name,
            dtype=str(dtype),
            n=int(a_host.shape[0]),
            tile_size=int(tile_size),
            shard_index=str(index),
            got=got.tolist(),
            expected=want.tolist(),
        )
        return False

    return True


def main():
    try:
        dtypes = (np.float32, np.float64, np.complex64)
        for dtype in dtypes:
            n = num_procs * 2
            diag = np.arange(1, n + 1, dtype=np.float64).astype(dtype)
            if not _run_case(
                "diagonal_no_padding",
                dtype,
                np.diag(diag),
                tile_size=1,
            ):
                return

            padded_n = num_procs * 3
            padded_diag = np.arange(1, padded_n + 1, dtype=np.float64).astype(
                dtype
            )
            if not _run_case(
                "diagonal_with_padding",
                dtype,
                np.diag(padded_diag),
                tile_size=2,
            ):
                return

        spd_n = num_procs * 3
        if not _run_case(
            "non_diagonal_spd",
            np.float64,
            _spd_matrix(spd_n, np.float64),
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
