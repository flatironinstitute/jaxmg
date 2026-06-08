import json
import os
import sys
import time

mode = sys.argv[1]
if mode == "mpmd":
    coord_addr = sys.argv[2]
    proc_id = int(sys.argv[3])
    num_procs = int(sys.argv[4])
elif mode == "spmd":
    coord_addr = None
    proc_id = 0
    num_procs = int(sys.argv[2])
else:
    raise ValueError("mode must be 'spmd' or 'mpmd'")

os.environ["JAXMG_NUMBER_OF_DEVICES"] = str(num_procs)

import jax

jax.config.update("jax_enable_x64", True)
if mode == "mpmd":
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

from jaxmg import potri, potrs, syevd


def _emit(**payload):
    print("JAXMG_BENCH_RESULT " + json.dumps(payload), flush=True)


def _spd_matrix(n, dtype):
    values = np.arange(1, n * n + 1, dtype=np.float64).reshape(n, n)
    a = values @ values.T
    a += np.eye(n, dtype=np.float64) * (n * n)
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        imag = np.tril(values / (n * n * 10), k=-1)
        skew = imag - imag.T
        a = a.astype(np.complex128) + 1j * skew
    return a.astype(dtype)


def _block_until_ready(result):
    if isinstance(result, tuple):
        for value in result:
            _block_until_ready(value)
        return
    result.block_until_ready()


def _time_call(label, fn, warmups, iterations):
    for _ in range(warmups):
        _block_until_ready(fn())
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        _block_until_ready(fn())
        samples.append(time.perf_counter() - start)
    if mode == "spmd" or proc_id == 0:
        _emit(
            mode=mode,
            proc=int(proc_id),
            benchmark=label,
            samples_s=samples,
            mean_s=float(np.mean(samples)),
            min_s=float(np.min(samples)),
        )


def main():
    n = int(os.environ.get("JAXMG_BENCH_N", str(num_procs * 256)))
    nrhs = int(os.environ.get("JAXMG_BENCH_NRHS", "4"))
    tile_size = int(os.environ.get("JAXMG_BENCH_TILE", "64"))
    warmups = int(os.environ.get("JAXMG_BENCH_WARMUPS", "1"))
    iterations = int(os.environ.get("JAXMG_BENCH_ITERS", "5"))

    devices = jax.devices("gpu")[:num_procs]
    mesh = Mesh(np.asarray(devices, dtype=object), ("x",))
    a_sharding = NamedSharding(mesh, P("x", None))
    b_sharding = NamedSharding(mesh, P(None, None))

    for dtype in (np.float64, np.complex64):
        a_host = _spd_matrix(n, dtype)
        b_host = (
            np.arange(n * nrhs, dtype=np.float64).reshape(n, nrhs) / n
            + np.ones((n, nrhs), dtype=np.float64)
        ).astype(dtype)
        a = jax.device_put(jnp.asarray(a_host), a_sharding)
        b = jax.device_put(jnp.asarray(b_host), b_sharding)

        _time_call(
            f"potrs_{np.dtype(dtype).name}",
            lambda a=a, b=b: potrs(
                a.copy(),
                b.copy(),
                tile_size,
                mesh=mesh,
                in_specs=(P("x", None),),
                return_status=True,
            ),
            warmups,
            iterations,
        )
        _time_call(
            f"potri_{np.dtype(dtype).name}",
            lambda a=a: potri(
                a.copy(),
                tile_size,
                mesh=mesh,
                in_specs=(P("x", None),),
                return_status=True,
            ),
            warmups,
            iterations,
        )
        _time_call(
            f"syevd_no_v_{np.dtype(dtype).name}",
            lambda a=a: syevd(
                a.copy(),
                tile_size,
                mesh=mesh,
                in_specs=(P("x", None),),
                return_eigenvectors=False,
                return_status=True,
            ),
            warmups,
            iterations,
        )


if __name__ == "__main__":
    main()
