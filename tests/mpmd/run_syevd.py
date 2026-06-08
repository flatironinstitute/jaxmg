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

from jaxmg import syevd


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


def _run_case(case_name, dtype, a_host, tile_size, return_eigenvectors):
    devices = jax.devices("gpu")
    mesh = Mesh(np.asarray(devices, dtype=object), ("x",))
    a_sharding = NamedSharding(mesh, P("x", None))
    expected_eigenvalues = np.linalg.eigvalsh(a_host)

    a = jax.device_put(jnp.asarray(a_host), a_sharding)
    if return_eigenvectors:
        eigenvalues, vectors, status = syevd(
            a.copy(),
            tile_size,
            mesh=mesh,
            in_specs=(P("x", None),),
            return_eigenvectors=True,
            return_status=True,
        )
    else:
        eigenvalues, status = syevd(
            a.copy(),
            tile_size,
            mesh=mesh,
            in_specs=(P("x", None),),
            return_eigenvectors=False,
            return_status=True,
        )
        vectors = None

    status_local = _addressable_values(status.reshape((1,)))
    if not np.array_equal(status_local, np.zeros((1,), dtype=np.int32)):
        _emit(
            "fail",
            check="syevd_status",
            case=case_name,
            dtype=str(dtype),
            vectors=bool(return_eigenvectors),
            n=int(a_host.shape[0]),
            tile_size=int(tile_size),
            got=status_local.tolist(),
            expected=[0],
        )
        return False

    eigenvalues_local = _addressable_values(eigenvalues)
    if not np.allclose(eigenvalues_local, expected_eigenvalues, rtol=2e-3, atol=2e-3):
        _emit(
            "fail",
            check="syevd_eigenvalues",
            case=case_name,
            dtype=str(dtype),
            vectors=bool(return_eigenvectors),
            n=int(a_host.shape[0]),
            tile_size=int(tile_size),
            got=eigenvalues_local.tolist(),
            expected=expected_eigenvalues.tolist(),
        )
        return False

    if return_eigenvectors and dtype is np.float64:
        # `syevd` returns eigenvectors transposed relative to cuSolverMg's
        # column-eigenvector storage: rows of the public `vectors` result are
        # eigenvectors. Check that convention directly and also check
        # orthogonality plus reconstruction to catch row/column layout errors.
        diag = jnp.diag(eigenvalues)
        eigen_residual = jnp.linalg.norm(vectors @ a - diag @ vectors)
        orthogonality = jnp.linalg.norm(
            vectors @ vectors.T - jnp.eye(a_host.shape[0], dtype=vectors.dtype)
        )
        reconstruction = jnp.linalg.norm(vectors.T @ diag @ vectors - a)
        residual_local = _addressable_values(
            jnp.asarray(
                [eigen_residual, orthogonality, reconstruction],
                dtype=jnp.float64,
            )
        )
        if not np.all(np.isfinite(residual_local)) or np.max(residual_local) > 1e-5:
            _emit(
                "fail",
                check="syevd_vector_residual",
                case=case_name,
                dtype=str(dtype),
                n=int(a_host.shape[0]),
                tile_size=int(tile_size),
                got=residual_local.tolist(),
                expected=[0.0, 0.0, 0.0],
            )
            return False

    return True


def main():
    try:
        for dtype in (np.float64, np.complex64):
            n = num_procs * 2
            diag = np.arange(1, n + 1, dtype=np.float64).astype(dtype)
            a_diag = np.diag(diag)
            if not _run_case(
                "diagonal_no_padding_vectors",
                dtype,
                a_diag,
                tile_size=1,
                return_eigenvectors=True,
            ):
                return
            if not _run_case(
                "diagonal_no_padding_no_vectors",
                dtype,
                a_diag,
                tile_size=1,
                return_eigenvectors=False,
            ):
                return

            padded_n = num_procs * 3
            padded_diag = np.arange(1, padded_n + 1, dtype=np.float64).astype(
                dtype
            )
            a_padded_diag = np.diag(padded_diag)
            if not _run_case(
                "diagonal_with_padding_vectors",
                dtype,
                a_padded_diag,
                tile_size=2,
                return_eigenvectors=True,
            ):
                return
            if not _run_case(
                "diagonal_with_padding_no_vectors",
                dtype,
                a_padded_diag,
                tile_size=2,
                return_eigenvectors=False,
            ):
                return

        spd_n = num_procs * 3
        a_spd = _spd_matrix(spd_n, np.float64)
        if not _run_case(
            "non_diagonal_spd_vectors",
            np.float64,
            a_spd,
            tile_size=1,
            return_eigenvectors=True,
        ):
            return
        if not _run_case(
            "non_diagonal_spd_no_vectors",
            np.float64,
            a_spd,
            tile_size=1,
            return_eigenvectors=False,
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
