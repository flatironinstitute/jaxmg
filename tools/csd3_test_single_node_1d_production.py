#!/usr/bin/env python3
"""Single-node production smoke test for the XLA communicator JAXMg backend."""

from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import jaxmg


def _make_mesh(num_devices: int) -> Mesh:
    devices = np.asarray(jax.devices()[:num_devices])
    if devices.size != num_devices:
        raise RuntimeError(
            f"Requested {num_devices} devices, but JAX only sees {devices.size}: "
            f"{jax.devices()}"
        )
    return Mesh(devices, ("x",))


def _put(array: np.ndarray, mesh: Mesh, spec: P):
    return jax.device_put(jnp.asarray(array), NamedSharding(mesh, spec))


def _spd_matrix(rng: np.random.Generator, n: int, dtype: np.dtype) -> np.ndarray:
    x = rng.standard_normal((n, n)).astype(dtype)
    return (x @ x.T + n * np.eye(n, dtype=dtype)).astype(dtype)


def _check_status(name: str, status) -> None:
    value = int(np.asarray(status))
    print(f"{name}: status={value}", flush=True)
    if value != 0:
        raise RuntimeError(f"{name} returned nonzero status {value}")


def run_cyclic(mesh: Mesh, n: int, tile: int, dtype: np.dtype) -> None:
    base = np.arange(n * n, dtype=dtype).reshape(n, n)
    a = _put(base, mesh, P("x", None))
    out = jaxmg.cyclic_1d(a, tile, mesh, P("x", None), pad=False)
    out.block_until_ready()
    jaxmg.verify_cyclic(jnp.asarray(base), out, tile)
    print(f"cyclic: n={n} tile={tile} dtype={dtype} PASS", flush=True)


def run_potrs(mesh: Mesh, n: int, tile: int, dtype: np.dtype) -> None:
    rng = np.random.default_rng(1001)
    a_host = _spd_matrix(rng, n, dtype)
    b_host = rng.standard_normal((n, 3)).astype(dtype)
    a = _put(a_host, mesh, P("x", None))
    b = _put(b_host, mesh, P(None, None))
    out, status = jaxmg.potrs(
        a, b, tile, mesh, P("x", None), return_status=True, pad=False
    )
    out.block_until_ready()
    _check_status("potrs", status)
    ref = np.linalg.solve(a_host.astype(np.float64), b_host.astype(np.float64))
    np.testing.assert_allclose(np.asarray(out), ref, rtol=3e-3, atol=3e-3)
    print(f"potrs: n={n} tile={tile} dtype={dtype} PASS", flush=True)


def run_potri(mesh: Mesh, n: int, tile: int, dtype: np.dtype) -> None:
    rng = np.random.default_rng(1002)
    a_host = _spd_matrix(rng, n, dtype)
    a = _put(a_host, mesh, P("x", None))
    out, status = jaxmg.potri(
        a, tile, mesh, P("x", None), return_status=True, pad=False
    )
    out.block_until_ready()
    _check_status("potri", status)
    ref = np.linalg.inv(a_host.astype(np.float64))
    np.testing.assert_allclose(np.asarray(out), ref, rtol=7e-3, atol=7e-3)
    print(f"potri: n={n} tile={tile} dtype={dtype} PASS", flush=True)


def run_syevd(mesh: Mesh, n: int, tile: int, dtype: np.dtype) -> None:
    rng = np.random.default_rng(1003)
    a_host = _spd_matrix(rng, n, dtype)
    a = _put(a_host, mesh, P("x", None))
    evals, status = jaxmg.syevd(
        a,
        tile,
        mesh,
        P("x", None),
        return_eigenvectors=False,
        return_status=True,
        pad=False,
    )
    evals.block_until_ready()
    _check_status("syevd_no_V", status)
    ref_evals, _ = np.linalg.eigh(a_host.astype(np.float64))
    np.testing.assert_allclose(np.asarray(evals), ref_evals, rtol=2e-3, atol=2e-3)

    a = _put(a_host, mesh, P("x", None))
    evals, vecs, status = jaxmg.syevd(
        a,
        tile,
        mesh,
        P("x", None),
        return_eigenvectors=True,
        return_status=True,
        pad=False,
    )
    evals.block_until_ready()
    vecs.block_until_ready()
    _check_status("syevd", status)
    evals_host = np.asarray(evals)
    np.testing.assert_allclose(evals_host, ref_evals, rtol=2e-3, atol=2e-3)

    vectors = np.asarray(vecs)
    residual = np.linalg.norm(a_host @ vectors - vectors * evals_host[None, :])
    transposed_residual = np.linalg.norm(
        a_host @ vectors.T - vectors.T * evals_host[None, :]
    )
    residual = min(residual, transposed_residual)
    scale = np.linalg.norm(a_host) * max(np.sqrt(n), 1.0)
    if residual / scale > 5e-3:
        raise AssertionError(f"syevd eigenvector residual too large: {residual / scale}")
    print(f"syevd: n={n} tile={tile} dtype={dtype} PASS", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--eig-n", type=int, default=128)
    parser.add_argument("--tile", type=int, default=64)
    args = parser.parse_args()

    if args.n % args.devices != 0 or (args.n // args.devices) % args.tile != 0:
        raise ValueError("n/devices must be divisible by tile for pad=False tests")
    if args.eig_n % args.devices != 0 or (args.eig_n // args.devices) % args.tile != 0:
        raise ValueError("eig-n/devices must be divisible by tile for pad=False tests")

    print(f"python={os.sys.executable}", flush=True)
    print(f"jax={jax.__version__} devices={jax.devices()}", flush=True)
    mesh = _make_mesh(args.devices)
    dtype = np.float32

    run_cyclic(mesh, args.n, args.tile, dtype)
    run_potrs(mesh, args.n, args.tile, dtype)
    run_potri(mesh, args.n, args.tile, dtype)
    run_syevd(mesh, args.eig_n, args.tile, dtype)
    print("PASS: single-node 1D production backend", flush=True)


if __name__ == "__main__":
    main()
