import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxmg import plot_block_to_cyclic
from jax.sharding import PartitionSpec as P, NamedSharding
from jaxmg import syevd, potrs, potrs_shardmap_ctx
from functools import partial


import time
import matplotlib.pyplot as plt

platforms = set(d.platform for d in jax.devices())
if "gpu" not in platforms:
    pytest.skip("No GPUs found. Skipping", allow_module_level=True)
else:

    def test_block_cyclic_data_layout_plot_1():
        N = 100
        T_A = 7
        ndev = 4
        plot_block_to_cyclic(N, T_A, ndev)


    def test_block_cyclic_data_layout_syevd_timing():
        ndev = jax.device_count()
        if ndev != 3:
            pytest.skip(f"Only testing when 4 GPUS are detected, round only {ndev}")

        print(f"Devices: {jax.devices()}")
        # Assumes we have at least one GPU available
        devices = jax.devices("gpu")
        N = 3**7
        print(f"N={N}")
        dtype = jnp.float64
        # Create random
        A = jax.random.normal(key=jax.random.key(100), shape=(N, N), dtype=dtype)
        A = A + A.T
        b = jnp.ones((N, 1), dtype=dtype)
        ndev = len(devices)
        # Make mesh and place data (rows sharded)
        mesh = jax.make_mesh((ndev,), ("x",))
        # Call syevd
        times = []
        tile_sizes = [2, 16, 32, 128, 256]
        for T_A in tile_sizes:
            Adev = jax.device_put(A, NamedSharding(mesh, P("x", None))).block_until_ready()
            start = time.time()
            ev, V, status = syevd(
                A, T_A=T_A, mesh=mesh, in_specs=(P("x", None),), return_status=True
            )
            ev.block_until_ready()
            V.block_until_ready()
            end = time.time()
            times.append(end - start)
            print(f"Status {status} - Elapsed time {end - start:1.5f}[s]")
        plt.plot(tile_sizes, times, marker=".", markersize=10)
        plt.xlabel("T_A")
        plt.ylabel("Time [s]")
        plt.grid()


    def test_block_cyclic_data_layout_plot_2():
        N = 3**9
        T_A = 3**6
        ndev = 3
        plot_block_to_cyclic(N, T_A, ndev)


    def test_potrs_examples():
        print(f"Devices: {jax.devices()}")
        # Assumes we have at least one GPU available
        ndev = jax.device_count()
        N = 4 * ndev
        T_A = 3
        dtype = jnp.float64
        # Create diagonal matrix and `b` all equal to one
        A = jnp.diag(jnp.arange(N, dtype=dtype) + 1)
        b = jnp.ones((N, 1), dtype=dtype)
        # Make mesh and place data (rows sharded)
        mesh = jax.make_mesh((ndev,), ("x",))
        A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
        b = jax.device_put(b, NamedSharding(mesh, P(None, None)))
        # Call potrs
        out = potrs(A, b, T_A=T_A, mesh=mesh, in_specs=(P("x", None),))
        print(out.block_until_ready())
        expected_out = 1.0 / (jnp.arange(N, dtype=dtype) + 1)
        assert jnp.allclose(out.flatten(), expected_out)


    def test_potrs_examples_shm_ctx():
        print(f"Devices: {jax.devices()}")

        def shard_mapped_fn(_a, _b, _T_A):
            _a = _a * (
                jax.lax.axis_index("x") + 1
            )  # Multiply each shard with the axis number
            return potrs_shardmap_ctx(_a, _b, _T_A)

        def my_fn(_a, _b, _T_A):
            out = jax.shard_map(
                partial(shard_mapped_fn, _T_A=_T_A),
                mesh=mesh,
                in_specs=(P("x", None), P(None, None)),
                out_specs=(P(None, None), P(None)),  # we always return a status.
                check_vma=False,
            )(_a, _b)
            return out

        # Assumes we have at least one GPU available
        ndev = jax.device_count()
        N = 6 * ndev
        T_A = 1
        dtype = jnp.float32
        # Create diagonal matrix
        _A = jnp.eye(N, dtype=dtype)
        _b = jnp.ones((N, 1), dtype=dtype)
        # Make mesh and place data (rows sharded)
        mesh = jax.make_mesh((ndev,), ("x",))
        A = jax.device_put(_A, NamedSharding(mesh, P("x", None)))
        b = jax.device_put(_b, NamedSharding(mesh, P(None, None)))
        # Call potrs
        out, status = my_fn(A, b, T_A)
        print(out.block_until_ready())
        print(f"Solver status: {status}")
        # compute residual on host
        b_target = jnp.concatenate([jnp.ones((6,1), dtype=dtype) / i for i in range(1, ndev + 1)], axis=0)
        assert jnp.allclose(out, b_target, rtol=10, atol=1e-5)
