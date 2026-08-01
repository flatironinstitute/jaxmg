"""Embed a rectangular singular-value decomposition in one JAX compilation."""

import argparse
import math
import os

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from jaxmg import gesvd_shardmap_ctx


T_A = 128
PROCESS_COLS = 2
TILES_PER_GRID_AXIS = 2
DTYPE = jnp.float64


def main() -> None:
    """Construct and decompose a distributed matrix inside a caller-owned JIT."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coordinator",
        default=os.environ.get("JAXMG_COORD", "127.0.0.1:12345"),
    )
    parser.add_argument(
        "--process-id", type=int, default=os.environ.get("JAXMG_PROCESS_ID")
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=int(os.environ.get("JAXMG_NUM_PROCS", "4")),
    )
    parser.add_argument(
        "--local-device-id",
        type=int,
        default=int(os.environ.get("JAXMG_LOCAL_DEVICE_ID", "0")),
    )
    args = parser.parse_args()
    if args.process_id is None:
        parser.error("--process-id or JAXMG_PROCESS_ID is required")
    if args.num_processes < PROCESS_COLS or args.num_processes % PROCESS_COLS:
        parser.error("this example requires an even number of Python processes")

    process_rows = args.num_processes // PROCESS_COLS
    base = T_A * TILES_PER_GRID_AXIS * math.lcm(process_rows, PROCESS_COLS)
    m, n = 2 * base, base

    # Allow JAX to discover the global process and GPU ranks.
    jax.distributed.initialize(
        coordinator_address=args.coordinator,
        num_processes=args.num_processes,
        process_id=args.process_id,
        local_device_ids=[args.local_device_id],
    )

    # Initialize the (num_processes / 2) x 2 GPU process mesh.
    mesh = jax.make_mesh((process_rows, PROCESS_COLS), ("pr", "pc"))
    jax.set_mesh(mesh)
    matrix_specs = P("pr", "pc")
    matrix_sharding = NamedSharding(mesh, matrix_specs)

    # The context interface lets matrix construction and GESVD compile together.
    @jax.jit
    def build_and_decompose():
        expected_singular_values = jnp.linspace(2.0, 1.0, n, dtype=DTYPE)
        a = jnp.zeros((m, n), dtype=DTYPE)
        a = a.at[jnp.arange(n), jnp.arange(n)].set(expected_singular_values)
        a = jax.reshard(a, matrix_sharding)

        # a_work stays internal because A is created inside this function.
        _, u, singular_values, vh, status = gesvd_shardmap_ctx(
            a,
            T_A=T_A,
            mesh=mesh,
            matrix_specs=matrix_specs,
        )
        return u, singular_values, vh, status, expected_singular_values

    u, singular_values, vh, status, expected_singular_values = (
        build_and_decompose()
    )
    vh.block_until_ready()

    # Validate the singular values against the known solution.
    correct = (
        u.shape == (m, n)
        and vh.shape == (n, n)
        and jnp.all(status == 0)
        and jnp.allclose(singular_values, expected_singular_values)
    )
    if hasattr(correct, "block_until_ready"):
        correct.block_until_ready()
    if jax.process_index() == 0:
        print("GESVD context decomposition correct:", bool(correct))


if __name__ == "__main__":
    main()
