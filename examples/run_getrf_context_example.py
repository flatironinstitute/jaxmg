"""Embed a pivoted LU factorization in one JAX compilation."""

import argparse
import math
import os

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from jaxmg import getrf_shardmap_ctx


T_A = 128
PROCESS_COLS = 2
TILES_PER_GRID_AXIS = 2
DTYPE = jnp.float64


def main() -> None:
    """Construct and factor a distributed matrix in a caller-owned JIT."""
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
    n = T_A * TILES_PER_GRID_AXIS * math.lcm(process_rows, PROCESS_COLS)

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

    @jax.jit
    def build_and_factor():
        """Construct A and compute its LU factorization in one compiled function."""
        # A reverses the rows of L0 @ U0, where L0 is unit lower triangular with
        # multipliers 0.5. Partial pivoting then recovers L0 and U0 exactly, and
        # the row permutation it applies is the reversal.
        lower = jnp.eye(n, dtype=DTYPE) + jnp.tril(
            jnp.full((n, n), 0.5, dtype=DTYPE), -1
        )
        upper = jnp.diag(jnp.linspace(1.0, 2.0, n, dtype=DTYPE)) + jnp.triu(
            jnp.full((n, n), 0.25, dtype=DTYPE), 1
        )
        a = (lower @ upper)[::-1]
        expected_lu = jnp.tril(lower, -1) + upper
        a = jax.reshard(a, matrix_sharding)
        expected_lu = jax.reshard(expected_lu, matrix_sharding)

        # The packed LU factor directly reuses A because GETRF overwrites it.
        lu, piv, _ = getrf_shardmap_ctx(a, T_A=T_A)
        permutation = jax.lax.linalg.lu_pivots_to_permutation(piv, n)
        return jnp.allclose(lu, expected_lu) & jnp.array_equal(
            permutation, jnp.arange(n - 1, -1, -1, dtype=permutation.dtype)
        )

    correct = build_and_factor()
    correct.block_until_ready()
    if jax.process_index() == 0:
        print("LU context factorization correct:", bool(correct))


if __name__ == "__main__":
    main()
