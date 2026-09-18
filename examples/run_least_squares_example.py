"""Solve an overdetermined least-squares problem on a 2D GPU grid."""

import argparse
import math
import os

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from jaxmg import least_squares


T_A = 128
PROCESS_COLS = 2
TILES_PER_GRID_AXIS = 2
DTYPE = jnp.float64


def main() -> None:
    """Create a distributed rectangular system and validate its solution."""
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
    m = 2 * n

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
    vector_sharding = NamedSharding(mesh, P("pr"))

    # Construct an overdetermined system with the known solution x = 1.
    @jax.jit
    def make_problem():
        diagonal = jnp.arange(1, n + 1, dtype=DTYPE)
        a = jnp.zeros((m, n), dtype=DTYPE)
        a = a.at[jnp.arange(n), jnp.arange(n)].set(diagonal)
        a = jax.reshard(a, matrix_sharding)
        b = jax.reshard(a @ jnp.ones((n,), dtype=DTYPE), vector_sharding)
        return a, b

    a, b = make_problem()

    # Run the distributed JAXMg least-squares solver.
    x = least_squares(a, b, T_A=T_A, mesh=mesh, matrix_specs=matrix_specs)
    x.block_until_ready()

    # Validate the result against the known solution.
    correct = jnp.allclose(x, jnp.ones((n,), dtype=DTYPE))
    correct.block_until_ready()

    if jax.process_index() == 0:
        print("Least-squares solution correct:", bool(correct))


if __name__ == "__main__":
    main()
