"""Embed a general linear solve in a caller-owned JAX compilation."""

import argparse
import os

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from jaxmg import lu_solve_shardmap_ctx


T_A = 128
PROCESS_COLS = 2
TILE_ROWS_PER_PROCESS = 4
DTYPE = jnp.float64


def main() -> None:
    """Build and solve the distributed system inside one jitted function."""
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
    N = T_A * TILE_ROWS_PER_PROCESS * process_rows

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

    # The context interface lets input construction and the solve compile together.
    @jax.jit
    def build_and_solve():
        diagonal = jnp.arange(1, N + 1, dtype=DTYPE) + N
        a = jnp.diag(diagonal)
        a = a + jnp.diag(jnp.full((N - 1,), 0.25, dtype=DTYPE), k=1)
        a = jax.reshard(a, matrix_sharding)
        b = jax.reshard(
            a @ jnp.ones((N,), dtype=DTYPE),
            vector_sharding,
        )

        # Run the JAXMg solver within the caller-owned JIT.
        _, x, _ = lu_solve_shardmap_ctx(
            a,
            b,
            T_A=T_A,
            mesh=mesh,
            matrix_specs=matrix_specs,
        )
        return x

    x = build_and_solve()
    x.block_until_ready()

    # Validate the result against the known solution.
    correct = jnp.allclose(x, jnp.ones((N,), dtype=DTYPE))
    correct.block_until_ready()

    if jax.process_index() == 0:
        print("LU context solution correct:", bool(correct))


if __name__ == "__main__":
    main()
