"""Compute an eigendecomposition on a two-column GPU process grid."""

import argparse
import os

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from jaxmg import syevd


T_A = 128
PROCESS_COLS = 2
TILE_ROWS_PER_PROCESS = 4
DTYPE = jnp.float64


def main() -> None:
    """Create the distributed matrix, solve it, and check the eigenpairs."""
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

    # Construct the matrix directly in its distributed layout.
    @jax.jit
    def make_matrix():
        diagonal = jnp.arange(1, N + 1, dtype=DTYPE)
        return jax.reshard(
            jnp.diag(diagonal),
            matrix_sharding,
        )

    a = make_matrix()

    # Run the distributed JAXMg eigensolver.
    eigenvalues, eigenvectors = syevd(
        a,
        T_A=T_A,
        mesh=mesh,
        matrix_specs=matrix_specs,
    )
    eigenvectors.block_until_ready()

    # Validate the result against the known solution.
    diagonal = jnp.arange(1, N + 1, dtype=DTYPE)
    residual = diagonal[:, None] * eigenvectors
    residual = residual - eigenvectors * eigenvalues[None, :]
    correct = jnp.allclose(eigenvalues, diagonal) & jnp.allclose(
        residual,
        0.0,
        atol=1e-8,
    )
    correct.block_until_ready()

    if jax.process_index() == 0:
        print("SYEVD eigenpairs correct:", bool(correct))


if __name__ == "__main__":
    main()
