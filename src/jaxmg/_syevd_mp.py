"""Public cuSOLVERMp symmetric/Hermitian eigensolver wrapper.

``syevd_mp`` follows the same data-layout pipeline as ``potrs_mp``:

1. validate that the input matrix is sharded over a 2D JAX process grid;
2. pad each local shard so both axes have tile-aligned capacity;
3. redistribute the padded JAX block-sharded layout into cuSOLVERMp's 2D
   block-cyclic, column-major local layout;
4. call cuSOLVERMp ``syevd`` through the XLA-owned NCCL communicator; and
5. if eigenvectors were requested, reverse-redistribute them back to the
   original JAX-facing block-sharded layout and remove local padding.

Eigenvalues are returned replicated. Eigenvectors, when requested, use the same
2D JAX sharding as the input matrix. The cuSOLVERMp no-vector path is still
under validation for the 0.7.2 runtime used in current CSD3 testing; keep
``eigvecs=False`` experimental until that runtime behavior is resolved.
"""

from __future__ import annotations

from typing import Tuple, Union

import jax
import jax.numpy as jnp
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ._block_cyclic_2d_execute import (
    required_padded_block_cyclic_2d_scratch_size,
)
from ._block_cyclic_2d_plan import TileShape
from ._cusolvermp import cusolvermp_syevd_shardmap
from ._potrs_mp import (
    _infer_mesh_and_matrix_specs,
    _pad_block_sharded_2d,
    _process_rank_map_from_mesh,
    _redistribute_from_cusolvermp,
    _redistribute_to_cusolvermp,
    _status_specs,
    _unpad_block_sharded_2d,
    _validate_2d_matrix_specs,
)
from ._setup import ensure_init_jaxmg_backend


def _require_rank_per_gpu_process_model(*, caller: str) -> None:
    """Reject launch modes that violate cuSOLVERMp SYEVD's process contract.

    NVIDIA documents cuSOLVERMp around a one-process-per-GPU model: the
    cuSOLVERMp handle is tied to one CUDA device, and the NCCL communicator
    passed to cuSOLVERMp represents process ranks. The host-generated NVIDIA
    samples follow that model, and the SYEVD path is sensitive to it because it
    uses additional internal collectives.

    Ordinary JAX distributed setup remains user-owned. This check only inspects
    the runtime that JAX has already created and gives an early, explicit error
    when one Python process is controlling multiple local GPUs.
    """
    local_device_count = int(jax.local_device_count())
    if local_device_count != 1:
        raise RuntimeError(
            f"{caller} requires a cuSOLVERMp rank-per-GPU launch: each Python "
            "process must own exactly one local GPU. Call "
            "jax.distributed.initialize(..., local_device_ids=[local_rank]) "
            "or otherwise launch one Python process per visible GPU before "
            "creating the input JAX array. This process currently sees "
            f"{local_device_count} local devices."
        )


def syevd_mp(
    a: Array,
    T_A: int,
    mesh: Mesh | None = None,
    matrix_specs: P | None = None,
    eigvecs: bool = True,
    return_status: bool = False,
    pad: bool = True,
) -> Union[Array, Tuple[Array, Array], Tuple[Array, Array, Array]]:
    """Compute eigenvalues/eigenvectors with cuSOLVERMp on a 2D process grid.

    Multi-node ``syevd_mp`` follows cuSOLVERMp's rank-per-GPU process model:
    initialize JAX so each Python process owns exactly one participating GPU.
    The JAX distributed setup remains user-owned; JAXMg validates the resulting
    runtime before entering the native cuSOLVERMp call.

    Args:
        a: Square symmetric/Hermitian matrix, normally sharded with
            ``NamedSharding(mesh, P(row_axis, col_axis))``.
        T_A: Square cuSOLVERMp tile size. JAXMg currently uses
            ``MB_A == NB_A == T_A``.
        mesh: Optional JAX mesh override. If omitted, inferred from
            ``a.sharding.mesh``.
        matrix_specs: Optional partition spec override. If omitted, inferred
            from ``a.sharding.spec``.
        eigvecs: If true, return eigenvectors as well as eigenvalues. If false,
            request eigenvalues only. The cuSOLVERMp 0.7.2 no-vector path is
            still under validation and may not be available on all runtimes.
        return_status: If true, append the per-rank native status array.
        pad: If true, add local row/column capacity so every local shard is
            tile-aligned. If false, incompatible local shard shapes raise.

    Returns:
        ``w`` when ``eigvecs=False``; ``(w, v)`` when ``eigvecs=True``. If
        ``return_status=True``, the status array is appended to that tuple.
    """
    if a.ndim != 2:
        raise ValueError("syevd_mp expects a rank-2 matrix A.")
    if a.dtype not in (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128):
        raise TypeError(
            "syevd_mp supports float32, float64, complex64, and complex128."
        )
    if a.shape[0] != a.shape[1]:
        raise ValueError("syevd_mp expects A to be square.")
    if int(T_A) <= 0:
        raise ValueError("T_A must be positive.")

    mesh, matrix_specs = _infer_mesh_and_matrix_specs(
        a,
        mesh=mesh,
        matrix_specs=matrix_specs,
    )
    row_axis, col_axis, grid = _validate_2d_matrix_specs(mesh, matrix_specs)
    rank_map = _process_rank_map_from_mesh(
        mesh,
        row_axis=row_axis,
        col_axis=col_axis,
        grid=grid,
        caller="syevd_mp",
    )
    _require_rank_per_gpu_process_model(caller="syevd_mp")
    scratch_specs = _status_specs(row_axis, col_axis, grid)
    tile_shape = TileShape(rows=int(T_A), cols=int(T_A))

    ensure_init_jaxmg_backend()

    a_padded, (a_local_rows, a_local_cols) = _pad_block_sharded_2d(
        a,
        mesh=mesh,
        matrix_specs=matrix_specs,
        grid=grid,
        tile_shape=tile_shape,
        pad=pad,
    )

    scratch_per_rank = required_padded_block_cyclic_2d_scratch_size(
        logical_rows=a.shape[0],
        logical_cols=a.shape[1],
        grid=grid,
        tile_rows=tile_shape.rows,
        tile_cols=tile_shape.cols,
    )
    scratch = jax.device_put(
        jnp.zeros((grid.num_processes * scratch_per_rank,), dtype=a.dtype),
        NamedSharding(mesh, scratch_specs),
    )

    a_cyclic, scratch = _redistribute_to_cusolvermp(
        a_padded,
        scratch,
        mesh=mesh,
        matrix_specs=matrix_specs,
        scratch_specs=scratch_specs,
        grid=grid,
        rank_map=rank_map,
        tile_shape=tile_shape,
        logical_rows=a.shape[0],
        logical_cols=a.shape[1],
    )

    eigenvalues, vectors_cyclic, status = cusolvermp_syevd_shardmap(
        a_cyclic,
        mesh,
        matrix_specs,
        scratch_specs,
        process_rows=grid.process_rows,
        process_cols=grid.process_cols,
        n=a.shape[0],
        tile_size=tile_shape.rows,
        rank_map=rank_map,
        grid_mapping=rank_map.cusolvermp_grid_mapping,
        compute_vectors=bool(eigvecs),
    )

    if eigvecs:
        vectors_padded, scratch = _redistribute_from_cusolvermp(
            vectors_cyclic,
            scratch,
            mesh=mesh,
            matrix_specs=matrix_specs,
            scratch_specs=scratch_specs,
            grid=grid,
            rank_map=rank_map,
            tile_shape=tile_shape,
            logical_rows=a.shape[0],
            logical_cols=a.shape[1],
        )
        vectors = _unpad_block_sharded_2d(
            vectors_padded,
            mesh=mesh,
            matrix_specs=matrix_specs,
            local_rows=a_local_rows,
            local_cols=a_local_cols,
        )
        if return_status:
            return eigenvalues, vectors, status
        return eigenvalues, vectors

    if return_status:
        return eigenvalues, status
    return eigenvalues
