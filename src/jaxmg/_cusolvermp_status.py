"""Private cuSOLVERMp native status-vector schemas.

The C++ cuSOLVERMp handlers return a small integer status vector from each
participating rank.  These vectors are diagnostic backend detail rather than
public JAXMg API, but Python must still declare their fixed lengths when it
builds the corresponding ``jax.ffi.ffi_call`` result types.

The field order and vector lengths must match the corresponding POTRS,
LU-solve, SYEVD, GESVD, polar, least-squares, and QR status writers in
``src/cuda/cusolvermp_routines``.
"""

from __future__ import annotations


# Mirrors kPotrsStatusSize/status_words in
# src/cuda/cusolvermp_routines/cusolvermp_potrs.cc.
_CUSOLVERMP_POTRS_STATUS_FIELDS = (
    "status_code",
    "cuda_device",
    "nccl_rank",
    "nccl_rank_count",
    "process_rows",
    "process_cols",
    "cusolvermp_version",
    "cusolvermp_runtime_available",
    "handle_created",
    "grid_created",
    "a_descriptor_created",
    "raw_cusolver_status",
    "a_size_bytes",
    "n",
    "tile_size",
    "a_local_rows",
    "a_local_cols",
    "b_local_rows",
    "a_numroc_rows",
    "a_numroc_cols",
    "b_numroc_rows",
    "b_numroc_cols",
    "potrf_device_workspace_kib",
    "potrf_host_workspace_kib",
    "potrs_device_workspace_kib",
    "potrs_host_workspace_kib",
    "potrf_called",
    "potrf_info",
    "potrs_called",
    "potrs_info",
    "a_native_redist",
    "b_native_redist",
    "b_reverse_redist",
    "logdet_computed",
    "dtype_code",
    "b_local_cols",
    "nrhs",
    "a_native_redist_final",
    "b_native_redist_final",
    "grid_mapping",
)


# Mirrors kLuSolveStatusSize/status_words in
# src/cuda/cusolvermp_routines/cusolvermp_lu_solve.cc.
_CUSOLVERMP_LU_SOLVE_STATUS_FIELDS = (
    "status_code",
    "cuda_device",
    "nccl_rank",
    "nccl_rank_count",
    "process_rows",
    "process_cols",
    "cusolvermp_version",
    "cusolvermp_runtime_available",
    "handle_created",
    "grid_created",
    "a_descriptor_created",
    "raw_cusolver_status",
    "a_size_bytes",
    "n",
    "tile_size",
    "a_local_rows",
    "a_local_cols",
    "b_local_rows",
    "a_numroc_rows",
    "a_numroc_cols",
    "b_numroc_rows",
    "b_numroc_cols",
    "getrf_device_workspace_kib",
    "getrf_host_workspace_kib",
    "getrs_device_workspace_kib",
    "getrs_host_workspace_kib",
    "getrf_called",
    "getrf_info",
    "getrs_called",
    "getrs_info",
    "a_native_redist",
    "b_native_redist",
    "b_reverse_redist",
    "ipiv_len",
    "dtype_code",
    "b_local_cols",
    "nrhs",
    "a_native_redist_final",
    "b_native_redist_final",
    "grid_mapping",
    "ipiv_bytes_kib",
)


# Mirrors kSyevdStatusSize/status_words in
# src/cuda/cusolvermp_routines/cusolvermp_syevd.cc.
_CUSOLVERMP_SYEVD_STATUS_FIELDS = (
    "status_code",
    "cuda_device",
    "nccl_rank",
    "nccl_rank_count",
    "process_rows",
    "process_cols",
    "cusolvermp_version",
    "cusolvermp_runtime_available",
    "handle_created",
    "grid_created",
    "a_descriptor_created",
    "raw_cusolver_status",
    "a_size_bytes",
    "n",
    "tile_size",
    "a_local_rows",
    "a_local_cols",
    "a_numroc_rows",
    "a_numroc_cols",
    "eigenvalues_size_bytes",
    "compute_eigenvectors",
    "syevd_device_workspace_kib",
    "syevd_host_workspace_kib",
    "syevd_called",
    "syevd_info",
    "dtype_code",
    "grid_mapping",
    "q_descriptor_created",
    "a_native_redist",
    "reserved_0",
    "reserved_1",
    "reserved_2",
    "reserved_3",
    "reserved_4",
    "reserved_5",
    "reserved_6",
)


# Mirrors kGesvdStatusSize/status_words in
# src/cuda/cusolvermp_routines/cusolvermp_gesvd.cc.
_CUSOLVERMP_GESVD_STATUS_FIELDS = (
    "status_code",
    "cuda_device",
    "nccl_rank",
    "nccl_rank_count",
    "process_rows",
    "process_cols",
    "cusolvermp_version",
    "cusolvermp_runtime_available",
    "handle_created",
    "grid_created",
    "a_descriptor_created",
    "raw_cusolver_status",
    "a_size_bytes",
    "m",
    "n",
    "tile_size",
    "a_local_rows",
    "a_local_cols",
    "a_numroc_rows",
    "a_numroc_cols",
    "singular_values_size_bytes",
    "compute_u",
    "compute_vh",
    "full_matrices",
    "gesvd_descriptor_created",
    "u_descriptor_created",
    "vh_descriptor_created",
    "u_numroc_rows",
    "u_numroc_cols",
    "vh_numroc_rows",
    "vh_numroc_cols",
    "gesvd_device_workspace_kib",
    "gesvd_host_workspace_kib",
    "gesvd_called",
    "gesvd_info",
    "singular_values_found",
    "dtype_code",
    "grid_mapping",
    "reserved_0",
    "reserved_1",
    "reserved_2",
    "reserved_3",
)


# Mirrors kPolarStatusSize/status_words in
# src/cuda/cusolvermp_routines/cusolvermp_polar.cc.
_CUSOLVERMP_POLAR_STATUS_FIELDS = (
    "status_code",
    "cuda_device",
    "nccl_rank",
    "nccl_rank_count",
    "process_rows",
    "process_cols",
    "cusolvermp_version",
    "cusolvermp_runtime_available",
    "handle_created",
    "grid_created",
    "a_descriptor_created",
    "raw_cusolver_status",
    "a_size_bytes",
    "m",
    "n",
    "tile_size",
    "a_local_rows",
    "a_local_cols",
    "a_numroc_rows",
    "a_numroc_cols",
    "compute_h",
    "h_descriptor_created",
    "h_numroc_rows",
    "h_numroc_cols",
    "polar_device_workspace_kib",
    "polar_host_workspace_kib",
    "polar_called",
    "polar_info",
    "dtype_code",
    "grid_mapping",
    "reserved_0",
    "reserved_1",
)


# Mirrors kGelsStatusSize/status_words in
# src/cuda/cusolvermp_routines/cusolvermp_gels.cc.
_CUSOLVERMP_LEAST_SQUARES_STATUS_FIELDS = (
    "status_code",
    "cuda_device",
    "nccl_rank",
    "nccl_rank_count",
    "process_rows",
    "process_cols",
    "cusolvermp_version",
    "cusolvermp_runtime_available",
    "handle_created",
    "grid_created",
    "a_descriptor_created",
    "raw_cusolver_status",
    "a_size_bytes",
    "m",
    "n",
    "tile_size",
    "a_local_rows",
    "a_local_cols",
    "b_local_rows",
    "b_local_cols",
    "a_numroc_rows",
    "a_numroc_cols",
    "b_numroc_rows",
    "b_numroc_cols",
    "gels_device_workspace_kib",
    "gels_host_workspace_kib",
    "gels_called",
    "gels_info",
    "a_native_redist",
    "b_native_redist",
    "b_reverse_redist",
    "dtype_code",
    "nrhs",
    "grid_mapping",
    "b_size_bytes",
)


# Mirrors kQrStatusSize/status_words in
# src/cuda/cusolvermp_routines/cusolvermp_qr.cc.
_CUSOLVERMP_QR_STATUS_FIELDS = (
    "status_code",
    "cuda_device",
    "nccl_rank",
    "nccl_rank_count",
    "process_rows",
    "process_cols",
    "cusolvermp_version",
    "cusolvermp_runtime_available",
    "handle_created",
    "grid_created",
    "q_descriptor_created",
    "r_descriptor_created",
    "raw_cusolver_status",
    "a_size_bytes",
    "m",
    "n",
    "tile_size",
    "q_local_rows",
    "q_local_cols",
    "q_numroc_rows",
    "q_numroc_cols",
    "r_numroc_rows",
    "r_numroc_cols",
    "tau_elements",
    "geqrf_device_workspace_kib",
    "geqrf_host_workspace_kib",
    "orgqr_device_workspace_kib",
    "orgqr_host_workspace_kib",
    "geqrf_called",
    "geqrf_info",
    "r_extracted",
    "orgqr_called",
    "orgqr_info",
    "dtype_code",
    "grid_mapping",
    "reserved_0",
)


_CUSOLVERMP_POTRS_STATUS_SIZE = len(_CUSOLVERMP_POTRS_STATUS_FIELDS)
_CUSOLVERMP_LU_SOLVE_STATUS_SIZE = len(_CUSOLVERMP_LU_SOLVE_STATUS_FIELDS)
_CUSOLVERMP_SYEVD_STATUS_SIZE = len(_CUSOLVERMP_SYEVD_STATUS_FIELDS)
_CUSOLVERMP_GESVD_STATUS_SIZE = len(_CUSOLVERMP_GESVD_STATUS_FIELDS)
_CUSOLVERMP_POLAR_STATUS_SIZE = len(_CUSOLVERMP_POLAR_STATUS_FIELDS)
_CUSOLVERMP_LEAST_SQUARES_STATUS_SIZE = len(
    _CUSOLVERMP_LEAST_SQUARES_STATUS_FIELDS
)
_CUSOLVERMP_QR_STATUS_SIZE = len(_CUSOLVERMP_QR_STATUS_FIELDS)
