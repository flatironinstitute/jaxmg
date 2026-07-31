# jaxmg.lu_solve

`lu_solve` is the high-level solver for general nonsingular matrices. It uses
cuSOLVERMp `Getrf` followed by `Getrs`, including distributed pivot storage and
the same fused memory-distribution path used by `potrs`.

Use `lu_solve_shardmap_ctx` when the solve is part of a larger caller-owned
`jax.jit` computation.

::: jaxmg.lu_solve

::: jaxmg.lu_solve_shardmap_ctx
