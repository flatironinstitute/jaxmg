# jaxmg.lu_solve

`lu_solve` is the high-level solver for general nonsingular matrices. It uses
cuSOLVERMp `Getrf` followed by `Getrs`, including distributed pivot storage and
the same fused memory-distribution path used by `potrs`.

Use `lu_solve_jit_ctx` when the solve is part of a larger caller-owned
`jax.jit` computation. Inside an existing `jax.shard_map`, use
`lu_solve_shardmap_ctx` with local matrix and RHS blocks instead. Both use the
same distributed solver.

!!! Note "Context API migration"

    The previous global-array `lu_solve_shardmap_ctx` is now named
    `lu_solve_jit_ctx`, with the same arguments and outputs.

::: jaxmg.lu_solve

::: jaxmg.lu_solve_jit_ctx

::: jaxmg.lu_solve_shardmap_ctx
