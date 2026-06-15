# Distributed Helpers

These helpers remain available for compatibility with earlier cuSOLVERMp
experiments. New cuSOLVERMp code should call `jax.distributed.initialize()`
directly and build meshes with ordinary JAX APIs such as `jax.make_mesh`.

::: jaxmg.initialize_node_process
