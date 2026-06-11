# jaxmg.potrs_mp

For multi-node runs, initialize JAX before constructing devices, arrays, or
meshes:

```python
import jaxmg

jaxmg.initialize_node_process()
mesh = jaxmg.make_cusolvermp_mesh(process_rows=2, process_cols=4)
```

The mesh uses row-major rank order, matching the native cuSOLVERMp descriptor:

```text
rank = process_row * process_cols + process_col
```

::: jaxmg.potrs_mp
