# Runnable examples

These examples run JAXMg on any even number of GPUs arranged as a
\((P/2)\times2\) process grid, where \(P\) is the number of Python processes.

The high-level examples construct sharded inputs and call the corresponding
JAXMg solver:

- `run_potrs_example.py`
- `run_lu_solve_example.py`
- `run_syevd_example.py`
- `run_gesvd_example.py`

The context examples construct the inputs and execute the solver inside one
caller-owned `jax.jit`:

- `run_potrs_context_example.py`
- `run_lu_solve_context_example.py`
- `run_syevd_context_example.py`
- `run_gesvd_context_example.py`

To run an example on a local node with an even number of GPUs:

```bash
script=examples/run_potrs_example.py
num_processes=4

for ((rank=0; rank<num_processes; rank++)); do
  CUDA_VISIBLE_DEVICES=$rank python -u "$script" \
    --coordinator 127.0.0.1:12345 \
    --process-id "$rank" \
    --num-processes "$num_processes" &
done
wait
```

Replace `script` with any of the files above. Rank 0 prints a final
correctness result:

```text
POTRS solution correct: True
```
