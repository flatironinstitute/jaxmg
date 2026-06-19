# Single Process Multiple Devices (SPMD)

When `potrs.cu` is called in a `jax.shard_map` context through the `jax.ffi` API with a single process for multiple devices,

```python
_out, status = jax.ffi.ffi_call(
            "potrs_mg",
            out_type,
            input_layouts=input_layouts,
            output_layouts=output_layouts,
        )(_a, _b, T_A=T_A)
```

a thread will spawn for each available GPU that executes the code in `potrs.cu`. Each thread will only have access to its local shard in GPU memory through a device pointer. The `cuSolverMgPotrf` API must be called in a single thread and requires an array of **all** device pointers containing the shards on each GPU. 

This raises the following two issues.

1. We need to synchronise the threads to set up `cuSolverMgPotrf` and the data. We then need to execute the solver in thread 0 and have the other threads wait for it to finish. However, JAX has spawned the threads and we do not have any explict control over the thread  syncronization.
2. Since each thread only has access to its local shard, we need to somehow make thread 0 aware of the device pointers across all other threads.

We solver the first issue by initializing a global barrier via `std::unique_ptr<std::barrier<>> barrier_ptr`. Here `std::unique_ptr` takes care of deleting the barrier when it goes out of scope (when the FFI call finishes). Then, in `potrs.cu` we use 
```C++
static std::once_flag barrier_initialized;
std::call_once(barrier_initialized, [&](){ sync_point.initialize(nbGpus); });
```
to initialize the barrier across all threads. The `std::once_flag` ensures that the barrier is initialized exactly once so that all threads see the same barrier.

We share device pointers between threads through the creation of shared memory:

```cpp
data_type **shmA = get_shm_device_ptrs<data_type>(currentDevice,
                                                  sync_point, 
                                                  shminfoA, 
                                                  "shmA"); 
```

In each thread, we then assign the device pointer of the local shard to this shared memory:

```cpp
shmA[currentDevice] = array_data_A;
```

which we can safely pass to `cuSolverMgPotrf`:
```cpp
cusolver_status = cusolverMgPotrs(cusolverH, CUBLAS_FILL_MODE_LOWER, N,NRHS, 
                                  reinterpret_cast<void **>(shmA), 
                                  IA, JA, descrA,
                                  reinterpret_cast<void **>(shmB), 
                                  IB, JB, descrB,
                                  compute_type,
                                  reinterpret_cast<void **>(shmwork), 
                                  *shmlwork,
                                  &info);
```
