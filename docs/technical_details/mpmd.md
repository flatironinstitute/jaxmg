# Multiple Process Multiple Devices (MPMD)

In a multi-process context it is not as straightforward to setup memory sharing between processes, especially when it comes to passing around device pointers which are bound to a specific CUDA context. 

The solution used here is to make use of the cudaIPC documentation, which allows one to export handles to device memory to
different processes. In `potrs_mp.cu`, we achieve this again through shared memory, although now we share the cudaIPC memory
handles:

```cpp
ipcGetHandleAndOffset(array_data_A, 
                      shmAipc[currentDevice], 
                      shmoffsetA[currentDevice]);
```

A significant complication is that JAX' memory allocation is managed by XLA, which means that device pointers are actually
base pointers together with some offset. cudaIPC only exports the base-pointer, so we have to manually pass around the 
offset and extract the true pointer:

```cpp
opened_ptrs_A = ipcGetDevicePointers<data_type>(currentDevice, 
                                                nbGpus,
                                                shmAipc, 
                                                shmoffsetA);
```

We gather all the pointers in process 0 and set up the solver in the same way as before. After completion, it is essential
to close the memory handles

```cpp
ipcCloseDevicePointers(currentDevice, 
                       opened_ptrs_A.bases, 
                       nbGpus);
```

to avoid memory leaks.

> **Note:** If you've made it this far and have experience or thoughts on this, please reach out!

