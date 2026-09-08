---
title: 'JAXMg: A multi-GPU linear solver in JAX'
tags:
  - Python
  - JAX
  - CUDA
  - distributed linear algebra
authors:
  - name: Jacob Tutt
    orcid: 0009-0002-5358-4292
    equal-contrib: true
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Roeland Wiersema
    orcid: 0000-0002-0839-4265
    equal-contrib: true
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3 # (Multiple affiliations must be quoted)
affiliations:
 - name: Cavendish Astrophysics, University of Cambridge, Cambridge CB3 0HE, UK
   index: 1
   ror: 0247acz73
 - name: Kavli Institute for Cosmology, University of Cambridge, Cambridge CB3 0HA, UK
   index: 2
   ror: 00pwqz914
 - name: Center for Computational Quantum Physics, Flatiron Institute, 162 Fifth Avenue, New York, NY 10010, USA
   index: 3
   ror: 00sekdz59
date: 4 August 2026
bibliography: paper.bib

---

# Summary

Solving large dense linear systems and eigenvalue problems is a core requirement
across scientific computing, but scaling these operations beyond a single GPU
remains challenging  within modern programming frameworks. Highly optimized multi-GPU solver libraries exist, yet they are hard to integrate into composable, just-in-time (JIT) compiled Python workflows.

JAXMg provides distributed dense linear algebra for JAX, enabling linear solves and decompositions for matrices that exceed single-GPU memory limits. By interfacing JAX with NVIDIA's cuSOLVERMp through an XLA Foreign Function Interface, JAXMg exposes distributed GPU routines as JIT-compatible JAX primitives. This design allows scalable linear algebra to be embedded directly within JAX programs, preserving composability with JAX transformations and enabling multi-GPU and multi-node execution in end-to-end scientific workflows.

# Statement of need

GPUs now supply a large fraction of the floating-point throughput of both
supercomputers and smaller multi-GPU workstations, and dense linear algebra
remains a critical building block of many numerical methods.

JAX [@jax2018github] has become a widely adopted framework for
scientific computing because it combines a simple user experience with JIT
compilation and automatic differentiation. The JAX ecosystem has expanded rapidly, with libraries for neural networks
[@flax2020github], Bayesian inference [@cabezas2024blackjax], differential equations [@kidger2021on],
variational Monte Carlo [@netket3:2022],
and full physics simulation environments [@brax2021github]. These workflows
frequently solve linear systems or compute eigendecompositions, either inside a
larger simulation loop or inside differentiable optimization, and the size of
those matrices is often what limits the science that can be done.

Despite this growth, and the availability of packages such as Lineax for composable linear solves
within JAX [@lineax2023], the ecosystem still lacks distributed dense linear
solver routines that scale across multiple GPUs while remaining usable
from idiomatic JAX programs. Existing
routes require leaving the JAX execution model, either by exporting arrays to an
external MPI-based solver or by orchestrating GPU kernels outside JAX's JIT.
Both break composability, force host round-trips, and complicate memory
management at exactly the point where memory is the binding constraint. JAXMg
fills this gap, making distributed solves and decompositions ordinary operations
inside a compiled JAX program.

# State of the field

Mature distributed dense linear algebra libraries are well established.
ScaLAPACK [@blackford1997scalapack] set the standard for block-cyclic
distributed factorizations on CPU clusters; SLATE [@gates2019slate] is its
modern, GPU-aware successor; MAGMA [@abdelfattah2024magma] targets hybrid
CPU-GPU nodes; and cuSOLVERMp [@cusolver] provides NVIDIA's multi-GPU,
multi-node implementations. All of these are driven from C, C++ or Fortran
programs that own their own MPI communicator and data distribution. 

# Software design

JAXMg connects JAX to NVIDIA's distributed dense linear algebra library cuSOLVERMp [@cusolver] via an XLA Foreign Function Interface (FFI) C++/CUDA extension. This design enables writing complex, JIT-compatible JAX programs while delegating the computationally intensive components to a compiled backend.

Simply pass JAXMg an ordinary JAX array sharded over a two-dimensional device mesh. The native backend handles the local memory-layout conversion, 2D block-cyclic redistribution, distributed solver execution, and restoration of the result to its original JAX layout.

The current release provides a JIT-compatible interface to four workflows:

- `potrs`: Solves $Ax=b$ for symmetric (Hermitian) positive-definite $A$ using a Cholesky
  factorization (`cusolverMpPotrf` and `cusolverMpPotrs`). The same factorization can optionally
  return $\log\det(A)$.
- `lu_solve`: Solves $Ax=b$ for general nonsingular $A$ using a pivoted LU factorization
  (`cusolverMpGetrf` and `cusolverMpGetrs`).
- `syevd`: Computes the eigenvalues $\lambda_i$ and eigenvectors $v_i$ of a symmetric
  (Hermitian) matrix $A$, satisfying $Av_i=\lambda_i v_i$ (`cusolverMpSyevd`).
- `gesvd`: Computes the singular-value decomposition of an $M\times N$ matrix
  $A=U\Sigma V^\dagger$, returning the singular values and optional left and right singular
  vectors (`cusolverMpGesvd`).

All four routines support the JAX dtypes float32, float64, complex64, and complex128, with CUDA 12 and CUDA 13 backends available for both x86_64 and aarch64 systems.

Parallelized linear algebra algorithms require a distributed data layout to ensure proper load balancing of the available computational power [@dongarra1994]. For JAXMg, the central challenge is constructing this layout without reducing the matrix sizes that can be held in aggregate GPU memory. JAXMg therefore transforms the donated matrix buffers in place and reuses a single bounded scratch allocation across all stages. Minimizing memory overhead alone, however, is not sufficient: the redistribution must also use the available interconnect bandwidth efficiently. Although arbitrary permutations can be decomposed into fine-grained cycles, repeated small transfers introduce synchronization and transfer overheads, leading to poor utilization of the bandwidth available from modern GPU interconnects [@li2020interconnect]. JAXMg addresses both requirements through a three-stage redistribution. Each stage moves the largest contiguous regions that fit within a shared scratch allocation and performs independent transfers concurrently wherever dependencies allow. The size of this allocation is determined by the tile slabs used in the final 2D block-cyclic stage, described in Section \ref{sec:block-cyclic}, and is reused throughout. The following sections describe the three forward redistribution stages used to prepare the matrix for distributed solver execution; after the solver completes, these stages are reversed to restore the original JAX layout.

### Local memory-layout conversion

The first stage reconciles the physical memory layouts used by JAX and cuSOLVERMp. JAX stores each local matrix shard in row-major order, whereas cuSOLVERMp requires column-major local buffers. While XLA can materialize a column-major FFI input, doing so requires a second full-sized local matrix allocation, defeating the low-memory design. JAXMg instead applies a parallel implementation of the rectangular permutation decomposition introduced by @catanzaro2014transpose directly to each donated buffer. The method expresses the layout change as modular column and row permutations, processed in batches bounded by the shared scratch allocation. The logical matrix remains unchanged and, because the conversion is entirely local, this stage runs concurrently on every GPU without inter-device communication.


### Edge-padding alignment

Due to the 2D block-cyclic layout required by the cuSOLVERMp backend, JAXMg pads a local JAX shard before it enters the native backend if either dimension is not divisible by the corresponding tile dimension. This provides enough local capacity for the solver layout, but leaves padding between neighbouring shards in the global process grid. As a result, the destination of a solver tile may still contain part of another tile, so the redistribution in Section \ref{sec:block-cyclic} cannot yet move complete tile slabs directly. JAXMg therefore compacts the real matrix towards the global top-left, leaving the padding on the global right and bottom edges, as illustrated in Figure \ref{fig:padding-alignment}.

The compaction proceeds in two passes. Column slabs are first shifted left within each process row, after which row slabs are shifted upwards within each process column. Since the padding provides empty destinations, these movements form open chains and do not require an additional temporary buffer for preserving overwritten data. Dependencies between movements prevent an entire pass from being executed at once, so each pass is divided into ordered waves. Within each wave, the largest slabs that fit in the shared scratch allocation are moved concurrently across independent process rows or columns.


![Demonstration of the edge-padding alignment across a $2 \times 4$ GPU process grid. (a) Initial JAX block-sharded layout after each local shard is padded to a multiple of $T_A$. (b) Horizontal compaction shifts valid column slabs to the left within each process row. (c) Vertical compaction then shifts valid row slabs upwards within each process column, leaving all padding on the global right and bottom edges. Colours indicate the GPU that originally owned each matrix entry, while grey cells denote padding.\label{fig:padding-alignment}](tikz/block_cyclic_padding_alignment.pdf){ width=100% }

Unlike the in-place redistribution handled by the native backend, this initial capacity padding must be performed by JAX because an existing donated allocation cannot be expanded once assigned. Materializing the padded array thus temporarily requires both the original and padded buffers, reducing the matrix size that fits in available GPU memory. Padding should therefore be avoided where possible by choosing a tile size that divides both dimensions of every local matrix shard.

### 2D block-cyclic redistribution

Finally, JAXMg constructs the 2D block-cyclic layout required by cuSOLVERMp. For a process grid with $P_r$ rows and $P_c$ columns, tiles are distributed in round-robin order over both axes, such that the tile at global tile coordinate $(i,j)$ is assigned to $\operatorname{owner}(i,j)=\left(i \bmod P_r,\;j \bmod P_c\right)$.

JAXMg constructs an explicit mapping from every source tile to its destination and applies it in two separable phases. First, the column-owner mapping is decomposed into disjoint permutation cycles that rotate complete tile-column slabs within each process row. The corresponding cycles run concurrently across all process rows until every tile has the correct process-column owner. The same procedure is then applied at the row level within each process column and parallelized across the process columns, as shown in Figure \ref{fig:block-cyclic-redistribution}.

![Demonstration of the redistribution from the compacted JAX block-sharded layout to the 2D block-cyclic layout required by cuSOLVERMp across a $2 \times 4$ GPU process grid. (a) The top-left-aligned matrix is partitioned into $T_A \times T_A$ tiles. (b) Tile-column slabs are redistributed cyclically within each process row, assigning global tile column $j$ to process column $j \bmod P_c$. (c) Tile-row slabs are subsequently redistributed within each process column, assigning global tile row $i$ to process row $i \bmod P_r$. Colours indicate the original GPU ownership of each matrix entry, while grey cells denote padding.\label{fig:block-cyclic-redistribution}](tikz/block_cyclic_redistribution.pdf){ width=100% }


During each cycle, a slab is packed into the send scratch slot, transferred into the receive scratch slot, and unpacked into its destination, while a third saved slot preserves data that would otherwise be overwritten before it is forwarded. This results in a highly memory efficient implementation of the memory remapping.

### Redistribution orchestration

A central design choice in JAXMg is to build the native backend against the matching XLA source through Bazel. This gives the FFI handler access to the underlying NCCL communicator handle (`ncclComm_t`) owned by XLA. JAXMg borrows this communicator for both intra-node and inter-node redistribution before passing the same handle to cuSOLVERMp for the distributed solver operation. Movements confined to one rank use local CUDA operations on the XLA-provided stream. The complete forward redistribution, solver operation, and reverse redistribution can therefore share one communication context within a single fused C++/CUDA FFI call.

# Research impact statement

JAXMg is integrated into NetKet [@netket3:2022], one of the most widely used open-source frameworks for neural quantum states, where it backs the distributed linear solve at the heart of stochastic reconfiguration [@sorella1998green]. Distributing it removes the ceiling on how large an ansatz NetKet users can optimize. Additionally, JAXMg produced the time-dependent variational Monte Carlo results of
@Wan2026BlurredSampling and [@Wiersema2026]. A future release of jVMC [@jVMC],
will also feature support for JAXMg.

**21-cm results**
[@gueuning2026mutual]. 

## Performance and scaling

Performance in distributed matrix routines depends not only on the available GPU resources, but also on how the calculation is divided between them. To quantify this dependence and provide practical guidance for users, we investigate the effects of the process-grid layout and matrix tile dimension, $T_A$, on wall-clock runtime in Figures \ref{fig:benchmark} and \ref{fig:tile_sweep}, respectively.

![Wall-clock runtime against matrix dimension $N$ for JAXMg on 16 NVIDIA H100 GPUs at fixed tile dimension $T_A=256$. The $16\times1$ process grid is shown in dark blue, the $4\times4$ process grid in orange, and native single-GPU JAX in dashed green. The panels show (a) `jaxmg.potrs` (`float32`), (b) `jaxmg.lu_solve` (`float64`), (c) `jaxmg.gesvd` (`complex64`), and (d) `jaxmg.syevd` (`complex128`). The $16\times1$ curves begin at larger $N$ because avoiding padding requires $N$ to be divisible by $16T_A$, compared with $4T_A$ for the $4\times4$ grid.\label{fig:benchmark}](jaxmg_benchmark.png){ width=100% }

We consider four representative combinations of routine and data type: `potrs` with `float32`, `lu_solve` with `float64`, `gesvd` with `complex64`, and `syevd` with `complex128`. For `potrs`, `lu_solve`, and `syevd`, we use the diagonal matrix $A=\operatorname{diag}(1,\ldots,N)$; for the two linear solves, we additionally set $b=(1,\ldots,1)^\mathsf{T}$. For `gesvd`, we use a random Gaussian matrix with zero mean and unit variance. Both benchmarks use 16 NVIDIA H100 SXM5 GPUs (94 GB each) across four nodes, with four NVLink-connected GPUs and four 400 Gb/s InfiniBand links per node. The benchmark implementation is available at [@jaxmg_benchmark].

Figure \ref{fig:benchmark} compares a one-dimensional $16\times1$ process grid with a two-dimensional $4\times4$ grid using a fixed tile dimension $T_A=256$. Under the block-cyclic redistribution outlined in Section \ref{sec:block-cyclic}, moving from the $16\times1$ to the $4\times4$ grid introduces a second inter-rank redistribution phase, but provides a more balanced two-dimensional distribution during cuSOLVERMp execution. The process-grid layout therefore determines the balance between redistribution overhead and distributed solver efficiency. The results show that, across every routine, the solver-side benefit outweighs the additional communication, with the $4\times4$ grid achieving geometric-mean speed-ups over the matrix dimensions completed by both layouts of $1.77\times$ for `potrs`, $1.23\times$ for `lu_solve`, $1.38\times$ for `gesvd`, and $1.21\times$ for `syevd` relative to the $16\times1$ grid. The smaller maximum matrix dimensions reached by `gesvd` and `syevd` reflect the significantly larger cuSOLVERMp workspaces required by these decompositions. Finally, Figure \ref{fig:benchmark} demonstrates that, for each representative routine--dtype combination, both distributed layouts outperform the corresponding native single-GPU JAX implementation at sufficiently large $N$, showing that the computational benefit of distributed execution outweighs its redistribution overhead before single-GPU memory becomes limiting.

![Effect of the tile size on JAXMg wall-clock runtime for the $4\times4$ process grid on the same 16 NVIDIA H100 GPUs. Each panel draws one curve per $T_A\in\{256,512,1024,2048,4096\}$, shaded light to dark with increasing tile size. There are four panels: (a) `jaxmg.potrs` (float32), (b) `jaxmg.lu_solve` (float64), (c) `jaxmg.gesvd` (complex64) and (d) `jaxmg.syevd` (complex128). Curves for larger $T_A$ begin at larger $N$ because a dimension must be a whole number of tiles along both grid axes.\label{fig:tile_sweep}](tile_sweep.png){ width=100% }

Having established the performance advantage of the $4\times4$ process grid, we next hold this layout fixed and vary the tile dimension from $T_A=2^8$ to $2^{12}$, as shown in Figure \ref{fig:tile_sweep}. The response is strongly routine-dependent: increasing $T_A$ reduces the runtime of `potrs`, leaves `lu_solve` broadly unchanged, and increases the runtime of `syevd`, while `gesvd` is non-monotonic, running fastest at an intermediate $T_A=1024$ and more slowly for both smaller and larger tiles.

We finally demonstrate the large-scale performance of JAXMg using 64 NVIDIA H200 SXM5 GPUs (143 GB each) across eight nodes, with eight NVLink-connected GPUs and eight 400 Gb/s InfiniBand links per node. By reducing the redistribution scratch required for a fixed tile dimension relative to the $64\times1$ layout, the balanced $8\times8$ process grid enabled a Cholesky solve with $N=1{,}499{,}136$ and $T_A=1024$. The distributed matrix occupied 8.2 TiB in aggregate, while each local shard and its redistribution scratch used 94.7\% of the available device memory. The warm solve completed in 654 seconds. At $N=1{,}310{,}720$, the largest dimension completed by both layouts, the $8\times8$ grid was also $6.8\times$ faster, requiring 452 seconds compared with 3065 seconds for $64\times1$.

These results highlight JAXMg's primary impact: enabling dense matrix solves and decompositions that are bottlenecked by the memory capacity of a single GPU, while remaining within JAX's composable and JIT-compiled programming model. On modern multi-GPU systems, distributed solvers make it possible to tackle matrix sizes that would otherwise be infeasible, and to increase throughput by using aggregate device memory and compute.

# AI usage disclosure

Claude Code was used during software development for code exploration and debugging. Large language models were used to assist with language polishing.

# Acknowledgements

We want to thank Dennis Bollweg, Alex Chavin, Geraud Krawezik, Dylan Simon and Nils Wentzell for their help with developing the code. We also want to acknowledge the help of Ao Chen and Riccardo Rende with testing the code in applied settings. RW is grateful to Simon Tartakovsky for his suggestions on the 1D cyclic algorithm. Finally, we want to thank Filippo Vincentini for his suggestions on code distribution. RW acknowledges support from the Flatiron Institute. The Flatiron Institute is a division of the Simons Foundation. JT is supported by the Harding Distinguished Postgraduate Scholars Programme (HDPSP) and the Science and Technology Facilities Council (STFC) DTP Studentship.

The authors acknowledge the use of resources provided by the Isambard-AI National AI Research Resource (AIRR). Isambard-AI [@Isamabard_2024] is operated by the University of Bristol and is funded by the UK Government's Department for Science, Innovation and Technology (DSIT) via UK Research and Innovation; and the Science and Technology Facilities Council [ST/AIRR/I-A-I/1023].


# References
