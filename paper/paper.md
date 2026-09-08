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
those matrices is often what limits the science that can be done. Within the JAX ecosystem, Lineax [@lineax2023] provides composable linear-operator abstractions and direct and iterative solvers.

Despite this growth, the ecosystem still lacks distributed dense linear
solver routines that scale across multiple GPUs while remaining usable
from idiomatic JAX programs. A dedicated backend package makes this integration reusable across scientific applications while containing the native CUDA dependencies and redistribution machinery in one implementation. JAXMg therefore complements existing solver libraries and JAX application frameworks by providing the execution and memory-management layer needed to connect them.

# State of the field

Mature distributed dense linear algebra libraries are well established.
ScaLAPACK [@blackford1997scalapack] set the standard for block-cyclic
distributed factorizations on CPU clusters; SLATE [@gates2019slate] is its
modern, GPU-aware successor; MAGMA [@abdelfattah2024magma] targets hybrid
CPU-GPU nodes; and cuSOLVERMp [@cusolver] provides NVIDIA's multi-GPU,
multi-node implementations. All of these are driven from C, C++ or Fortran
programs that own their own MPI communicator and data distribution. 

# Software design

JAXMg connects JAX to NVIDIA's distributed dense linear algebra library cuSOLVERMp [@cusolver] via an XLA Foreign Function Interface (FFI) C++/CUDA extension. The supported cuSOLVERMp routines support the JAX dtypes float32, float64, complex64, and complex128, with CUDA 12 and CUDA 13 backends available for both x86_64 and aarch64 systems. 

Parallelized linear algebra algorithms, like the ones implemented by cuSOLVERMp, require a distributed data layout to ensure proper load balancing of the available computational power [@dongarra1994]. For JAXMg, the central challenge is constructing this layout in a fast and memory efficient manner. JAXMg transforms the matrix buffers that are donated through the FFI in place and reuses a single bounded scratch allocation across all stages. 
JAXMg performs the required data redistribution in three stages.

### Local memory-layout conversion

The first stage reconciles the physical memory layouts used by JAX and cuSOLVERMp. JAX stores each local matrix shard in row-major order, whereas cuSOLVERMp requires column-major local buffers. While XLA can materialize a column-major FFI input, doing so requires a second full-sized local matrix allocation, defeating the low-memory design. JAXMg instead applies a parallel implementation of the rectangular permutation decomposition introduced by @catanzaro2014transpose directly to each donated buffer. The method expresses the layout change as modular column and row permutations, processed in batches bounded by the shared scratch allocation. The logical matrix remains unchanged and, because the conversion is entirely local, this stage runs concurrently on every GPU without inter-device communication.

### Edge-padding alignment

Due to the 2D block-cyclic layout required by the cuSOLVERMp backend, JAXMg pads a local JAX shard before it enters the native backend if either dimension is not divisible by the corresponding tile dimension. This provides enough local capacity for the solver layout, but leaves padding between neighbouring shards in the global process grid. As a result, the destination of a solver tile may still contain part of another tile, so the block-cyclic redistribution described below cannot yet move complete tile slabs directly. JAXMg therefore compacts the logical matrix towards the global top-left, leaving the padding on the global right and bottom edges, as illustrated in Figure \ref{fig:padding-alignment}.

The compaction proceeds in two passes. Column slabs are first shifted left within each process row, after which row slabs are shifted upwards within each process column. Since the padding provides empty destinations, these movements form open chains and do not require an additional temporary buffer for preserving overwritten data. Dependencies between movements prevent an entire pass from being executed at once, so each pass is divided into ordered waves. Within each wave, the largest slabs that fit in the shared scratch allocation are moved concurrently across independent process rows or columns.


![Demonstration of the edge-padding alignment across a $2 \times 4$ GPU process grid. (a) Initial JAX block-sharded layout after each local shard is padded to a multiple of $T_A$. (b) Horizontal compaction shifts valid column slabs to the left within each process row. (c) Vertical compaction then shifts valid row slabs upwards within each process column, leaving all padding on the global right and bottom edges. Colours indicate the GPU that originally owned each matrix entry, while grey cells denote padding.\label{fig:padding-alignment}](tikz/block_cyclic_padding_alignment.pdf){ width=100% }

Unlike the in-place redistribution handled by the native backend, this initial capacity padding must be performed by JAX because an existing donated allocation cannot be expanded once assigned. Materializing the padded array thus temporarily requires both the original and padded buffers, reducing the matrix size that fits in available GPU memory. Padding should therefore be avoided where possible by choosing a tile size that divides both dimensions of every local matrix shard.


### 2D block-cyclic redistribution

Finally, JAXMg constructs the 2D block-cyclic layout required by cuSOLVERMp. For a process grid with $P_r$ rows and $P_c$ columns, tiles are distributed in round-robin order over both axes, such that the tile at global tile coordinate $(i,j)$ is assigned to $\operatorname{owner}(i,j)=\left(i \bmod P_r,\;j \bmod P_c\right)$.

JAXMg constructs an explicit mapping from every source tile to its destination and applies it in two separable phases. First, the column-owner mapping is decomposed into disjoint permutation cycles that rotate complete tile-column slabs within each process row. The corresponding cycles run concurrently across all process rows until every tile has the correct process-column owner. The same procedure is then applied at the row level within each process column and parallelized across the process columns, as shown in Figure \ref{fig:block-cyclic-redistribution}.

![Demonstration of the redistribution from the compacted JAX block-sharded layout to the 2D block-cyclic layout required by cuSOLVERMp across a $2 \times 4$ GPU process grid. (a) The top-left-aligned matrix is partitioned into $T_A \times T_A$ tiles. (b) Tile-column slabs are redistributed cyclically within each process row, assigning global tile column $j$ to process column $j \bmod P_c$. (c) Tile-row slabs are subsequently redistributed within each process column, assigning global tile row $i$ to process row $i \bmod P_r$. Colours indicate the original GPU ownership of each matrix entry, while grey cells denote padding.\label{fig:block-cyclic-redistribution}](tikz/block_cyclic_redistribution.pdf){ width=100% }

During each cycle, a slab is packed into the send scratch slot, transferred into the receive scratch slot, and unpacked into its destination, while a third saved slot preserves data that would otherwise be overwritten before it is forwarded. This results in a highly memory efficient implementation of the memory remapping.

### Solver orchestration
An important design feature of JAXMg is that this entire pipeline is performed within a single fused C++/CUDA FFI call that the user never has to interact with. This design enables writing complex, JIT-compatible JAX programs while delegating the computationally intensive components to a compiled backend.
Simply pass JAXMg an ordinary JAX array sharded over a two-dimensional device mesh. The native backend handles the local memory-layout conversion, 2D block-cyclic redistribution, distributed solver execution, and restoration of the result to its original JAX layout. 

# Research impact statement

JAXMg is integrated into NetKet [@netket3:2022], one of the most widely used open-source frameworks for variational Monte Carlo, where it backs the distributed linear solve at the heart of stochastic reconfiguration [@sorella1998green]. Distributing it removes the ceiling on how large an ansatz NetKet users can optimize. Additionally, JAXMg produced the time-dependent variational Monte Carlo [@Carleo2017;@Schmitt2020QuantumDynamics] results of
[@Wiersema2026,@Wan2026BlurredSampling]. A future release of jVMC [@jVMC],
will also feature support for JAXMg. In a series of benchmark experiments, we also illustrate the power of JAXMg by investigating the scalability of the currently implemented routines across a large number of GPUs. As a highlight, we perform a successful Cholesky solve of a float32 matrix of size $1.5\times10^6 \times 1.5\times10^6$ across 64 NVIDIA H200s in approximately 11 minutes [@jaxmg_benchmark].

**21-cm results**
[@gueuning2026mutual]. 

We envision future scientific applications in areas such as Bayesian inference [@liu2026gpr;@burba2023allsky], tensor networks [@schollwock2011density;@banuls2023tensor], computational electromagnetics [@Harrington1993;@gueuning2026mutual].

# AI usage disclosure

Claude Code was used during software development for code exploration and debugging. Large language models were used to assist with language polishing. All code and text was reviewed by Humans.

# Acknowledgements

We want to thank Dennis Bollweg, Alex Chavin, Geraud Krawezik, Dylan Simon and Nils Wentzell for their help with code development. We also want to acknowledge the help of Ao Chen and Riccardo Rende with testing the code in applied settings. RW is grateful to Simon Tartakovsky for his suggestions on the 1D cyclic algorithm. Finally, we want to thank Filippo Vicentini for his suggestions on code distribution. RW acknowledges support from the Flatiron Institute. The Flatiron Institute is a division of the Simons Foundation. JT is supported by the Harding Distinguished Postgraduate Scholars Programme (HDPSP) and the Science and Technology Facilities Council (STFC) DTP Studentship.

The authors acknowledge the use of resources provided by the Isambard-AI National AI Research Resource (AIRR). Isambard-AI [@Isamabard_2024] is operated by the University of Bristol and is funded by the UK Government's Department for Science, Innovation and Technology (DSIT) via UK Research and Innovation; and the Science and Technology Facilities Council [ST/AIRR/I-A-I/1023].


# References
