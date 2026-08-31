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
 - name: Cavendish Astrophysics, University of Cambridge, Cambridge CB30HE, UK
   index: 1
   ror: 0247acz73
 - name: Kavli Institute for Cosmology, University of Cambridge, Cambridge CB30HA, UK
   index: 2
   ror: 00pwqz914
 - name: Center for Computational Quantum Physics, Flatiron Institute, 162 Fifth Avenue, New York, NY 10010, USA
   index: 3
   ror: 00sekdz59
date: 4 August 2026
bibliography: paper.bib

---

# Summary

Solving large dense linear systems and eigenvalue problems is a core requirement in many areas of scientific computing, but scaling these operations beyond a single GPU remains challenging within modern programming frameworks. While highly optimized multi-GPU solver libraries exist, they are typically difficult to integrate into composable, just-in-time (JIT) compiled Python workflows.

JAXMg provides distributed dense linear algebra for JAX, enabling linear solves and decompositions for matrices that exceed single-GPU memory limits. By interfacing JAX with NVIDIA’s cuSOLVERMp through an XLA Foreign Function Interface, JAXMg exposes distributed GPU routines as JIT-compatible JAX primitives. This design allows scalable linear algebra to be embedded directly within JAX programs, preserving composability with JAX transformations and enabling multi-GPU and multi-node execution in end-to-end scientific workflows.

# Statement of need

Modern scientific computing increasingly relies on GPUs, which now provide a large fraction of the
available floating-point throughput in both supercomputers and smaller multi-GPU
workstations. At the same time, dense linear algebra remains a critical building block for many
numerical methods. A number of mature libraries therefore provide high-performance linear algebra on
CPUs and GPUs, including distributed and accelerator-aware packages such as ScaLAPACK
[@blackford1997scalapack], MAGMA [@abdelfattah2024magma], and SLATE [@gates2019slate].

In parallel, JAX [@jax2018github] has become a widely adopted framework for scientific computing
because it combines a simple user experience with JIT compilation and automatic
differentiation. The JAX ecosystem has expanded rapidly, with libraries for neural networks
[@flax2020github], Bayesian inference [@cabezas2024blackjax], differential equations [@kidger2021on],
Variational Monte Carlo [@netket3:2022]
and full physics simulation environments [@brax2021github]. These workflows often require repeatedly
solving linear systems or computing eigenvalue decompositions, either as part of a larger simulation
loop or inside differentiable optimization.

Despite this growth, and the availability of packages such as Lineax for composable linear solves
within JAX [@lineax2023], the ecosystem still lacks distributed dense linear
solver routines that scale across multiple GPUs while remaining usable
from idiomatic JAX programs. This gap makes it challenging to take
existing JAX-based scientific applications to problem sizes that exceed
a single GPU, or to integrate multi-GPU linear algebra into more complex
JAX pipelines. Existing approaches require leaving the JAX execution
model, either by exporting arrays to external MPI-based solvers or by
manually orchestrating GPU kernels outside JAX's JIT. These approaches
break composability and complicate memory management.

JAXMg addresses this need by providing a distributed multi-GPU and multi-node interface to GPU-accelerated solver
backends, enabling scalable linear solves and eigendecompositions from within JAX. 

# Software design

JAXMg connects JAX to NVIDIA’s distributed dense linear algebra library cuSOLVERMp [@cusolver] via an XLA Foreign Function Interface (FFI) C++/CUDA extension. This design enables writing complex, JIT-compatible JAX programs while delegating the computationally intensive components to a compiled backend.

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

For example, the `potrs` routine can be called over a two-dimensional mesh:

```python
mesh = jax.make_mesh((jax.process_count(), 1), ("pr", "pc"))
matrix_specs = P("pr", "pc")
```

Here, `A` is an $N\times N$ positive-definite matrix sharded over the process rows and columns.
The solve input `b` has shape $N \times N_{\mathrm{RHS}}$ and is sharded over the process rows:

```python
A = jax.device_put(A, NamedSharding(mesh, P("pr", "pc")))
b = jax.device_put(b, NamedSharding(mesh, P("pr", None)))
```

The sharded arrays can then be passed directly to `potrs`:

```python
out = potrs(A, b, T_A=T_A, mesh=mesh, matrix_specs=P("pr", "pc"))
```

The tile size $T_A$ is user-configurable and controls the trade-off between memory usage and
performance; larger tiles typically improve throughput once the problem is sufficiently large
(see Figure \ref{fig:benchmark}).

## Memory-efficient data redistribution

Parallelized linear algebra algorithms require a distributed data layout to ensure proper load balancing of the available computational power [@dongarra1994]. For JAXMg, the central challenge is constructing this layout without reducing the matrix sizes that can be held in aggregate GPU memory. JAXMg therefore transforms the donated matrix buffers in place and reuses a single bounded scratch allocation across all stages. Minimizing memory overhead alone, however, is not sufficient: the redistribution must also use the available interconnect bandwidth efficiently. Although arbitrary permutations can be decomposed into fine-grained cycles, repeated small transfers introduce synchronization and transfer overheads, leading to poor utilization of the bandwidth available from modern GPU interconnects [@li2020interconnect]. JAXMg addresses both requirements through a three-stage redistribution. Each stage moves the largest contiguous regions that fit within a shared scratch allocation and performs independent transfers concurrently wherever dependencies allow. The size of this allocation is determined by the tile slabs used in the final 2D block-cyclic stage, described in Section \ref{sec:block-cyclic}, and is reused throughout. The following sections describe the three forward redistribution stages used to prepare the matrix for distributed solver execution; after the solver completes, these stages are reversed to restore the original JAX layout.

### Local memory-layout conversion

The first stage reconciles the physical memory layouts used by JAX and cuSOLVERMp. JAX stores each local matrix shard in row-major order, whereas cuSOLVERMp requires column-major local buffers. While XLA can materialize a column-major FFI input, doing so requires a second full-sized local matrix allocation, defeating the low-memory design. JAXMg instead applies a parallel implementation of the rectangular permutation decomposition introduced by @catanzaro2014transpose directly to each donated buffer. The method expresses the layout change as modular column and row permutations, processed in batches bounded by the shared scratch allocation. The logical matrix remains unchanged and, because the conversion is entirely local, this stage runs concurrently on every GPU without inter-device communication.


### Edge-padding alignment

Due to the 2D block-cyclic layout required by the cuSOLVERMp backend, JAXMg pads a local JAX shard before it enters the native backend if either dimension is not divisible by the corresponding tile dimension. This provides enough local capacity for the solver layout, but leaves padding between neighbouring shards in the global process grid. As a result, the destination of a solver tile may still contain part of another tile, so the redistribution in Section \ref{sec:block-cyclic} cannot yet move complete tile slabs directly. JAXMg therefore compacts the real matrix towards the global top-left, leaving the padding on the global right and bottom edges, as illustrated in Figure \ref{fig:padding-alignment}.

The compaction proceeds in two passes. Column slabs are first shifted left within each process row, after which row slabs are shifted upwards within each process column. Since the padding provides empty destinations, these movements form open chains and do not require an additional temporary buffer for preserving overwritten data. Dependencies between movements prevent an entire pass from being executed at once, so each pass is divided into ordered waves. Within each wave, the largest slabs that fit in the shared scratch allocation are moved concurrently across independent process rows or columns.


![Demonstration of the edge-padding alignment across a $2 \times 4$ GPU process grid. (a) Initial JAX block-sharded layout after each local shard is padded to a multiple of $T_A$. (b) Horizontal compaction shifts valid column slabs to the left within each process row. (c) Vertical compaction then shifts valid row slabs upwards within each process column, leaving all padding on the global right and bottom edges. Colours indicate the GPU that originally owned each matrix entry, while grey cells denote padding.\label{fig:padding-alignment}](tikz/block_cyclic_padding_alignment.pdf){ width=100% }

Unlike the in-place redistribution handled by the native backend, this initial capacity padding must be performed by JAX because an existing donated allocation cannot be expanded once assigned. Materializing the padded array thus temporarily requires both the original and padded buffers, reducing the matrix size that fits in available GPU memory. Padding should therefore be avoided where possible by choosing a tile size that divides both dimensions of every local matrix shard.

### 2D block-cyclic redistribution {#sec:block-cyclic}

Finally, JAXMg constructs the 2D block-cyclic layout required by cuSOLVERMp. For a process grid with $P_r$ rows and $P_c$ columns, tiles are distributed in round-robin order over both axes, such that the tile at global tile coordinate $(i,j)$ is assigned to

$$
\operatorname{owner}(i,j)
=
\left(i \bmod P_r,\;j \bmod P_c\right).
$$

JAXMg constructs an explicit mapping from every source tile to its destination and applies it in two separable phases. First, the column-owner mapping is decomposed into disjoint permutation cycles that rotate complete tile-column slabs within each process row. The corresponding cycles run concurrently across all process rows until every tile has the correct process-column owner. The same procedure is then applied at the row level within each process column and parallelized across the process columns, as shown in Figure \ref{fig:block-cyclic-redistribution}.

![Demonstration of the redistribution from the compacted JAX block-sharded layout to the 2D block-cyclic layout required by cuSOLVERMp across a $2 \times 4$ GPU process grid. (a) The top-left-aligned matrix is partitioned into $T_A \times T_A$ tiles. (b) Tile-column slabs are redistributed cyclically within each process row, assigning global tile column $j$ to process column $j \bmod P_c$. (c) Tile-row slabs are subsequently redistributed within each process column, assigning global tile row $i$ to process row $i \bmod P_r$. Colours indicate the original GPU ownership of each matrix entry, while grey cells denote padding.\label{fig:block-cyclic-redistribution}](tikz/block_cyclic_redistribution.pdf){ width=100% }


During each cycle, a slab is packed into the send scratch slot, transferred into the receive scratch slot, and unpacked into its destination, while a third saved slot preserves data that would otherwise be overwritten before it is forwarded. Let $N_r$ and $N_c$ denote the local row and column capacities, and let $T_r$ and $T_c$ denote the tile dimensions. Since a tile-column slab contains at most $T_cN_r$ elements and a tile-row slab contains at most $T_rN_c$ elements, the bounded scratch allocation is

$$
S_{\mathrm{scratch}}
=
3\max\left(T_cN_r,\;T_rN_c\right)
$$

elements, where the factor of three accounts for the send, receive, and saved slots.

### Redistribution orchestration

A central design choice in JAXMg is to build the native backend against the matching XLA source through Bazel. This gives the FFI handler access to the underlying NCCL communicator handle (`ncclComm_t`) owned by XLA. JAXMg borrows this communicator for both intra-node and inter-node redistribution before passing the same handle to cuSOLVERMp for the distributed solver operation. Movements confined to one rank use local CUDA operations on the XLA-provided stream. The complete forward redistribution, solver operation, and reverse redistribution can therefore share one communication context within a single fused C++/CUDA FFI call.


# Research Impact Statement

## Scientific applications

We first highlight a selection of examples from across physics in which the dense linear algebra routines provided by JAXMg can be embedded within wider workflows, allowing scientific applications to scale beyond the capabilities of a single GPU while remaining within the JAX ecosystem:

### Cholesky solve (`potrs`)

Across statistical inference, positive-definite linear solves are ubiquitous, particularly in Gaussian processes, a widely used Bayesian nonparametric model, and in the marginalization of nuisance parameters. For a Gaussian process with observations $\mathbf{y}$, mean vector $\mathbf{m}$, kernel covariance $\mathbf{K}$, and noise covariance $\boldsymbol{\Sigma}$, the marginal log likelihood is:

  $$
  \log \mathcal{L}
  = -\frac{1}{2}(\mathbf{y}-\mathbf{m})^\mathsf{T}
      (\mathbf{K}+\boldsymbol{\Sigma})^{-1}(\mathbf{y}-\mathbf{m})
    -\frac{1}{2}\log\det(\mathbf{K}+\boldsymbol{\Sigma})
    -\frac{N_{\mathrm{data}}}{2}\log(2\pi).
  $$

The quadratic and determinant terms thus require a solve against the $N_{\mathrm{data}}\times N_{\mathrm{data}}$ matrix $\mathbf{K}+\boldsymbol{\Sigma}$ and its log determinant; analytically integrating linear nuisance parameters with Gaussian priors yields analogous operations for their posterior precision matrix. One application in which both requirements arise is cosmological inference of the 21-cm brightness-temperature field. As next-generation radio interferometers such as the Square Kilometre Array (SKA) come online, JAXMg's distributed Cholesky factorization allows such analyses to accommodate exascale datasets [@liu2026gpr] and higher-fidelity forward models with larger sets of nuisance parameters [@burba2023allsky] while remaining within JAX.


Positive-definite solves also arise in stochastic reconfiguration (SR), a natural-gradient method widely used to optimize variational quantum states in variational Monte Carlo [@sorella1998green]. For a state $\psi_{\boldsymbol{\theta}}$, SR determines the parameter update $\delta\boldsymbol{\theta}$ from

$$
(\mathbf{S}+\lambda\mathbf{I})\delta\boldsymbol{\theta}
= \mathbf{F},
$$

where $\mathbf{S}$ is the quantum geometric tensor, $\mathbf{F}$ is the variational force associated with the energy gradient, and $\lambda>0$ is a diagonal regularization. The Monte Carlo estimate of $\mathbf{S}$ is Hermitian positive semidefinite; the regularization makes it positive definite and therefore suitable for Cholesky factorization. Because this solve is repeated at every optimization step, explicitly forming $\mathbf{S}$ for $N_{\mathrm{par}}$ parameters requires an $N_{\mathrm{par}}\times N_{\mathrm{par}}$ dense matrix and can quickly become limited by single-GPU memory. Alternatively, in the MinSR formulation [@rende2024simple,@chen2024empowering], the neural tangent kernel is of size $N_{\mathrm{samples}}\times N_{\mathrm{samples}}$, which faces a similar problem as the number of Monte Carlo samples increases. JAXMg's distributed `potrs` allows the parameter update to remain within one JIT-compiled JAX workflow, enabling direct stochastic-reconfiguration solves for larger neural quantum states and other parameterized variational ansätze.
JAXMg is integrated in the latest version of NetKet [@netket3:2022].

### General linear solve (`lu_solve`)

Across computational electromagnetics, general linear solves are central to antenna simulations that model electromagnetic fields and their interaction with complex structures through numerical solutions of Maxwell's equations. A common approach is the method of moments [@Harrington1993], in which the governing integral equations are discretized to produce the dense complex system $ZI=V$, where $Z$ is the impedance matrix, $I$ contains the unknown current coefficients, and $V$ represents one or more excitations. Recovering the current coefficients therefore requires solving this general nonsingular system. One application in which this becomes computationally demanding is the full-wave simulation of mutual coupling across the 320-element core of the Hydrogen Epoch of Reionization Array for 21-cm cosmology [@gueuning2026mutual]. As precision requirements drive larger arrays and finer geometric meshes, the number of basis functions and hence the dimensions of $Z$ grow. JAXMg's distributed LU factorization allows the construction of $Z$, the solution of $ZI=V$, and downstream analysis to remain within a single compiled JAX workflow, supporting efficient end-to-end simulation at increasing model fidelity.


### Symmetric eigendecomposition (`syevd`)

The time-dependent variational principle (TDVP) projects Schrödinger evolution onto the tangent space of a parameterized quantum state, producing equations of motion of the form [@Carleo2017,@Schmitt2020QuantumDynamics]

$$
\dot{\boldsymbol{\theta}}
= -i\mathbf{S}^+ \mathbf{F},
$$

where $\mathbf{S}$ and $\mathbf{F}$ are the same objects as in the Cholesky-solve section. In practical time-dependent variational Monte Carlo calculations, redundant parameters, gauge directions, and Monte Carlo noise can make $\mathbf{S}$ singular or severely ill-conditioned. An eigendecomposition $\mathbf{S}=\mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{\dagger}$ exposes these directions and permits stable spectral regularization, for example by discarding eigenvalues below a threshold or replacing $\boldsymbol{\Lambda}^{-1}$ with a smooth pseudoinverse. Since the decomposition may be required at every stage of an ODE integrator, its memory and computational cost become substantial as the number of variational parameters grows. JAXMg's distributed `syevd` enables full-spectrum regularization and diagnostics to be carried out on matrices exceeding single-GPU memory while keeping state evaluation, Monte Carlo estimation, and time integration inside a compiled JAX program. JAXMg has been used for the t-VMC simulations of [@Wan2026BlurredSampling,@Wiersema2026].

### Singular-value decomposition (`gesvd`)

Principal component analysis (PCA) uses the singular-value decomposition to identify dominant directions of variation in high-dimensional data. For a centered data matrix $X\in\mathbb{R}^{N_{\mathrm{samples}}\times N_{\mathrm{features}}}$,

$$
X=U\Sigma V^\mathsf{T},
$$

the columns of $V$ are the principal directions, while $U\Sigma$ contains the corresponding low-dimensional representations of the samples. The variance associated with the $i$th component is proportional to $\sigma_i^2$, allowing the data to be compressed by retaining only the leading singular values and vectors. JAXMg's distributed `gesvd` routine enables PCA for dense datasets whose sample or feature dimensions make the full data matrix and its decomposition too large for a single GPU.

Singular-value decompositions are a central primitive in tensor-network algorithms for quantum many-body systems [@schollwock2011density; @banuls2023tensor]. After applying a local gate, contracting neighbouring tensors, or enlarging an auxiliary bond, the resulting tensor is reshaped into a matrix and decomposed as $X=U\Sigma V^\mathsf{T}$. Retaining the largest singular values yields the optimal low-rank approximation in the Frobenius norm, controls the tensor-network bond dimension, and provides an estimate of the truncation error through the discarded singular-value weight. The SVD is therefore the backbone of modern tensor network algorithms such as the density matrix renormalization group (DMRG), time-evolving block decimation (TEBD) and approximate contraction schemes for hig-dimensional networks. As the physical dimension and bond dimensions grow, the intermediate matrices entering these decompositions can become considerably larger than the tensors retained after truncation and may exceed the memory of one GPU. JAXMg's distributed `gesvd` allows tensor contractions, reshaping, singular-vector construction, and truncation to remain within a JIT-compiled JAX workflow, supporting tensor-network calculations at larger bond dimensions without exporting arrays to an external distributed solver.

## Performance and scaling

To assess performance, we benchmark JAXMg against the single-GPU routines currently available in JAX. These simulations are performed on 16 NVIDIA H100s (94 GB VRAM) connected with SXM5 NVLink and 4x 400Gb InfiniBand connections for inter-node communication.

We report four representative cases: `potrs` (float32), `lu_solve` (float64), and `gesvd` (complex64) and `syevd` (complex128).
For `potrs`, `lu_solve` and `syevd`, we use a diagonal matrix $A=\mathrm{diag}(1,\ldots,N)$, and for `potrs` we set $b=(1,\ldots,1)^\mathsf{T}$. For `gesvd`, we use a random Gaussian matrix with mean zero and unit variance.

We set the tile size $T_A=1024$ and report wall-clock timings in \autoref{fig:benchmark}; the benchmark code is available at [@jaxmg_benchmark].
We see that JAXMg scales better than the native single-GPU linear algebra routines and surpasses them in performance, especially for larger matrices.
Both `syevd` and `syevd` require significantly more workspace memory than `potrs` and `lu_solve`, which is reflected in the matrix sizes that can be reached.

![Benchmark comparing the native single-GPU JAX routines (which call cuSOLVERDn) to JAXMg. (a) Comparison of `jaxmg.potrs` with `jax.scipy.linalg.cho_factor` + `jax.scipy.linalg.cho_solve` for a float32 matrix. (b) Comparison of `jaxmg.lu_solve` with `jax.numpy.linalg.inv` for a complex128 matrix. (c) Comparison of `jaxmg.gesvd` for(d) Comparison of `jaxmg.syevd` with `jax.numpy.linalg.eigh`.\label{fig:benchmark}](jaxmg_benchmark_v1.png){ width=100% }

![Benchmark comparing the native single-GPU JAX routines (which call cuSOLVERDn) to JAXMg. (a) Comparison of `jaxmg.potrs` with `jax.scipy.linalg.cho_factor` + `jax.scipy.linalg.cho_solve` for a float32 matrix. (b) Comparison of `jaxmg.lu_solve` with `jax.numpy.linalg.inv` for a complex128 matrix. (c) Comparison of `jaxmg.gesvd` for(d) Comparison of `jaxmg.syevd` with `jax.numpy.linalg.eigh`.\label{fig:benchmark}](tile_sweep.png){ width=100% }

To test the limits of what JAXMg can do, we performed a Cholesky solve of a matrix of size $N=1499136$ with $T_A=1024$ on 64 NVIDIA H200s (143 GB VRAM) laid out on an 8x8 grid with. Here, we used 8 GPUs per node with SXM5 NVLink and 8x 400Gb InfiniBand connections for inter-node communication. The average solve time was 666.5 seconds, which was approximately 6 times faster than a calculation of similar size on a 64x1 grid.

These results highlight JAXMg’s primary impact: enabling dense linear solves and eigendecompositions that are bottlenecked by the memory capacity of a single GPU, while remaining within JAX’s composable and JIT-compiled programming model. On modern multi-GPU nodes, distributed in-node solvers make it possible to tackle matrix sizes that would otherwise be infeasible, and to increase throughput by using aggregate device memory and compute.

# AI usage disclosure

Claude code was used during software development for code exploration and debugging. Large language models were used to assist with language polishing.

# Acknowledgements

We want to thank Dennis Bollweg, Alex Chavin, Geraud Krawezik, Dylan Simon and Nils Wentzell for their help with developing the code. We also want to acknowledge the help of Ao Chen and Riccardo Rende with testing the code in applied settings. RW is grateful to Simon Tartakovsky for his suggestions on the 1D cyclic algorithm. Finally, we want to thank Filippo Vincentini for his suggestions on code distribution. I acknowledge support from the Flatiron Institute. The Flatiron Institute is a division of the Simons Foundation. 

The authors acknowledge the use of resources provided by the Isambard-AI National AI Research Resource (AIRR). Isambard-AI [@Isamabrd_2024] is operated by the University of Bristol and is funded by the UK Government’s Department for Science, Innovation and Technology (DSIT) via UK Research and Innovation; and the Science and Technology Facilities Council [ST/AIRR/I-A-I/1023].


# References
