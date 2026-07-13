# Choose a tile size $T_A$

cuSOLVERMp divides a matrix into square $T_A\times T_A$ tiles. JAXMg accepts a
matrix in a regular JAX block sharding, so each local shard should contain a
whole number of tiles whenever possible. A well-chosen tile size avoids an
additional padded copy of the coefficient matrix.

## The no-padding condition

Consider a square $N\times N$ matrix distributed over a
$P_r\times P_c$ process grid. Each process initially owns a rectangular JAX
shard with shape

$$
N_{\mathrm{local\ rows}}=\frac{N}{P_r},
\qquad
N_{\mathrm{local\ cols}}=\frac{N}{P_c}.
$$

The global dimension must first be divisible by both process-grid dimensions:

$$
N\bmod P_r=0,
\qquad
N\bmod P_c=0.
$$

JAXMg does not need to pad $A$ when $T_A$ divides both local shard dimensions:

$$
\frac{N}{P_r}\bmod T_A=0,
\qquad
\frac{N}{P_c}\bmod T_A=0.
$$

Equivalently, choose $T_A$ from the divisors of

$$
\gcd\left(\frac{N}{P_r},\frac{N}{P_c}\right).
$$

## Worked example

For $N=98{,}304$ and a $4\times2$ process grid, every process starts with

$$
\left(
\frac{98{,}304}{4},
\frac{98{,}304}{2}
\right)
=
(24{,}576,49{,}152)
$$

matrix entries. Tile sizes such as

$$
T_A\in\{256,512,1024,2048,4096\}
$$

divide both dimensions, so no coefficient-matrix padding is required. For
example, $T_A=1024$ gives a local tile grid of

$$
24\times48.
$$

The resulting global tile ownership follows the ordinary 2D block-cyclic rule:

$$
\operatorname{owner}(i,j)
=
\left(i\bmod P_r,\ j\bmod P_c\right),
$$

where $(i,j)$ is the global tile coordinate.

## What happens when the tile size does not divide

If either local dimension is not divisible by $T_A$, JAXMg extends every local
shard to the next tile boundary:

$$
\Delta_r
=
\left(-N_{\mathrm{local\ rows}}\right)\bmod T_A,
\qquad
\Delta_c
=
\left(-N_{\mathrm{local\ cols}}\right)\bmod T_A.
$$

The padded local shape is therefore

$$
\left(
N_{\mathrm{local\ rows}}+\Delta_r,
N_{\mathrm{local\ cols}}+\Delta_c
\right).
$$

This allocates a padded matrix before the fused native call. Inside native code,
the top-left alignment stage consolidates that capacity on the global right and
bottom edges before the 2D block-cyclic redistribution. Padding is supported,
but avoiding it is important for matrices close to the GPU memory limit.

## Performance and memory trade-off

A tile size should normally satisfy three requirements:

1. Use $T_A\ge128$; very small tiles create more solver and communication work.
2. Choose a divisor of both local matrix dimensions to avoid padding $A$.
3. Benchmark several valid values. Larger tiles can improve solver throughput,
   but also increase native scratch memory.

For square tiles, the shared redistribution scratch allocation contains

$$
N_{\mathrm{scratch}}
=
3T_A\max\left(
N_{\mathrm{local\ rows}},
N_{\mathrm{local\ cols}}
\right)
$$

elements. Increasing $T_A$ therefore increases scratch memory linearly. A good
starting point is $T_A=256$, $512$, or $1024$, restricted to values that divide
both local dimensions.

## Right-hand-side padding

For `potrs` and `lu_solve`, a narrow right-hand side $B$ may still require a
small padded routing width, particularly when the process grid has more than
one process column. This is independent of whether $A$ is tile-aligned. The
coefficient matrix dominates memory use, so the primary tile-size decision is
to avoid padding $A$; leave `pad=True` to let JAXMg handle a skinny $B$.

See [Memory distribution](../technical_details/memory_distribution.md) for the
layout conversion, padding alignment, 2D block-cyclic redistribution, and
scratch-buffer implementation.
