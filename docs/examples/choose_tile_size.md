# Choosing a Tile Size

JAXMg divides a matrix into square $T_A\times T_A$ tiles and redistributes
them into the 2D block-cyclic layout required by cuSOLVERMp. Whenever
possible, choose $T_A$ so that each local JAX shard contains a whole number
of tiles. This avoids allocating a padded copy of the input matrix and
reduces the work required before native redistribution.

## The no-padding condition

Consider a square $N\times N$ matrix distributed over a
$P_r\times P_c$ process grid. Each process initially owns a rectangular JAX
shard with shape

$$
N_{\mathrm{local\ rows}}=\frac{N}{P_r},
\qquad
N_{\mathrm{local\ cols}}=\frac{N}{P_c}.
$$

For the regular 2D block sharding used here, the global dimension must first
be divisible by both process-grid dimensions:

$$
N\bmod P_r=0,
\qquad
N\bmod P_c=0.
$$

No input-matrix padding is required if $T_A$ also divides both local
shard dimensions:

$$
\frac{N}{P_r}\bmod T_A=0,
\qquad
\frac{N}{P_c}\bmod T_A=0.
$$

In this case, JAXMg can proceed to the native redistribution without first
allocating a padded copy of each local shard.

## What happens when the tile size does not divide

If either local dimension is not divisible by $T_A$, the tile-aligned local
capacity required by the downstream 2D redistribution is larger than the
initial JAX shard. JAXMg therefore pads each local matrix to the next tile
boundary:

$$
\Delta_r
=
\left(-N_{\mathrm{local\ rows}}\right)\bmod T_A,
\qquad
\Delta_c
=
\left(-N_{\mathrm{local\ cols}}\right)\bmod T_A.
$$

The padded local shape is

$$
\left(
N_{\mathrm{local\ rows}}+\Delta_r,
N_{\mathrm{local\ cols}}+\Delta_c
\right).
$$

Creating the padded array requires a new allocation while the original local
shard remains live. During this step, the input-matrix storage per
process can therefore reach

$$
N_{\mathrm{local\ rows}}N_{\mathrm{local\ cols}}
+
\left(N_{\mathrm{local\ rows}}+\Delta_r\right)
\left(N_{\mathrm{local\ cols}}+\Delta_c\right)
$$

elements, before accounting for native redistribution scratch or cuSOLVERMp
workspace. Padding is fully supported, but avoiding it is particularly
important for matrices close to the available GPU-memory limit.

## Worked example

For $N=98{,}304$ and a $4\times2$ process grid, every process starts with

$$
\left(
\frac{98{,}304}{4},
\frac{98{,}304}{2}
\right)
=
(24{,}576, 49{,}152)
$$

matrix entries. Tile sizes such as

$$
T_A\in\{256,512,1024,2048,4096\}
$$

divide both local dimensions, so no input-matrix padding is required.
For example, $T_A=1024$ gives a local tile grid of

$$
24\times48.
$$

## Performance and memory trade-off

A tile size should normally satisfy three requirements:

1. Use $T_A\ge128$ as a practical starting point; very small tiles create
   additional solver and communication work.
2. Choose a divisor of both local matrix dimensions to avoid padding $A$.
3. Ensure that the resulting redistribution scratch allocation fits in local
   GPU memory alongside the matrix and cuSOLVERMp workspace.

For square tiles, JAXMg bounds the native redistribution scratch allocation
on each process by

$$
N_{\mathrm{scratch}}
=
3T_A\max\left(
N_{\mathrm{local\ rows}},
N_{\mathrm{local\ cols}}
\right)
$$

elements. Increasing $T_A$ therefore increases scratch memory
linearly, so the best tile size balances solver performance against memory
availability.

See [Memory distribution](../technical_details/memory_distribution.md) for the
layout conversion, padding alignment, 2D block-cyclic redistribution, and
scratch-buffer implementation.
