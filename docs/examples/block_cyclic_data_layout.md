# 2D block-cyclic data layout

JAX arrays normally enter JAXMg as large rectangular shards. cuSOLVERMp instead
assigns square tiles cyclically over a two-dimensional process grid.

For a tile at global tile coordinate $(i,j)$, the owner is

$$
\operatorname{owner}(i,j)
=
\left(
  i \bmod P_r,
  j \bmod P_c
\right),
$$

where $P_r$ and $P_c$ are the process-grid dimensions.

For a $2\times2$ process grid and a $4\times4$ tile matrix, ownership is:

$$
\begin{bmatrix}
P_{00} & P_{01} & P_{00} & P_{01} \\
P_{10} & P_{11} & P_{10} & P_{11} \\
P_{00} & P_{01} & P_{00} & P_{01} \\
P_{10} & P_{11} & P_{10} & P_{11}
\end{bmatrix}.
$$

JAXMg reaches this layout in two phases:

```text
JAX rectangular shards
        |
        | column-owner phase within each process row
        v
every tile has the correct process column
        |
        | row-owner phase within each process column
        v
cuSOLVERMp 2D block-cyclic ownership
```

The column-owner phase can operate independently across process rows. The
row-owner phase can operate independently across process columns. Each movement
wave transfers whole tile slabs rather than individual matrix elements.

Padding is handled before this redistribution. Real matrix entries are aligned
to the global top-left, leaving any padded capacity on the global right and
bottom edges. This makes the padded matrix tile-addressable without exposing
the layout transformation to users.

See [Memory distribution](../technical_details/memory_distribution.md) for the
complete native workflow, movement diagrams, scratch calculation, and reverse
path.
