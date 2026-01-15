# Column major layout expectations

This note explains the difference between row-major and column-major memory
layouts, why cuSolverMg requires column-major data, how asking JAX to present
column-major memory via `input_layouts=(1,0)` forces a copy (and therefore
temporarily doubles memory), and how we can instead choose a sharding that
avoids the extra copy by presenting per-device buffers already in the solver's
expected column-major order.

Row-major vs column-major
- Row-major (C order): rows are laid out contiguously. For an $m\times n$
	matrix $A$ the memory is the concatenation of rows $1,2,\dots,m$.
- Column-major (Fortran order): columns are laid out contiguously. The
	memory is the concatenation of columns $1,2,\dots,n$.

Concretely, for a small $4\times 4$ matrix

$$
A = \begin{bmatrix}
a_{11} & a_{12} & a_{13} & a_{14} \\
a_{21} & a_{22} & a_{23} & a_{24} \\
a_{31} & a_{32} & a_{33} & a_{34} \\
a_{41} & a_{42} & a_{43} & a_{44}
\end{bmatrix}
$$

- Row-major flattening (C order) produces

$$
\operatorname{row\_major}(A) = [a_{11}, a_{12}, a_{13}, a_{14},\; a_{21}, a_{22},\dots,a_{44}]
$$

- Column-major flattening (Fortran order) produces

$$
\operatorname{col\_major}(A) = [a_{11}, a_{21}, a_{31}, a_{41},\; a_{12}, a_{22},\dots,a_{44}]
$$

Why this matters for cuSolverMg
- cuSolverMg (like many Fortran-style numerical libraries) expects matrix
	buffers in column-major order. If the Python/JAX side stores matrices in
	row-major order, the native call must be given a column-major view.

Using `input_layouts=(1,0)`
- When you construct an FFI call with `input_layouts=(1,0)` you are telling
	JAX to present the native function with the axes swapped (i.e. the memory
	should look column-major to the native code). JAX will therefore allocate a
	new buffer and copy/reorder the data into the requested layout before the
	call.
- Practically this means: original buffer (size $N$) + reordered buffer
	(size $N$) exist at peak, so the memory used for this array is roughly
	doubled during the call. For large matrices this spike can be prohibitive.

Avoiding the copy by matching sharding to column-major
- Instead of asking JAX to transpose the whole matrix, we can arrange the
	per-device sharding so that the collection of device-local buffers already
	matches the solver's expected column-major ordering. The key idea is to map
	global columns (or column tiles) to contiguous device-local memory.

Example: split into two column-blocks

Split $A$ into two column-blocks $B$ and $C$ (each $4\times 2$):

$$
A = [\; B \;|\; C \;],\quad
B=\begin{bmatrix} a_{11} & a_{12} \\
a_{21} & a_{22} \\
a_{31} & a_{32} \\
a_{41} & a_{42}\end{bmatrix},\quad
C=\begin{bmatrix} a_{13} & a_{14} \\
a_{23} & a_{24} \\
a_{33} & a_{34} \\
a_{43} & a_{44}\end{bmatrix}
$$

In column-major flattening, $B$ appears as a contiguous chunk followed by $C$:

$$
\operatorname{col\_major}(A) = [\operatorname{col\_major}(B),\; \operatorname{col\_major}(C)].
$$

If device 0 stores $B$ and device 1 stores $C$ in their native local buffers
then the solver can consume device-local memory directly in the expected
column-major order without a global transpose or full-array copy.

