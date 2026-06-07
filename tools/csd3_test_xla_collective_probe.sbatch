#!/usr/bin/env bash
#SBATCH --job-name=jaxmg_xla_probe_test
#SBATCH --partition=ampere
#SBATCH --account=acedo-sl2-gpu
#SBATCH --qos=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=jaxmg_xla_probe_test_%j.out

set -euo pipefail

module purge
module load gcc/11 cuda/12.1 cudnn/8.9_cuda-12.1

ROOT="${SLURM_SUBMIT_DIR}"
ENV_DIR="/rds/user/jlt67/hpc-work/PhD/JAXMG/conda-env"

export PYTHONNOUSERSITE=1
export JAX_PLATFORMS=cuda

NVIDIA_LIB_ROOT="${ENV_DIR}/lib/python3.11/site-packages/nvidia"
if [[ -d "${NVIDIA_LIB_ROOT}" ]]; then
  for lib_dir in "${NVIDIA_LIB_ROOT}"/*/lib; do
    if [[ -d "${lib_dir}" ]]; then
      export LD_LIBRARY_PATH="${lib_dir}:${LD_LIBRARY_PATH:-}"
    fi
  done
fi

cd "${ROOT}"
"${ENV_DIR}/bin/python" - <<'PY'
import traceback

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import jaxmg


def show(name, fn):
    print(f"\n=== {name} ===", flush=True)
    try:
        value = fn()
        value.block_until_ready()
        host = np.asarray(value)
        print(host.tolist(), flush=True)
        return host
    except Exception:
        traceback.print_exc()
        return None


print("python:", __import__("sys").executable, flush=True)
print("jax:", jax.__version__, flush=True)
print("devices:", jax.devices(), flush=True)
print("local_device_count:", jax.local_device_count(), flush=True)
print("device_count:", jax.device_count(), flush=True)

token = jnp.arange(4, dtype=jnp.int32)
if jax.local_device_count() == 1:
    show("collective direct probe", lambda: jaxmg.xla_comm_collective_probe(token))
else:
    print("\n=== collective direct probe ===", flush=True)
    print("skipped for multi-GPU: collective clique execution belongs inside shard_map", flush=True)

devices = np.array(jax.devices())
mesh = Mesh(devices, ("x",))
sharded_token = jax.device_put(
    jnp.arange(devices.size, dtype=jnp.int32), NamedSharding(mesh, P("x"))
)
sharded_u32 = jax.device_put(
    jnp.arange(1, devices.size + 1, dtype=jnp.uint32), NamedSharding(mesh, P("x"))
)
vector_width = 8
vector_payload = jax.device_put(
    jnp.arange(devices.size * vector_width, dtype=jnp.uint32),
    NamedSharding(mesh, P("x")),
)

collective = show(
    "collective shard_map probe "
    "[status, cuda_device, local_device_id, global_device_id, local_device_count, "
    "has_platform_handle, token_nbytes, cuda_device_count]",
    lambda: jaxmg.xla_comm_collective_probe_shardmap(sharded_token, mesh, P("x")),
)

if collective is None:
    raise SystemExit(10)

rows = collective.reshape((-1, 8))
print("\ncollective rows:", rows.tolist(), flush=True)
if not np.all(rows[:, 0] == 0):
    raise SystemExit(f"collective probe returned nonzero status: {rows[:, 0].tolist()}")
if not np.all(rows[:, 5] == 1):
    raise SystemExit(
        "collective probe did not see a non-null platform communicator handle: "
        f"{rows[:, 5].tolist()}"
    )

allreduce = show(
    "all-reduce shard_map probe",
    lambda: jaxmg.xla_comm_allreduce_probe_shardmap(sharded_u32, mesh, P("x")),
)
if allreduce is None:
    raise SystemExit(11)

expected_sum = np.uint32(devices.size * (devices.size + 1) // 2)
expected = np.full((devices.size,), expected_sum, dtype=np.uint32)
if not np.array_equal(allreduce, expected):
    raise SystemExit(
        f"all-reduce probe returned {allreduce.tolist()}, expected {expected.tolist()}"
    )

ring = show(
    "ring-permute shard_map probe",
    lambda: jaxmg.xla_comm_ring_permute_probe_shardmap(sharded_u32, mesh, P("x")),
)
if ring is None:
    raise SystemExit(12)

expected_ring = np.roll(np.arange(1, devices.size + 1, dtype=np.uint32), 1)
if not np.array_equal(ring, expected_ring):
    raise SystemExit(
        f"ring-permute probe returned {ring.tolist()}, expected {expected_ring.tolist()}"
    )

for shift in (1, 2):
    shifted = show(
        f"shift-permute shard_map probe shift={shift}",
        lambda shift=shift: jaxmg.xla_comm_shift_permute_probe_shardmap(
            sharded_u32, mesh, P("x"), shift=shift
        ),
    )
    if shifted is None:
        raise SystemExit(13)

    expected_shift = np.roll(
        np.arange(1, devices.size + 1, dtype=np.uint32), shift % devices.size
    )
    if not np.array_equal(shifted, expected_shift):
        raise SystemExit(
            f"shift-permute probe shift={shift} returned {shifted.tolist()}, "
            f"expected {expected_shift.tolist()}"
        )

for shift in (1, 2):
    shifted_vector = show(
        f"shift-permute vector payload probe shift={shift}",
        lambda shift=shift: jaxmg.xla_comm_shift_permute_probe_shardmap(
            vector_payload, mesh, P("x"), shift=shift
        ),
    )
    if shifted_vector is None:
        raise SystemExit(14)

    vector_chunks = np.arange(
        devices.size * vector_width, dtype=np.uint32
    ).reshape(devices.size, vector_width)
    expected_vector = np.roll(vector_chunks, shift % devices.size, axis=0).reshape(-1)
    if not np.array_equal(shifted_vector, expected_vector):
        raise SystemExit(
            f"shift-permute vector probe shift={shift} returned "
            f"{shifted_vector.tolist()}, expected {expected_vector.tolist()}"
        )

if devices.size == 1:
    targets = np.array([0], dtype=np.int64)
elif devices.size == 2:
    targets = np.array([1, 0], dtype=np.int64)
elif devices.size == 4:
    targets = np.array([2, 0, 3, 1], dtype=np.int64)
else:
    targets = (np.arange(devices.size, dtype=np.int64) + 1) % devices.size

planned = show(
    f"explicit target-map permute vector probe targets={targets.tolist()}",
    lambda: jaxmg.xla_comm_permute_probe_shardmap(
        vector_payload, mesh, P("x"), targets=targets
    ),
)
if planned is None:
    raise SystemExit(15)

vector_chunks = np.arange(
    devices.size * vector_width, dtype=np.uint32
).reshape(devices.size, vector_width)
expected_chunks = np.empty_like(vector_chunks)
for source, target in enumerate(targets.tolist()):
    expected_chunks[target] = vector_chunks[source]
expected_planned = expected_chunks.reshape(-1)
if not np.array_equal(planned, expected_planned):
    raise SystemExit(
        f"explicit target-map permute probe returned {planned.tolist()}, "
        f"expected {expected_planned.tolist()}"
    )

chunk_count = 3
closed_targets = (np.arange(devices.size, dtype=np.int64) + 1) % devices.size
closed_src_offsets = np.full(devices.size, 1, dtype=np.int64)
closed_dst_offsets = np.full(devices.size, 4, dtype=np.int64)
chunked_closed = show(
    f"chunk-permute closed vector probe targets={closed_targets.tolist()}",
    lambda: jaxmg.xla_comm_chunk_permute_probe_shardmap(
        vector_payload,
        mesh,
        P("x"),
        targets=closed_targets,
        src_offsets=closed_src_offsets,
        dst_offsets=closed_dst_offsets,
        count=chunk_count,
    ),
)
if chunked_closed is None:
    raise SystemExit(16)

expected_closed = vector_chunks.copy()
for source, target in enumerate(closed_targets.tolist()):
    expected_closed[target, closed_dst_offsets[source]:closed_dst_offsets[source] + chunk_count] = (
        vector_chunks[source, closed_src_offsets[source]:closed_src_offsets[source] + chunk_count]
    )
expected_closed = expected_closed.reshape(-1)
if not np.array_equal(chunked_closed, expected_closed):
    raise SystemExit(
        f"chunk-permute closed vector probe returned {chunked_closed.tolist()}, "
        f"expected {expected_closed.tolist()}"
    )

if devices.size >= 2:
    open_count = 2
    open_targets = np.full(devices.size, -1, dtype=np.int64)
    open_targets[0] = devices.size - 1
    open_src_offsets = np.zeros(devices.size, dtype=np.int64)
    open_dst_offsets = np.full(devices.size, 5, dtype=np.int64)
    chunked_open = show(
        f"chunk-permute open vector probe targets={open_targets.tolist()}",
        lambda: jaxmg.xla_comm_chunk_permute_probe_shardmap(
            vector_payload,
            mesh,
            P("x"),
            targets=open_targets,
            src_offsets=open_src_offsets,
            dst_offsets=open_dst_offsets,
            count=open_count,
        ),
    )
    if chunked_open is None:
        raise SystemExit(17)

    expected_open = vector_chunks.copy()
    expected_open[
        devices.size - 1,
        open_dst_offsets[0]:open_dst_offsets[0] + open_count,
    ] = vector_chunks[0, open_src_offsets[0]:open_src_offsets[0] + open_count]
    expected_open = expected_open.reshape(-1)
    if not np.array_equal(chunked_open, expected_open):
        raise SystemExit(
            f"chunk-permute open vector probe returned {chunked_open.tolist()}, "
            f"expected {expected_open.tolist()}"
        )

print("\nPASS: XLA collective FFI contexts and platform communicator handle are visible.", flush=True)
print("PASS: XLA communicator all-reduce probe produced the expected result.", flush=True)
print("PASS: XLA communicator ring-permute probe produced the expected result.", flush=True)
print("PASS: XLA communicator shift-permute probe produced the expected result.", flush=True)
print("PASS: XLA communicator vector and explicit target-map probes produced the expected result.", flush=True)
print("PASS: XLA communicator chunk-permute probes produced the expected result.", flush=True)
PY
