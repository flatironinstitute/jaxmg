"""Run an unchanged example with per-rank progress and timeout stack traces."""

import faulthandler
import linecache
import os
from pathlib import Path
import runpy
import sys
import time


script = str(Path(sys.argv[1]).resolve())
sys.argv = sys.argv[1:]
sys.path.insert(0, str(Path(script).parent))
started = time.monotonic()


def trace_main(frame, event, arg):
    """Record main's last Python statement without tracing JAX internals."""
    if frame.f_code.co_filename != script or frame.f_code.co_name != "main":
        return None
    if event == "line":
        source = linecache.getline(script, frame.f_lineno).strip()
        print(
            f"[example +{time.monotonic() - started:.3f}s] "
            f"{Path(script).name}:{frame.f_lineno}: {source}",
            flush=True,
        )
    elif event in ("return", "exception"):
        print(f"[example] main {event}", flush=True)
    return trace_main


faulthandler.enable()
faulthandler.dump_traceback_later(
    0.8 * float(os.environ.get("JAXMG_GPU_TEST_TIMEOUT", "300")), repeat=True
)
print(f"[example] loading {script}", flush=True)
sys.settrace(trace_main)
try:
    runpy.run_path(script, run_name="__main__")
finally:
    sys.settrace(None)
# Keep the stack timer active through interpreter shutdown.
print("[example] script returned; entering interpreter shutdown", flush=True)
