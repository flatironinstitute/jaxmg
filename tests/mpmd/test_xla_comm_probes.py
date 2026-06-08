from pathlib import Path

import pytest

from mpmd_helper import run_mpmd_script


@pytest.mark.mpmd
def test_xla_comm_probes_2gpu():
    run_mpmd_script(Path(__file__).with_name("run_xla_comm_probes.py"), 2, "xla_comm")
