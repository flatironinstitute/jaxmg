from pathlib import Path

import pytest

from mpmd_helper import run_mpmd_script


@pytest.mark.mpmd
def test_potrs_2gpu():
    run_mpmd_script(Path(__file__).with_name("run_potrs.py"), 2, "potrs")
