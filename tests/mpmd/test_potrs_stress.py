from pathlib import Path

import pytest

from mpmd_helper import run_mpmd_script


@pytest.mark.mpmd
def test_potrs_stress_2gpu():
    run_mpmd_script(Path(__file__).with_name("run_potrs_stress.py"), 2, "potrs_stress")


@pytest.mark.mpmd
def test_potrs_stress_4gpu():
    run_mpmd_script(Path(__file__).with_name("run_potrs_stress.py"), 4, "potrs_stress")
