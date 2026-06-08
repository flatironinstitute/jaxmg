from pathlib import Path

import pytest

from mpmd_helper import run_mpmd_script


@pytest.mark.mpmd
def test_potri_2gpu():
    run_mpmd_script(Path(__file__).with_name("run_potri.py"), 2, "potri")


@pytest.mark.mpmd
def test_potri_4gpu():
    run_mpmd_script(Path(__file__).with_name("run_potri.py"), 4, "potri")
