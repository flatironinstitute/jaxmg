from pathlib import Path

import pytest

from mpmd_helper import run_mpmd_script


@pytest.mark.mpmd
def test_syevd_2gpu():
    run_mpmd_script(Path(__file__).with_name("run_syevd.py"), 2, "syevd")


@pytest.mark.mpmd
def test_syevd_4gpu():
    run_mpmd_script(Path(__file__).with_name("run_syevd.py"), 4, "syevd")
