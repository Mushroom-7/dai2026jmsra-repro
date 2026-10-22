import sys
from pathlib import Path


def test_tests_run_inside_project_virtualenv():
    assert sys.prefix != sys.base_prefix
    assert Path(sys.prefix).name == ".venv"
