import subprocess
import sys
from pathlib import Path

from simulations.multi_round import run_multi_round
from simulations.plot_results import plot_multi_round, plot_single_round
from simulations.single_round import run_single_round


def test_single_round_script_runs_from_project_root():
    project_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "simulations/single_round.py"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_multi_round_writes_schedule(tmp_path):
    output = tmp_path / "multi_round.json"
    result = run_multi_round("configs/sim_params.yaml", str(output))
    assert output.exists()
    assert len(result["rounds"]) > 0
    assert {
        "round",
        "active_agents",
        "latency_seconds",
        "modes",
    }.issubset(result["rounds"][0])


def test_plot_scripts_generate_pngs(tmp_path):
    single = tmp_path / "single_round.json"
    multi = tmp_path / "multi_round.json"
    single_png = tmp_path / "single_round.png"
    multi_png = tmp_path / "multi_round.png"
    run_single_round("configs/sim_params.yaml", str(single))
    run_multi_round("configs/sim_params.yaml", str(multi))
    assert Path(plot_single_round(str(single), str(single_png))).exists()
    assert Path(plot_multi_round(str(multi), str(multi_png))).exists()
