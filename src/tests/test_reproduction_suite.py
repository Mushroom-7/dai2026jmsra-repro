import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from simulations.run_paper_reproduction import (
    build_parser,
    create_run_directory,
    load_suite_config,
    run_reproduction_suite,
    select_experiments,
    validate_run_id,
)


def _write_suite(path: Path, project_root: Path) -> Path:
    config_dir = project_root / "configs"
    config_dir.mkdir(parents=True)
    for name in ("mode.yaml", "jmsra.yaml", "multi.yaml"):
        (config_dir / name).write_text("value: 1\n", encoding="utf-8")
    suite = {
        "schema_version": 1,
        "output_root": "results/runs",
        "record_packages": ["numpy", "PyYAML"],
        "experiments": {
            "fig45": {
                "config": "configs/mode.yaml",
                "log": "paper_mode_analysis.json",
            },
            "fig67": {
                "config": "configs/jmsra.yaml",
                "log": "paper_jmsra_calibration.json",
            },
            "fig8": {
                "config": "configs/multi.yaml",
                "log": "paper_multi_round.json",
            },
        },
    }
    path.write_text(yaml.safe_dump(suite, sort_keys=False), encoding="utf-8")
    return path


def test_load_suite_config_resolves_existing_project_paths(tmp_path):
    suite_path = _write_suite(tmp_path / "suite.yaml", tmp_path)

    suite = load_suite_config(suite_path, project_root=tmp_path)

    assert suite["output_root"] == tmp_path / "results" / "runs"
    assert suite["experiments"]["fig45"]["config"] == (
        tmp_path / "configs" / "mode.yaml"
    )


def test_load_suite_config_rejects_unknown_experiment(tmp_path):
    suite_path = _write_suite(tmp_path / "suite.yaml", tmp_path)
    data = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    data["experiments"]["fig9"] = data["experiments"].pop("fig8")
    suite_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError, match="experiments must be exactly"):
        load_suite_config(suite_path, project_root=tmp_path)


@pytest.mark.parametrize("run_id", ["../escape", "a/b", "a\\b", ".hidden", ""])
def test_validate_run_id_rejects_unsafe_values(run_id):
    with pytest.raises(ValueError, match="run ID"):
        validate_run_id(run_id)


def test_select_experiments_supports_groups(tmp_path):
    suite_path = _write_suite(tmp_path / "suite.yaml", tmp_path)
    suite = load_suite_config(suite_path, project_root=tmp_path)

    assert select_experiments(suite, "all") == ["fig45", "fig67", "fig8"]
    assert select_experiments(suite, "fig67") == ["fig67"]


def test_create_run_directory_never_overwrites(tmp_path):
    first = create_run_directory(tmp_path, "baseline")
    assert first == tmp_path / "baseline"

    with pytest.raises(FileExistsError, match="already exists"):
        create_run_directory(tmp_path, "baseline")


def test_run_reproduction_suite_writes_success_manifest(tmp_path):
    suite_path = _write_suite(tmp_path / "suite.yaml", tmp_path)

    def execute(config_path, log_path, figures_dir, make_plots):
        assert config_path.name == "mode.yaml"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text('{"checks": {"paper_trend": true}}\n', encoding="utf-8")
        artifacts = [log_path]
        if make_plots:
            figure = figures_dir / "figure.png"
            figures_dir.mkdir(parents=True, exist_ok=True)
            figure.write_bytes(b"png")
            artifacts.append(figure)
        return {"checks": {"paper_trend": True}}, artifacts

    manifest_path = run_reproduction_suite(
        suite_config_path=suite_path,
        suite="fig45",
        output_root=tmp_path / "runs",
        run_id="baseline",
        project_root=tmp_path,
        executors={"fig45": execute},
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_dir = manifest_path.parent
    assert manifest["status"] == "succeeded"
    assert manifest["selection"] == ["fig45"]
    assert manifest["tasks"]["fig45"]["checks"] == {"paper_trend": True}
    assert manifest["tasks"]["fig45"]["status"] == "succeeded"
    assert (run_dir / "configs" / "suite.yaml").exists()
    assert (run_dir / "configs" / "mode.yaml").exists()
    assert len(manifest["configs"]) == 2
    assert all(len(item["sha256"]) == 64 for item in manifest["configs"])
    assert set(manifest["artifacts"]) == {
        "logs/paper_mode_analysis.json",
        "figures/figure.png",
    }


def test_run_reproduction_suite_persists_failure_manifest(tmp_path):
    suite_path = _write_suite(tmp_path / "suite.yaml", tmp_path)

    def fail(config_path, log_path, figures_dir, make_plots):
        raise RuntimeError("synthetic failure")

    with pytest.raises(RuntimeError, match="synthetic failure"):
        run_reproduction_suite(
            suite_config_path=suite_path,
            suite="fig8",
            output_root=tmp_path / "runs",
            run_id="failed-run",
            project_root=tmp_path,
            executors={"fig8": fail},
        )

    manifest_path = tmp_path / "runs" / "failed-run" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["tasks"]["fig8"]["status"] == "failed"
    assert manifest["error"] == {
        "type": "RuntimeError",
        "message": "synthetic failure",
    }


def test_build_parser_exposes_repeat_experiment_controls():
    parser = build_parser()

    defaults = parser.parse_args([])
    assert defaults.suite == "all"
    assert defaults.config == "configs/reproduction_suite.yaml"
    assert defaults.output_root is None
    assert defaults.run_id is None
    assert defaults.skip_plots is False
    assert defaults.with_tests is False

    selected = parser.parse_args(
        [
            "--suite",
            "fig8",
            "--config",
            "custom.yaml",
            "--output-root",
            "results/runs-alt",
            "--run-id",
            "trial-01",
            "--skip-plots",
            "--with-tests",
        ]
    )
    assert selected.suite == "fig8"
    assert selected.config == "custom.yaml"
    assert selected.output_root == "results/runs-alt"
    assert selected.run_id == "trial-01"
    assert selected.skip_plots is True
    assert selected.with_tests is True


def test_script_bootstraps_project_root_for_direct_execution():
    project_root = Path(__file__).resolve().parents[2]
    runner = project_root / "simulations" / "run_paper_reproduction.py"
    probe = (
        "import runpy; "
        f"runpy.run_path({str(runner)!r}, run_name='runner_probe'); "
        "import simulations.paper_mode_analysis"
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_run_reproduction_suite_records_requested_pytest(tmp_path):
    suite_path = _write_suite(tmp_path / "suite.yaml", tmp_path)

    def execute(config_path, log_path, figures_dir, make_plots):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("{}\n", encoding="utf-8")
        return {}, [log_path]

    def run_tests(project_root):
        assert project_root == tmp_path
        return subprocess.CompletedProcess(
            args=["python", "-m", "pytest"],
            returncode=0,
            stdout="13 passed\n",
            stderr="",
        )

    manifest_path = run_reproduction_suite(
        suite_config_path=suite_path,
        suite="fig45",
        output_root=tmp_path / "runs",
        run_id="with-tests",
        run_tests=True,
        project_root=tmp_path,
        executors={"fig45": execute},
        test_runner=run_tests,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["tests"]["status"] == "succeeded"
    assert manifest["tests"]["returncode"] == 0
    assert manifest["tests"]["artifact"] == "logs/pytest.txt"
    assert "logs/pytest.txt" in manifest["artifacts"]
    assert (manifest_path.parent / "logs" / "pytest.txt").read_text(
        encoding="utf-8"
    ) == "13 passed\n"


def test_run_reproduction_suite_fails_when_requested_pytest_fails(tmp_path):
    suite_path = _write_suite(tmp_path / "suite.yaml", tmp_path)

    def execute(config_path, log_path, figures_dir, make_plots):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("{}\n", encoding="utf-8")
        return {}, [log_path]

    def fail_tests(project_root):
        return subprocess.CompletedProcess(
            args=["python", "-m", "pytest"],
            returncode=1,
            stdout="1 failed\n",
            stderr="failure details\n",
        )

    with pytest.raises(RuntimeError, match="pytest failed with exit code 1"):
        run_reproduction_suite(
            suite_config_path=suite_path,
            suite="fig45",
            output_root=tmp_path / "runs",
            run_id="failed-tests",
            run_tests=True,
            project_root=tmp_path,
            executors={"fig45": execute},
            test_runner=fail_tests,
        )

    manifest = json.loads(
        (tmp_path / "runs" / "failed-tests" / "run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "failed"
    assert manifest["tests"]["status"] == "failed"
    assert manifest["tests"]["returncode"] == 1
