import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENT_NAMES = ("fig45", "fig67", "fig8")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
Executor = Callable[[Path, Path, Path, bool], tuple[dict, list[Path]]]
TestRunner = Callable[[Path], subprocess.CompletedProcess[str]]


def _project_path(path: str | Path, project_root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def load_suite_config(
    config_path: str | Path = "configs/reproduction_suite.yaml",
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict:
    root = Path(project_root).resolve()
    path = _project_path(config_path, root)
    if not path.is_file():
        raise FileNotFoundError(f"suite config does not exist: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("suite config must contain a mapping")
    if data.get("schema_version") != 1:
        raise ValueError("suite config schema_version must be 1")

    experiments = data.get("experiments")
    if not isinstance(experiments, dict) or set(experiments) != set(
        EXPERIMENT_NAMES
    ):
        names = ", ".join(EXPERIMENT_NAMES)
        raise ValueError(f"experiments must be exactly: {names}")

    resolved_experiments = {}
    for name in EXPERIMENT_NAMES:
        item = experiments[name]
        if not isinstance(item, dict):
            raise ValueError(f"experiments.{name} must be a mapping")
        config = _project_path(item.get("config", ""), root)
        if not config.is_file():
            raise FileNotFoundError(
                f"experiments.{name}.config does not exist: {config}"
            )
        log_name = item.get("log")
        if (
            not isinstance(log_name, str)
            or not log_name.endswith(".json")
            or Path(log_name).name != log_name
        ):
            raise ValueError(
                f"experiments.{name}.log must be a JSON filename"
            )
        resolved_experiments[name] = {"config": config, "log": log_name}

    output_root = data.get("output_root")
    if not isinstance(output_root, str) or not output_root.strip():
        raise ValueError("output_root must be a nonempty path")
    packages = data.get("record_packages", [])
    if not isinstance(packages, list) or not all(
        isinstance(item, str) and item for item in packages
    ):
        raise ValueError("record_packages must be a list of package names")

    return {
        "schema_version": 1,
        "path": path,
        "output_root": _project_path(output_root, root),
        "record_packages": packages,
        "experiments": resolved_experiments,
    }


def select_experiments(suite_config: dict, suite: str) -> list[str]:
    if suite == "all":
        return list(EXPERIMENT_NAMES)
    if suite not in EXPERIMENT_NAMES:
        choices = ", ".join(("all", *EXPERIMENT_NAMES))
        raise ValueError(f"suite must be one of: {choices}")
    if suite not in suite_config["experiments"]:
        raise ValueError(f"suite config does not define {suite}")
    return [suite]


def validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run ID must start with an alphanumeric character and contain "
            "only letters, digits, dots, underscores, or hyphens"
        )
    return run_id


def create_run_directory(output_root: str | Path, run_id: str) -> Path:
    validated = validate_run_id(run_id)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / validated
    try:
        run_dir.mkdir()
    except FileExistsError as error:
        raise FileExistsError(f"run directory already exists: {run_dir}") from error
    return run_dir


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_output(project_root: Path, *arguments: str) -> str | None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _git_provenance(project_root: Path) -> dict:
    status = _git_output(project_root, "status", "--porcelain")
    return {
        "commit": _git_output(project_root, "rev-parse", "HEAD"),
        "branch": _git_output(project_root, "branch", "--show-current"),
        "dirty": None if status is None else bool(status),
    }


def _environment_provenance(package_names: list[str]) -> dict:
    packages = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
    }


def _snapshot_config(source: Path, destination: Path, run_dir: Path) -> dict:
    if destination.exists():
        raise ValueError(f"duplicate config snapshot filename: {destination.name}")
    shutil.copy2(source, destination)
    return {
        "source": str(source),
        "snapshot": destination.relative_to(run_dir).as_posix(),
        "sha256": _sha256(destination),
    }


def _relative_artifact(path: Path, run_dir: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(run_dir.resolve())
    except ValueError as error:
        raise ValueError(f"artifact is outside run directory: {resolved}") from error
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise RuntimeError(f"artifact is missing or empty: {resolved}")
    return relative.as_posix()


def _extract_checks(result: dict) -> dict[str, bool]:
    checks = result.get("checks", result.get("trend_checks", {}))
    if not isinstance(checks, dict) or not all(
        isinstance(name, str) and isinstance(value, bool)
        for name, value in checks.items()
    ):
        raise ValueError("experiment checks must be a mapping of booleans")
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError("paper trend checks failed: " + ", ".join(failures))
    return checks


def _execute_fig45(
    config_path: Path,
    log_path: Path,
    figures_dir: Path,
    make_plots: bool,
) -> tuple[dict, list[Path]]:
    from simulations.paper_mode_analysis import run_paper_mode_analysis
    from simulations.plot_paper_mode_analysis import plot_paper_mode_analysis

    result = run_paper_mode_analysis(str(config_path), str(log_path))
    artifacts = [log_path]
    if make_plots:
        artifacts.extend(
            Path(path)
            for path in plot_paper_mode_analysis(
                str(log_path), str(figures_dir)
            )
        )
    return result, artifacts


def _execute_fig67(
    config_path: Path,
    log_path: Path,
    figures_dir: Path,
    make_plots: bool,
) -> tuple[dict, list[Path]]:
    from simulations.paper_jmsra_calibration import (
        run_paper_jmsra_calibration,
    )
    from simulations.plot_paper_jmsra import plot_paper_jmsra

    result = run_paper_jmsra_calibration(str(config_path), str(log_path))
    artifacts = [log_path]
    if make_plots:
        artifacts.extend(
            Path(path)
            for path in plot_paper_jmsra(str(log_path), str(figures_dir))
        )
    return result, artifacts


def _execute_fig8(
    config_path: Path,
    log_path: Path,
    figures_dir: Path,
    make_plots: bool,
) -> tuple[dict, list[Path]]:
    from simulations.paper_multi_round import run_paper_multi_round
    from simulations.plot_paper_multi_round import plot_paper_multi_round

    result = run_paper_multi_round(str(config_path), str(log_path))
    artifacts = [log_path]
    if make_plots:
        output = figures_dir / "paper_fig8_multi_round.png"
        artifacts.append(
            Path(plot_paper_multi_round(str(log_path), str(output)))
        )
    return result, artifacts


def _default_executors() -> dict[str, Executor]:
    return {
        "fig45": _execute_fig45,
        "fig67": _execute_fig67,
        "fig8": _execute_fig8,
    }


def _run_pytest(project_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )


def _pytest_output(completed: subprocess.CompletedProcess[str]) -> str:
    output = completed.stdout or ""
    error_output = completed.stderr or ""
    if error_output:
        if output and not output.endswith("\n"):
            output += "\n"
        output += error_output
    return output or "pytest completed without output\n"


def run_reproduction_suite(
    suite_config_path: str | Path = "configs/reproduction_suite.yaml",
    *,
    suite: str = "all",
    output_root: str | Path | None = None,
    run_id: str | None = None,
    make_plots: bool = True,
    run_tests: bool = False,
    project_root: Path = PROJECT_ROOT,
    executors: dict[str, Executor] | None = None,
    test_runner: TestRunner | None = None,
    command: list[str] | None = None,
) -> Path:
    root = Path(project_root).resolve()
    suite_config = load_suite_config(suite_config_path, project_root=root)
    selection = select_experiments(suite_config, suite)
    selected_run_id = validate_run_id(run_id or _default_run_id())
    selected_output_root = (
        suite_config["output_root"]
        if output_root is None
        else _project_path(output_root, root)
    )
    run_dir = create_run_directory(selected_output_root, selected_run_id)
    logs_dir = run_dir / "logs"
    figures_dir = run_dir / "figures"
    configs_dir = run_dir / "configs"
    logs_dir.mkdir()
    figures_dir.mkdir()
    configs_dir.mkdir()

    config_records = [
        {
            "role": "suite",
            **_snapshot_config(
                suite_config["path"],
                configs_dir / suite_config["path"].name,
                run_dir,
            ),
        }
    ]
    snapshot_paths = {}
    for name in selection:
        source = suite_config["experiments"][name]["config"]
        destination = configs_dir / source.name
        config_records.append(
            {
                "role": name,
                **_snapshot_config(source, destination, run_dir),
            }
        )
        snapshot_paths[name] = destination

    manifest_path = run_dir / "run_manifest.json"
    started_at = _utc_now()
    started_clock = time.perf_counter()
    manifest = {
        "schema_version": 1,
        "run_id": selected_run_id,
        "status": "running",
        "started_at": started_at,
        "finished_at": None,
        "duration_seconds": None,
        "command": list(command if command is not None else sys.argv),
        "selection": selection,
        "options": {"make_plots": make_plots, "run_tests": run_tests},
        "git": _git_provenance(root),
        "environment": _environment_provenance(
            suite_config["record_packages"]
        ),
        "configs": config_records,
        "tasks": {},
        "tests": {
            "requested": run_tests,
            "status": "pending" if run_tests else "not_requested",
        },
        "artifacts": [],
        "error": None,
    }
    _write_manifest(manifest_path, manifest)

    available_executors = _default_executors()
    if executors is not None:
        available_executors.update(executors)
    current_task = None
    try:
        for name in selection:
            current_task = name
            task_started = time.perf_counter()
            manifest["tasks"][name] = {
                "status": "running",
                "duration_seconds": None,
                "checks": {},
                "artifacts": [],
            }
            _write_manifest(manifest_path, manifest)
            item = suite_config["experiments"][name]
            result, artifact_paths = available_executors[name](
                snapshot_paths[name],
                logs_dir / item["log"],
                figures_dir,
                make_plots,
            )
            checks = _extract_checks(result)
            artifacts = [
                _relative_artifact(Path(path), run_dir)
                for path in artifact_paths
            ]
            manifest["tasks"][name] = {
                "status": "succeeded",
                "duration_seconds": round(
                    time.perf_counter() - task_started, 6
                ),
                "checks": checks,
                "artifacts": artifacts,
            }
            manifest["artifacts"].extend(artifacts)
            _write_manifest(manifest_path, manifest)
            current_task = None

        if run_tests:
            tests_started = time.perf_counter()
            manifest["tests"] = {
                "requested": True,
                "status": "running",
                "returncode": None,
                "duration_seconds": None,
                "artifact": None,
            }
            _write_manifest(manifest_path, manifest)
            completed = (test_runner or _run_pytest)(root)
            pytest_log = logs_dir / "pytest.txt"
            pytest_log.write_text(_pytest_output(completed), encoding="utf-8")
            pytest_artifact = _relative_artifact(pytest_log, run_dir)
            manifest["tests"].update(
                {
                    "status": (
                        "succeeded" if completed.returncode == 0 else "failed"
                    ),
                    "returncode": completed.returncode,
                    "duration_seconds": round(
                        time.perf_counter() - tests_started, 6
                    ),
                    "artifact": pytest_artifact,
                }
            )
            manifest["artifacts"].append(pytest_artifact)
            _write_manifest(manifest_path, manifest)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"pytest failed with exit code {completed.returncode}"
                )

        manifest["status"] = "succeeded"
        return manifest_path
    except Exception as error:
        if current_task is not None:
            task = manifest["tasks"][current_task]
            task["status"] = "failed"
            task["duration_seconds"] = round(
                time.perf_counter() - task_started, 6
            )
        manifest["status"] = "failed"
        manifest["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        raise
    finally:
        manifest["finished_at"] = _utc_now()
        manifest["duration_seconds"] = round(
            time.perf_counter() - started_clock, 6
        )
        _write_manifest(manifest_path, manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the paper reproduction experiments with isolated outputs "
            "and an auditable manifest."
        )
    )
    parser.add_argument(
        "--suite",
        choices=("all", *EXPERIMENT_NAMES),
        default="all",
        help="Experiment group to run (default: all).",
    )
    parser.add_argument(
        "--config",
        default="configs/reproduction_suite.yaml",
        help="Top-level reproduction suite YAML.",
    )
    parser.add_argument(
        "--output-root",
        help="Run directory parent (default: value from suite YAML).",
    )
    parser.add_argument(
        "--run-id",
        help="Portable unique run ID (default: current UTC timestamp).",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Generate numerical logs without PNG figures.",
    )
    parser.add_argument(
        "--with-tests",
        action="store_true",
        help="Run the complete pytest suite after the experiments.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    manifest_path = run_reproduction_suite(
        suite_config_path=arguments.config,
        suite=arguments.suite,
        output_root=arguments.output_root,
        run_id=arguments.run_id,
        make_plots=not arguments.skip_plots,
        run_tests=arguments.with_tests,
        command=(
            list(sys.argv)
            if argv is None
            else [str(Path(__file__)), *argv]
        ),
    )
    print(f"Reproduction succeeded: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
