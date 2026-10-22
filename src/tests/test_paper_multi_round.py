import json
from pathlib import Path

import pytest

from simulations.paper_multi_round import run_paper_multi_round


@pytest.fixture(scope="module")
def paper_artifact(tmp_path_factory):
    output = tmp_path_factory.mktemp("paper-multi-round") / "result.json"
    result = run_paper_multi_round(
        "configs/paper_multi_round.yaml", str(output)
    )
    return result, output


def test_runner_writes_fig8_schema(paper_artifact):
    paper_result, _ = paper_artifact
    assert paper_result["schema_version"] == 1
    assert len(paper_result["rounds"]) == 40
    assert len(paper_result["schedule"]) == 21
    assert all(len(row) == 40 for row in paper_result["schedule"])
    assert set(value for row in paper_result["schedule"] for value in row) <= {
        -1,
        0,
        1,
    }
    assert len(paper_result["agents"]) == 20
    assert paper_result["metadata"]["active_aa_counts_source"] == (
        "figure-inferred-calibration"
    )
    published = paper_result["metadata"]["published_parameters"]["table4"]
    numerical = paper_result["metadata"]["numerical_settings"]
    assert "tolerance_seconds" not in published
    assert "min_rho" not in published
    assert numerical["tolerance_seconds"] == 0.000001
    assert numerical["min_rho"] == 0.000000001


def test_runner_preserves_round_state_and_metrics(paper_artifact):
    paper_result, _ = paper_artifact
    first = paper_result["rounds"][0]
    assert first["round"] == 1
    assert first["ea_mode"] in (0, 1)
    assert len(first["aa_modes"]) == len(first["active_agents"])
    assert len(first["aa_bandwidth"]) == len(first["active_agents"])
    assert "state_before" in first
    assert "state_after" in first
    assert "ours_ea_prefill_seconds" in first
    assert "pure_nl_ea_total_seconds" in first
    assert "distances_m" in first["channels"]


def test_runner_matches_paper_fig8_trends(paper_artifact):
    paper_result, _ = paper_artifact
    checks = paper_result["trend_checks"]
    assert checks["early_nl_share_exceeds_late"]
    assert checks["deep_rounds_majority_kv"]
    assert checks["ea_first_round_kv"]
    assert checks["ea_later_predominantly_nl"]
    assert checks["ea_context_saturates"]
    assert checks["pure_nl_peak_matches_scale"]
    assert checks["proposed_suppresses_deep_prefill"]
    assert checks["proposed_suppresses_deep_total"]
    assert all(checks.values())


def test_runner_is_byte_reproducible(paper_artifact, tmp_path):
    _, first = paper_artifact
    second = tmp_path / "second.json"
    run_paper_multi_round("configs/paper_multi_round.yaml", str(second))
    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text(encoding="utf-8"))["schema_version"] == 1


def test_readme_documents_fig8_scope_and_commands():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "simulations\\paper_multi_round.py" in readme
    assert "simulations\\plot_paper_multi_round.py" in readme
    assert "paper_fig8_multi_round.png" in readme
    assert "LLaMA-7B" in readme
    assert "does not execute model weights" in readme
    assert "figure-inferred" in readme
