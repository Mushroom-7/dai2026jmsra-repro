import json
from copy import deepcopy

from simulations.paper_mode_analysis import (
    _fig4_checks,
    _has_single_valley,
    _surface_alpha_decreases,
    run_paper_mode_analysis,
)


def test_paper_mode_runner_writes_versioned_reproducible_result(tmp_path):
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first = run_paper_mode_analysis(output_path=str(first_path))
    second = run_paper_mode_analysis(output_path=str(second_path))

    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()
    assert json.loads(first_path.read_text(encoding="utf-8")) == first
    assert first["metadata"]["schema_version"] == 1
    assert first["metadata"]["assumptions"]["fig4_b_input_tokens"] == 512
    assert set(first["fig4"]) == {
        "panel_a",
        "panel_b",
        "panel_c",
        "panel_d",
    }
    assert set(first["fig5"]) == {"panel_a", "panel_b", "panel_c"}


def test_paper_mode_runner_confirms_paper_trends(tmp_path):
    result = run_paper_mode_analysis(
        output_path=str(tmp_path / "result.json")
    )
    checks = result["checks"]

    assert checks["fig4_has_nl_and_kv_regions"]
    assert checks["fig4_a_nonmonotonic"]
    assert checks["fig4_c_increases_with_snr"]
    assert checks["fig4_d_increases_with_aa_count"]
    assert checks["fig4_b_homogeneous_moves_toward_nl"]
    assert checks["fig5_scores_increase_with_rho"]
    assert checks["fig5_xi_increases_threshold"]
    assert checks["fig5_snr_decreases_threshold"]
    assert checks["fig5_theta_decreases_threshold"]
    assert checks["fig5_alpha_decreases_threshold"]

    panel_d = result["fig4"]["panel_d"]["series"]
    low_snr = [series for series in panel_d if series["snr_db"] == 0]
    high_snr = [series for series in panel_d if series["snr_db"] > 0]
    assert all(
        min(series["ratios"]) < series["ratios"][0]
        and series["ratios"][-1] > min(series["ratios"])
        for series in low_snr
    )
    assert all(
        all(left < right for left, right in zip(ratios, ratios[1:]))
        for ratios in (series["ratios"] for series in high_snr)
    )

    for surface in result["fig5"]["panel_c"]["surfaces"]:
        for row in surface["thresholds"]:
            present = [value for value in row if value is not None]
            if len(present) < 2:
                continue
            decreases = sum(
                left > right for left, right in zip(present, present[1:])
            )
            assert present[-1] < present[0]
            assert decreases / (len(present) - 1) >= 0.9


def test_fig4_trend_checks_reject_oscillation_and_require_single_valley(tmp_path):
    assert _has_single_valley([3.0, 2.0, 1.0, 2.0, 4.0])
    assert not _has_single_valley([3.0, 1.0, 2.0, 1.5, 4.0])

    result = run_paper_mode_analysis(output_path=str(tmp_path / "result.json"))
    fig4 = deepcopy(result["fig4"])
    homogeneous = next(
        series
        for series in fig4["panel_b"]["series"]
        if series["system"] == "homogeneous"
    )
    homogeneous["ratios"] = [2.0, 3.0, 1.0]
    assert not _fig4_checks(fig4)["fig4_b_homogeneous_moves_toward_nl"]


def test_fig5_surface_check_rejects_holes_and_sparse_rows():
    assert _surface_alpha_decreases(
        {"thresholds": [[None, 0.8, 0.6, 0.5], [0.9, 0.7, 0.6, 0.5]]}
    )
    assert not _surface_alpha_decreases(
        {"thresholds": [[0.8, None, 0.6, 0.5]]}
    )
    assert not _surface_alpha_decreases(
        {"thresholds": [[None, None, None, 0.5]]}
    )
