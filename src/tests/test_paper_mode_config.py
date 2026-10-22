import copy

import pytest

from src.utils.data_gen import (
    load_config,
    validate_paper_mode_analysis_config,
)


def test_paper_mode_config_matches_disclosed_parameters_and_assumption():
    config = load_config("configs/paper_mode_analysis.yaml")
    validate_paper_mode_analysis_config(config)
    analysis = config["paper_mode_analysis"]
    assert analysis["schema_version"] == 1
    assert analysis["bits_per_token"] == 16
    assert analysis["kv_cache_compression_ratio"] == 2
    assert analysis["bandwidth_ghz"] == 2
    assert analysis["fig4"]["default_aa_count"] == 5
    assert analysis["fig4"]["panel_b_input_tokens"] == 512
    assert analysis["fig4"]["panel_b_input_tokens_source"] == (
        "calibration-assumption"
    )
    assert analysis["fig5"]["receiver_compute_tflops"] == 1
    assert analysis["fig5"]["generated_tokens"] == 512


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("paper_mode_analysis", "bandwidth_ghz"), 0, "bandwidth_ghz"),
        (
            ("paper_mode_analysis", "fig4", "panel_a_beta_points"),
            0,
            "points",
        ),
        (("paper_mode_analysis", "fig5", "rho_min"), 0, "rho_min"),
        (("paper_mode_analysis", "fig5", "rho_max"), 1.1, "rho_max"),
        (("paper_mode_analysis", "fig5", "xi_values"), [], "xi_values"),
    ],
)
def test_paper_mode_config_rejects_invalid_values(path, value, message):
    config = load_config("configs/paper_mode_analysis.yaml")
    broken = copy.deepcopy(config)
    target = broken
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        validate_paper_mode_analysis_config(broken)
