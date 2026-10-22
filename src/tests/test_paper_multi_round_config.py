import copy

import pytest

from src.utils.data_gen import load_config, validate_paper_multi_round_config


def test_table4_values_and_assumptions_are_explicit():
    config = load_config("configs/paper_multi_round.yaml")
    validate_paper_multi_round_config(config)
    paper = config["paper_multi_round"]
    assert config["llm"] == {
        "layers": 32,
        "heads": 32,
        "head_dim": 128,
        "hidden_dim": 4096,
        "ffn_dim": 11008,
        "vocab_size": 32000,
    }
    assert paper["schema_version"] == 1
    assert paper["rounds"] == 40
    assert paper["max_aas"] == 20
    assert paper["ea_compute_tflops"] == 20
    assert paper["ea_context_limit_tokens"] == 409600
    assert paper["ea_kv_debt_window_tokens"] == 102400
    assert paper["aa_context_limit_tokens"] == 51200
    assert paper["aa_kv_debt_window_tokens"] == 10240
    assert len(paper["active_aa_counts"]) == 40
    assert paper["active_aa_counts_source"] == (
        "figure-inferred-calibration"
    )


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("rounds", 0, "rounds"),
        ("active_aa_counts", [20], "active_aa_counts"),
        ("ea_context_limit_tokens", 0, "ea_context_limit_tokens"),
        ("aa_kv_debt_window_tokens", 60000, "debt window"),
        ("path_loss_model", "free-space", "path_loss_model"),
    ],
)
def test_invalid_multi_round_config_is_rejected(key, value, message):
    config = load_config("configs/paper_multi_round.yaml")
    broken = copy.deepcopy(config)
    broken["paper_multi_round"][key] = value
    with pytest.raises(ValueError, match=message):
        validate_paper_multi_round_config(broken)


def test_active_counts_must_stay_within_agent_limit():
    config = load_config("configs/paper_multi_round.yaml")
    config["paper_multi_round"]["active_aa_counts"][0] = 21
    with pytest.raises(ValueError, match="active_aa_counts"):
        validate_paper_multi_round_config(config)
