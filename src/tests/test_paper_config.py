import math

import pytest

from src.algorithms.jmsra import JmsraProblem
from src.utils.data_gen import (
    load_config,
    make_paper_jmsra_problem,
    make_paper_jmsra_topology,
    validate_paper_jmsra_config,
)


def test_paper_jmsra_config_matches_table_3_values():
    config = load_config("configs/paper_jmsra.yaml")
    validate_paper_jmsra_config(config)
    paper = config["paper_jmsra"]
    assert paper["bits_per_token"] == 16
    assert paper["kv_cache_compression_ratio"] == 2
    assert paper["bandwidth_ghz"] == 2
    assert paper["noise_dbm_per_hz"] == -140
    assert paper["distance_min_m"] == 5
    assert paper["distance_max_m"] == 10
    assert paper["theta0_tokens"] == 1024 * 5
    assert paper["fig6"]["aa_count"] == 20
    assert paper["fig7"]["ea_compute_tflops"] == 10


def test_make_paper_jmsra_problem_is_deterministic_and_positive():
    config = load_config("configs/paper_jmsra.yaml")
    problem = make_paper_jmsra_problem(
        config,
        aa_count=20,
        ea_compute_tflops=5,
        bandwidth_ghz=2,
        seed=2026,
    )
    again = make_paper_jmsra_problem(
        config,
        aa_count=20,
        ea_compute_tflops=5,
        bandwidth_ghz=2,
        seed=2026,
    )
    assert isinstance(problem, JmsraProblem)
    assert problem.size == 20
    assert problem.alphas == again.alphas
    assert problem.xis == again.xis
    assert problem.gains == again.gains
    assert all(819.2 <= value <= 1228.8 for value in problem.alphas)
    assert all(value > 0 for value in problem.gains)
    assert all(value > 0 for value in problem.powers_watt)
    assert math.isclose(problem.theta0, 5120)


def test_paper_problem_population_is_nested_by_seed():
    config = load_config("configs/paper_jmsra.yaml")
    small = make_paper_jmsra_problem(config, 5, 10, 2, 2026)
    large = make_paper_jmsra_problem(config, 10, 10, 2, 2026)
    assert small.alphas == large.alphas[:5]
    assert small.xis == large.xis[:5]
    assert small.gains == large.gains[:5]
    assert small.powers_watt == large.powers_watt[:5]


def test_paper_topology_is_deterministic_and_matches_problem_vectors():
    config = load_config("configs/paper_jmsra.yaml")
    problem = make_paper_jmsra_problem(config, 20, 5, 2, 2026)
    topology = make_paper_jmsra_topology(config, 20, 2026)
    again = make_paper_jmsra_topology(config, 20, 2026)
    assert topology == again
    assert len(topology) == problem.size
    for index, node in enumerate(topology):
        assert node["agent_index"] == index
        assert 5 <= node["distance_m"] <= 10
        assert math.isclose(
            node["x_m"] ** 2 + node["y_m"] ** 2,
            node["distance_m"] ** 2,
            rel_tol=1e-12,
        )
        assert node["gain"] == problem.gains[index]
        assert node["power_watt"] == problem.powers_watt[index]


def test_paper_config_rejects_missing_nested_scan_key():
    config = load_config("configs/paper_jmsra.yaml")
    del config["paper_jmsra"]["fig7"]["aa_count_values"]
    with pytest.raises(ValueError, match="fig7"):
        validate_paper_jmsra_config(config)


def test_paper_config_rejects_empty_scan():
    config = load_config("configs/paper_jmsra.yaml")
    config["paper_jmsra"]["fig7"]["bandwidth_ghz_values"] = []
    with pytest.raises(ValueError, match="bandwidth_ghz_values"):
        validate_paper_jmsra_config(config)


def test_paper_config_rejects_unsupported_path_loss_model():
    config = load_config("configs/paper_jmsra.yaml")
    config["paper_jmsra"]["path_loss_model"] = "free-space"
    with pytest.raises(ValueError, match="path_loss_model"):
        validate_paper_jmsra_config(config)
