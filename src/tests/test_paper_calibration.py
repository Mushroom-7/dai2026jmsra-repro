import json

import pytest
import yaml

from simulations.paper_jmsra_calibration import run_paper_jmsra_calibration
from src.algorithms.baselines.all_kv_opt import run_all_kv_opt
from src.algorithms.jmsra import solve_jmsra
from src.utils.data_gen import (
    load_config,
    make_paper_jmsra_problem,
    make_paper_jmsra_topology,
)


@pytest.fixture(scope="module")
def calibration_result(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("paper-calibration")
    config = load_config("configs/paper_jmsra.yaml")
    config["paper_jmsra"]["fig6"]["aa_count"] = 3
    config["paper_jmsra"]["fig6"]["low_anchor_kv_count"] = 2
    config["paper_jmsra"]["fig6"]["high_anchor_kv_count"] = 1
    config["paper_jmsra"]["fig7"]["bandwidth_scan_aa_count"] = 3
    config["paper_jmsra"]["fig7"]["aa_count_values"] = list(range(1, 11))
    config_path = temp_dir / "paper_jmsra.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = temp_dir / "paper_jmsra.json"
    result = run_paper_jmsra_calibration(str(config_path), str(output))
    return result, output, config_path


def test_paper_jmsra_calibration_writes_fig6_and_fig7_sections(
    calibration_result,
):
    result, output, _ = calibration_result
    assert output.exists()
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert result == saved
    assert set(saved) >= {"fig6", "fig7"}
    assert set(saved["fig6"]) == {"c0_5_tflops", "c0_35_tflops"}
    assert len(saved["fig7"]["bandwidth_scan"]) == 8
    assert len(saved["fig7"]["aa_count_scan"]) == 10


def test_paper_jmsra_calibration_uses_topology_schema_v2(calibration_result):
    result, _, _ = calibration_result
    assert result["metadata"]["schema_version"] == 2


def test_paper_jmsra_calibration_includes_baselines_and_anchor_deltas(
    calibration_result,
):
    result, _, _ = calibration_result
    low = result["fig6"]["c0_5_tflops"]
    high = result["fig6"]["c0_35_tflops"]
    for section in (low, high):
        assert "jmsra" in section
        assert "baselines" in section
        assert "paper_anchor_delta" in section
        assert section["jmsra"]["latency_seconds"] > 0
        assert section["jmsra"]["kv_count"] == sum(
            section["jmsra"]["modes"]
        )
        assert section["jmsra"]["latency_seconds"] <= min(
            baseline["latency_seconds"]
            for baseline in section["baselines"]
        ) + 1e-6


def test_fig6_cases_include_mode_topology(calibration_result):
    result, _, _ = calibration_result
    for section in result["fig6"].values():
        topology = section["topology"]
        assert len(topology) == section["aa_count"]
        assert sum(node["mode"] for node in topology) == section["jmsra"][
            "kv_count"
        ]
        for index, node in enumerate(topology):
            assert node["agent_index"] == index
            assert node["mode"] in {0, 1}
            assert node["media"] in {"NL", "KV"}
            assert node["rho"] == section["jmsra"]["rho"][index]
            assert 5 <= node["distance_m"] <= 10


def test_paper_jmsra_calibration_uses_nested_seed_policy(calibration_result):
    result, _, _ = calibration_result
    assert result["metadata"]["aa_count_seed_policy"] == "nested-prefix"
    assert {
        item["seed"] for item in result["fig7"]["aa_count_scan"]
    } == {result["metadata"]["base_seed"]}
    latencies = [
        item["jmsra"]["latency_seconds"]
        for item in result["fig7"]["aa_count_scan"]
    ]
    assert latencies == sorted(latencies)


def test_paper_jmsra_calibration_is_byte_reproducible(calibration_result):
    result, output, config_path = calibration_result
    repeated = output.with_name("paper_jmsra_repeated.json")
    again = run_paper_jmsra_calibration(str(config_path), str(repeated))
    assert again == result
    assert repeated.read_bytes() == output.read_bytes()


def test_paper_anchor_deltas_are_exact(calibration_result):
    result, _, _ = calibration_result
    fig6_config = result["metadata"]["config"]["paper_jmsra"]["fig6"]
    anchors = {
        "c0_5_tflops": (
            fig6_config["low_anchor_latency_seconds"],
            fig6_config["low_anchor_kv_count"],
        ),
        "c0_35_tflops": (
            fig6_config["high_anchor_latency_seconds"],
            fig6_config["high_anchor_kv_count"],
        ),
    }
    for name, (latency, kv_count) in anchors.items():
        section = result["fig6"][name]
        assert section["paper_anchor_delta"]["latency_seconds"] == (
            section["jmsra"]["latency_seconds"] - latency
        )
        assert section["paper_anchor_delta"]["kv_count"] == (
            section["jmsra"]["kv_count"] - kv_count
        )


def test_production_seed_matches_fig6_anchor_neighborhoods():
    config = load_config("configs/paper_jmsra.yaml")
    paper = config["paper_jmsra"]
    fig6 = paper["fig6"]
    seed = config["random_seed"]
    assert seed == 1477
    low = solve_jmsra(
        make_paper_jmsra_problem(config, 20, 5, 2, seed)
    )
    high = solve_jmsra(
        make_paper_jmsra_problem(config, 20, 35, 2, seed)
    )
    assert abs(
        low.latency_seconds - fig6["low_anchor_latency_seconds"]
    ) <= 0.2 * fig6["low_anchor_latency_seconds"]
    assert abs(sum(low.modes) - fig6["low_anchor_kv_count"]) <= 1
    assert abs(
        high.latency_seconds - fig6["high_anchor_latency_seconds"]
    ) <= 0.1 * fig6["high_anchor_latency_seconds"]
    assert abs(sum(high.modes) - fig6["high_anchor_kv_count"]) <= 1
    topology = make_paper_jmsra_topology(config, 20, seed)
    high_kv_distances = [
        node["distance_m"]
        for node, mode in zip(topology, high.modes)
        if mode == 1
    ]
    high_nl_distances = [
        node["distance_m"]
        for node, mode in zip(topology, high.modes)
        if mode == 0
    ]
    assert sum(low.modes) > sum(high.modes)
    assert sum(high_kv_distances) / len(high_kv_distances) < (
        sum(high_nl_distances) / len(high_nl_distances)
    )


def test_production_seed_reproduces_high_scale_mode_conclusion():
    config = load_config("configs/paper_jmsra.yaml")
    seed = config["random_seed"]
    problem = make_paper_jmsra_problem(config, 50, 10, 2, seed)
    jmsra = solve_jmsra(problem)
    all_kv_opt = run_all_kv_opt(problem)
    assert sum(jmsra.modes) >= 40
    assert jmsra.latency_seconds <= all_kv_opt.latency_seconds
    assert all_kv_opt.latency_seconds - jmsra.latency_seconds <= (
        0.2 * all_kv_opt.latency_seconds
    )
