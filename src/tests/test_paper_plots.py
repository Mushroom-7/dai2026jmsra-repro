import json
import math
from pathlib import Path

import pytest
import yaml

from simulations.paper_jmsra_calibration import run_paper_jmsra_calibration
from simulations.plot_paper_jmsra import plot_paper_jmsra
from src.utils.data_gen import load_config


def _valid_case() -> dict:
    return {
        "aa_count": 1,
        "bandwidth_ghz": 2.0,
        "jmsra": {
            "kv_count": 1,
            "modes": [1],
            "rho": [1.0],
            "latency_seconds": 1.0,
            "forward_path": [{"kv_count": 0, "latency_seconds": 2.0}],
            "backward_path": [{"kv_count": 1, "latency_seconds": 1.0}],
        },
        "topology": [
            {
                "agent_index": 0,
                "x_m": 5.0,
                "y_m": 0.0,
                "distance_m": 5.0,
                "mode": 1,
                "media": "KV",
                "rho": 1.0,
            }
        ],
        "baselines": [
            {"name": "All-NL", "latency_seconds": 2.0},
            {"name": "All-KV", "latency_seconds": 3.0},
            {"name": "All-NL-Opt", "latency_seconds": 2.0},
            {"name": "All-KV-Opt", "latency_seconds": 2.5},
        ],
    }


def _valid_plot_data() -> dict:
    return {
        "metadata": {"schema_version": 2},
        "fig6": {
            "c0_5_tflops": _valid_case(),
            "c0_35_tflops": _valid_case(),
        },
        "fig7": {
            "bandwidth_scan": [_valid_case()],
            "aa_count_scan": [_valid_case()],
        },
    }


def test_plot_paper_jmsra_generates_expected_pngs(tmp_path):
    config = load_config("configs/paper_jmsra.yaml")
    config["paper_jmsra"]["fig6"]["aa_count"] = 3
    config["paper_jmsra"]["fig6"]["low_anchor_kv_count"] = 2
    config["paper_jmsra"]["fig6"]["high_anchor_kv_count"] = 1
    config["paper_jmsra"]["fig7"]["bandwidth_scan_aa_count"] = 3
    config["paper_jmsra"]["fig7"]["aa_count_values"] = list(range(1, 11))
    config_path = tmp_path / "paper_jmsra.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    log_path = tmp_path / "paper_jmsra.json"
    figure_dir = tmp_path / "figures"
    run_paper_jmsra_calibration(str(config_path), str(log_path))
    outputs = plot_paper_jmsra(str(log_path), str(figure_dir))
    names = {Path(path).name for path in outputs}
    assert names == {
        "paper_fig6_latency_paths.png",
        "paper_fig6_topology_c0_5_tflops.png",
        "paper_fig6_topology_c0_35_tflops.png",
        "paper_fig7_bandwidth_scan.png",
        "paper_fig7_aa_count_scan.png",
    }
    for path in outputs:
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0


def test_plot_paper_jmsra_rejects_missing_required_sections(tmp_path):
    input_path = tmp_path / "malformed.json"
    input_path.write_text(
        json.dumps({"metadata": {"schema_version": 2}, "fig7": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fig6"):
        plot_paper_jmsra(str(input_path), str(tmp_path / "figures"))


def test_plot_paper_jmsra_rejects_unsupported_schema(tmp_path):
    data = _valid_plot_data()
    data["metadata"]["schema_version"] = 1
    input_path = tmp_path / "old-schema.json"
    input_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version must be 2"):
        plot_paper_jmsra(str(input_path), str(tmp_path / "figures"))


def test_plot_paper_jmsra_rejects_invalid_topology_mode(tmp_path):
    data = _valid_plot_data()
    data["fig6"]["c0_5_tflops"]["topology"][0]["mode"] = 2
    input_path = tmp_path / "invalid-mode.json"
    input_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match=r"topology\[0\]\.mode"):
        plot_paper_jmsra(str(input_path), str(tmp_path / "figures"))


def test_plot_paper_jmsra_rejects_topology_count_mismatch(tmp_path):
    data = _valid_plot_data()
    data["fig6"]["c0_5_tflops"]["aa_count"] = 2
    input_path = tmp_path / "count-mismatch.json"
    input_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="topology length"):
        plot_paper_jmsra(str(input_path), str(tmp_path / "figures"))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("x_m", math.inf, r"topology\[0\]\.x_m"),
        ("x_m", 6.0, "coordinate radius"),
        ("distance_m", -1.0, r"topology\[0\]\.distance_m"),
        ("media", "NL", r"topology\[0\]\.media"),
        ("rho", 0.0, r"topology\[0\]\.rho"),
        ("rho", True, r"topology\[0\]\.rho"),
        ("agent_index", 1, r"topology\[0\]\.agent_index"),
    ],
)
def test_plot_paper_jmsra_rejects_semantically_invalid_topology(
    tmp_path, field, value, message
):
    data = _valid_plot_data()
    data["fig6"]["c0_5_tflops"]["topology"][0][field] = value
    input_path = tmp_path / f"invalid-{field}.json"
    input_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        plot_paper_jmsra(str(input_path), str(tmp_path / "figures"))
