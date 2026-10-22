import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.baselines.all_kv import run_all_kv
from src.algorithms.baselines.all_kv_opt import run_all_kv_opt
from src.algorithms.baselines.all_nl import run_all_nl
from src.algorithms.jmsra import solve_jmsra_with_trace
from src.utils.data_gen import (
    load_config,
    make_paper_jmsra_problem,
    make_paper_jmsra_topology,
    validate_paper_jmsra_config,
)


def _baseline_to_dict(result):
    return {
        "name": result.name,
        "modes": result.modes,
        "rho": result.rho,
        "latency_seconds": result.latency_seconds,
        "kv_count": sum(result.modes),
    }


def _trace_point_to_dict(point):
    return {
        "kv_count": point.kv_count,
        "latency_seconds": point.latency_seconds,
        "modes": point.modes,
    }


def _jmsra_to_dict(traced):
    result = traced.result
    return {
        "modes": result.modes,
        "rho": result.rho,
        "latency_seconds": result.latency_seconds,
        "transmission_latency_seconds": (
            result.transmission_latency_seconds
        ),
        "kv_count": sum(result.modes),
        "selected_direction": traced.selected_direction,
        "forward_path": [
            _trace_point_to_dict(point) for point in traced.forward_path
        ],
        "backward_path": [
            _trace_point_to_dict(point) for point in traced.backward_path
        ],
    }


def _topology_to_dict(
    config: dict, aa_count: int, seed: int, traced
) -> list[dict]:
    modes = traced.result.modes
    rho = traced.result.rho
    topology = make_paper_jmsra_topology(config, aa_count, seed)
    return [
        {
            **node,
            "mode": modes[index],
            "media": "KV" if modes[index] == 1 else "NL",
            "rho": rho[index],
        }
        for index, node in enumerate(topology)
    ]


def _run_case(
    config: dict,
    aa_count: int,
    ea_compute_tflops: float,
    bandwidth_ghz: float,
    seed: int,
) -> dict:
    problem = make_paper_jmsra_problem(
        config,
        aa_count=aa_count,
        ea_compute_tflops=ea_compute_tflops,
        bandwidth_ghz=bandwidth_ghz,
        seed=seed,
    )
    traced = solve_jmsra_with_trace(problem)
    baselines = [
        run_all_nl(problem),
        run_all_nl(problem, optimized=True),
        run_all_kv(problem),
        run_all_kv_opt(problem),
    ]
    return {
        "aa_count": aa_count,
        "ea_compute_tflops": ea_compute_tflops,
        "bandwidth_ghz": bandwidth_ghz,
        "seed": seed,
        "jmsra": _jmsra_to_dict(traced),
        "topology": _topology_to_dict(config, aa_count, seed, traced),
        "baselines": [_baseline_to_dict(item) for item in baselines],
    }


def _with_anchor_delta(
    section: dict, anchor_latency: float, anchor_kv_count: int
) -> dict:
    jmsra = section["jmsra"]
    section["paper_anchor_delta"] = {
        "latency_seconds": jmsra["latency_seconds"] - anchor_latency,
        "kv_count": jmsra["kv_count"] - anchor_kv_count,
    }
    return section


def run_paper_jmsra_calibration(
    config_path: str = "configs/paper_jmsra.yaml",
    output_path: str = "results/logs/paper_jmsra_calibration.json",
) -> dict:
    config = load_config(config_path)
    validate_paper_jmsra_config(config)
    paper = config["paper_jmsra"]
    seed = int(config["random_seed"])
    fig6 = paper["fig6"]
    low = _run_case(
        config,
        int(fig6["aa_count"]),
        float(fig6["low_ea_compute_tflops"]),
        float(paper["bandwidth_ghz"]),
        seed,
    )
    high = _run_case(
        config,
        int(fig6["aa_count"]),
        float(fig6["high_ea_compute_tflops"]),
        float(paper["bandwidth_ghz"]),
        seed,
    )
    fig7 = paper["fig7"]
    bandwidth_scan = [
        _run_case(
            config,
            int(fig7["bandwidth_scan_aa_count"]),
            float(fig7["ea_compute_tflops"]),
            float(bandwidth),
            seed,
        )
        for bandwidth in fig7["bandwidth_ghz_values"]
    ]
    aa_count_scan = [
        _run_case(
            config,
            int(aa_count),
            float(fig7["ea_compute_tflops"]),
            float(fig7["aa_count_scan_bandwidth_ghz"]),
            seed,
        )
        for aa_count in fig7["aa_count_values"]
    ]
    result = {
        "metadata": {
            "schema_version": 2,
            "base_seed": seed,
            "aa_count_seed_policy": "nested-prefix",
            "config": config,
        },
        "fig6": {
            "c0_5_tflops": _with_anchor_delta(
                low,
                float(fig6["low_anchor_latency_seconds"]),
                int(fig6["low_anchor_kv_count"]),
            ),
            "c0_35_tflops": _with_anchor_delta(
                high,
                float(fig6["high_anchor_latency_seconds"]),
                int(fig6["high_anchor_kv_count"]),
            ),
        },
        "fig7": {
            "bandwidth_scan": bandwidth_scan,
            "aa_count_scan": aa_count_scan,
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    run_paper_jmsra_calibration()
