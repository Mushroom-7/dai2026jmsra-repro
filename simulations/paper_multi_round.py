import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.multi_round import (
    DialogueState,
    ExperimentResult,
    RoundResult,
    build_experiment,
    run_experiment,
)
from src.utils.data_gen import load_config, validate_paper_multi_round_config


def _state_dict(state: DialogueState) -> dict:
    return {
        "ea_context_tokens": state.ea_context_tokens,
        "ea_kv_debt_tokens": state.ea_kv_debt_tokens,
        "aa_context_tokens": list(state.aa_context_tokens),
        "aa_kv_debt_tokens": list(state.aa_kv_debt_tokens),
    }


def _round_dict(item: RoundResult) -> dict:
    return {
        "round": item.round_index,
        "active_agents": list(item.active_agents),
        "ea_mode": item.ea_mode,
        "aa_modes": list(item.aa_modes),
        "aa_bandwidth": list(item.aa_bandwidth),
        "broadcast_rate_bps": item.broadcast_rate_bps,
        "broadcast_nl_latency_seconds": (
            item.broadcast_nl_latency_seconds
        ),
        "broadcast_kv_latency_seconds": (
            item.broadcast_kv_latency_seconds
        ),
        "jmsra_latency_seconds": item.jmsra_latency_seconds,
        "communication_latency_seconds": (
            item.communication_latency_seconds
        ),
        "ours_ea_prefill_seconds": item.ours_ea_prefill_seconds,
        "ours_ea_autoregressive_seconds": (
            item.ours_ea_autoregressive_seconds
        ),
        "ours_ea_total_seconds": item.ours_ea_total_seconds,
        "pure_nl_ea_prefill_seconds": item.pure_nl_ea_prefill_seconds,
        "pure_nl_ea_autoregressive_seconds": (
            item.pure_nl_ea_autoregressive_seconds
        ),
        "pure_nl_ea_total_seconds": item.pure_nl_ea_total_seconds,
        "ea_autoregressive_context_tokens": (
            item.ea_autoregressive_context_tokens
        ),
        "aa_debt_before_return": list(item.aa_debt_before_return),
        "channels": {
            "distances_m": list(item.channels.distances_m),
            "fading_powers": list(item.channels.fading_powers),
            "gains": list(item.channels.gains),
        },
        "state_before": _state_dict(item.state_before),
        "state_after": _state_dict(item.state_after),
    }


def _mode_share(rounds: tuple[RoundResult, ...], mode: int) -> float:
    total = sum(len(item.aa_modes) for item in rounds)
    if total == 0:
        raise ValueError("trend interval must contain active AAs")
    selected = sum(
        sum(value == mode for value in item.aa_modes) for item in rounds
    )
    return selected / total


def _ratio(numerator: float, denominator: float, name: str) -> float:
    if denominator <= 0 or not math.isfinite(denominator):
        raise ValueError(f"{name} denominator must be finite and positive")
    return numerator / denominator


def _evaluate_trends(
    result: ExperimentResult, paper: dict
) -> tuple[dict[str, bool], dict]:
    rounds = result.rounds
    trend = paper["trend"]
    early = rounds[: int(trend["early_round_end"])]
    deep = rounds[
        int(trend["deep_round_start"]) - 1 : int(trend["deep_round_end"])
    ]
    early_nl_share = _mode_share(early, 0)
    deep_nl_share = _mode_share(deep, 0)
    deep_kv_share = _mode_share(deep, 1)
    contexts = [item.state_after.ea_context_tokens for item in rounds]
    context_limit = float(paper["ea_context_limit_tokens"])
    pure_nl_peak = max(
        rounds, key=lambda item: item.pure_nl_ea_total_seconds
    )
    deep_prefill_ratio = _ratio(
        sum(item.ours_ea_prefill_seconds for item in deep),
        sum(item.pure_nl_ea_prefill_seconds for item in deep),
        "deep prefill ratio",
    )
    deep_total_ratio = _ratio(
        sum(item.ours_ea_total_seconds for item in deep),
        sum(item.pure_nl_ea_total_seconds for item in deep),
        "deep total ratio",
    )
    later_ea_kv_share = sum(item.ea_mode for item in rounds[1:]) / (
        len(rounds) - 1
    )
    checks = {
        "early_nl_share_exceeds_late": early_nl_share > deep_nl_share,
        "deep_rounds_majority_kv": deep_kv_share
        >= float(trend["deep_kv_share_min"]),
        "ea_first_round_kv": rounds[0].ea_mode == 1,
        "ea_later_predominantly_nl": later_ea_kv_share <= 0.25,
        "ea_context_saturates": (
            all(left <= right for left, right in zip(contexts, contexts[1:]))
            and contexts[-1] == context_limit
        ),
        "pure_nl_peak_matches_scale": (
            float(trend["pure_nl_peak_min_seconds"])
            <= pure_nl_peak.pure_nl_ea_total_seconds
            <= float(trend["pure_nl_peak_max_seconds"])
        ),
        "proposed_suppresses_deep_prefill": deep_prefill_ratio
        <= float(trend["proposed_prefill_ratio_max"]),
        "proposed_suppresses_deep_total": deep_total_ratio
        <= float(trend["proposed_total_ratio_max"]),
    }
    details = {
        "early_nl_share": early_nl_share,
        "deep_nl_share": deep_nl_share,
        "deep_kv_share": deep_kv_share,
        "later_ea_kv_share": later_ea_kv_share,
        "context_limit_tokens": context_limit,
        "context_saturation_round": next(
            item.round_index
            for item in rounds
            if item.state_after.ea_context_tokens == context_limit
        ),
        "pure_nl_peak_round": pure_nl_peak.round_index,
        "pure_nl_peak_seconds": pure_nl_peak.pure_nl_ea_total_seconds,
        "deep_prefill_ratio": deep_prefill_ratio,
        "deep_total_ratio": deep_total_ratio,
    }
    return checks, details


def _schedule(result: ExperimentResult, max_aas: int) -> list[list[int]]:
    rows = [[item.ea_mode for item in result.rounds]]
    for agent_index in range(max_aas):
        row = []
        for item in result.rounds:
            if agent_index not in item.active_agents:
                row.append(-1)
            else:
                local_index = item.active_agents.index(agent_index)
                row.append(item.aa_modes[local_index])
        rows.append(row)
    return rows


def run_paper_multi_round(
    config_path: str = "configs/paper_multi_round.yaml",
    output_path: str = "results/logs/paper_multi_round.json",
) -> dict:
    config = load_config(config_path)
    validate_paper_multi_round_config(config)
    paper = config["paper_multi_round"]
    experiment = build_experiment(config)
    result = run_experiment(experiment)
    checks, details = _evaluate_trends(result, paper)
    published_parameter_names = {
        "rounds",
        "max_aas",
        "bits_per_token",
        "kv_cache_compression_ratio",
        "bandwidth_ghz",
        "noise_dbm_per_hz",
        "distance_min_m",
        "distance_max_m",
        "path_loss_model",
        "ea_compute_tflops",
        "aa_compute_tflops_min",
        "aa_compute_tflops_max",
        "ea_transmit_power_dbm",
        "aa_transmit_power_dbm_min",
        "aa_transmit_power_dbm_max",
        "ea_initial_prompt_tokens",
        "ea_generated_tokens",
        "aa_sensory_input_tokens",
        "aa_generated_tokens",
        "ea_context_limit_tokens",
        "ea_kv_debt_window_tokens",
        "aa_context_limit_tokens",
        "aa_kv_debt_window_tokens",
    }
    published_parameters = {
        key: paper[key] for key in sorted(published_parameter_names)
    }
    data = {
        "schema_version": int(paper["schema_version"]),
        "metadata": {
            "llm_profile": "LLaMA-7B analytical parameters",
            "executes_model_weights": False,
            "random_seed": int(config["random_seed"]),
            "active_aa_counts_source": paper["active_aa_counts_source"],
            "published_parameters": {
                "llm": config["llm"],
                "table4": published_parameters,
            },
            "calibration_assumptions": {
                "random_seed": int(config["random_seed"]),
                "active_aa_counts": paper["active_aa_counts"],
            },
            "numerical_settings": {
                "schema_version": int(paper["schema_version"]),
                "tolerance_seconds": float(paper["tolerance_seconds"]),
                "min_rho": float(paper["min_rho"]),
                "trend_thresholds": paper["trend"],
            },
        },
        "agents": [
            {
                "agent_index": item.agent_index,
                "compute_tflops": item.compute_tflops,
                "compute_flops": item.compute_flops,
                "transmit_power_dbm": item.transmit_power_dbm,
                "transmit_power_watt": item.transmit_power_watt,
            }
            for item in result.agent_profiles
        ],
        "rounds": [_round_dict(item) for item in result.rounds],
        "schedule": _schedule(result, int(paper["max_aas"])),
        "trend_checks": checks,
        "trend_check_details": details,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "paper multi-round trend checks failed: " + ", ".join(failed)
        )
    return data


if __name__ == "__main__":
    run_paper_multi_round()
