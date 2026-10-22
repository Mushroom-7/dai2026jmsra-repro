import math

import numpy as np
import yaml

from src.algorithms.jmsra import JmsraProblem
from src.channels.wireless_channel import channel_gain
from src.models.transformer import TransformerConfig
from src.utils.units import (
    dbm_to_watt,
    ghz_to_hz,
    noise_dbm_per_hz_to_watt_per_hz,
    tflops_to_flops,
)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def transformer_config(config: dict) -> TransformerConfig:
    llm = config["llm"]
    return TransformerConfig(
        layers=llm["layers"],
        heads=llm["heads"],
        head_dim=llm["head_dim"],
        hidden_dim=llm["hidden_dim"],
        ffn_dim=llm["ffn_dim"],
        vocab_size=llm["vocab_size"],
    )


def make_single_round_problem(config: dict) -> JmsraProblem:
    rng = np.random.default_rng(config["random_seed"])
    single_round = config["single_round"]
    communication = config["communication"]
    count = single_round["num_aas"]
    alphas = single_round["alpha_base"] * rng.uniform(
        single_round["alpha_jitter_min"],
        single_round["alpha_jitter_max"],
        size=count,
    )
    xis = rng.uniform(
        single_round["xi_multiplier_min"],
        single_round["xi_multiplier_max"],
        size=count,
    ) * alphas
    distances = rng.uniform(
        single_round["distance_min_m"],
        single_round["distance_max_m"],
        size=count,
    )
    fading = rng.exponential(scale=1.0, size=count)
    gains = [
        channel_gain(float(distance), float(power))
        for distance, power in zip(distances, fading)
    ]
    powers = rng.uniform(
        single_round["aa_transmit_power_dbm_min"],
        single_round["aa_transmit_power_dbm_max"],
        size=count,
    )
    return JmsraProblem(
        config=transformer_config(config),
        alphas=[float(value) for value in alphas],
        xis=[float(value) for value in xis],
        gammas=[float(communication["kv_cache_compression_ratio"])]
        * count,
        powers_watt=[dbm_to_watt(float(value)) for value in powers],
        gains=gains,
        bandwidth_hz=ghz_to_hz(
            float(communication["bandwidth_ghz"])
        ),
        noise_psd_w_per_hz=noise_dbm_per_hz_to_watt_per_hz(
            float(communication["noise_dbm_per_hz"])
        ),
        bits_per_token=float(communication["bits_per_token"]),
        ea_compute_flops=tflops_to_flops(
            float(single_round["ea_compute_tflops"])
        ),
        theta0=float(single_round["theta0_tokens"]),
        tolerance_seconds=float(config["jmsra"]["tolerance_seconds"]),
        min_rho=float(config["jmsra"]["min_rho"]),
    )


def _require_keys(section: dict, name: str, keys: set[str]) -> None:
    if not isinstance(section, dict):
        raise ValueError(f"{name} must be a mapping")
    missing = keys - set(section)
    if missing:
        raise ValueError(f"missing {name} keys: {sorted(missing)}")


def _finite_number(value, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive_number(value, name: str) -> float:
    number = _finite_number(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _nonnegative_number(value, name: str) -> float:
    number = _finite_number(value, name)
    if number < 0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_increasing_values(
    values, name: str, *, integers: bool = False
) -> None:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a nonempty list")
    if integers:
        parsed = [_positive_int(value, name) for value in values]
    else:
        parsed = [_positive_number(value, name) for value in values]
    if any(left >= right for left, right in zip(parsed, parsed[1:])):
        raise ValueError(f"{name} must be strictly increasing")


def _finite_increasing_values(
    values, name: str, *, nonnegative: bool = False
) -> None:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a nonempty list")
    parser = _nonnegative_number if nonnegative else _finite_number
    parsed = [parser(value, name) for value in values]
    if any(left >= right for left, right in zip(parsed, parsed[1:])):
        raise ValueError(f"{name} must be strictly increasing")


def validate_paper_mode_analysis_config(config: dict) -> None:
    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    _require_keys(config, "config", {"llm", "paper_mode_analysis"})
    llm = config["llm"]
    llm_keys = {
        "layers",
        "heads",
        "head_dim",
        "hidden_dim",
        "ffn_dim",
        "vocab_size",
    }
    _require_keys(llm, "llm", llm_keys)
    for key in llm_keys:
        _positive_int(llm[key], f"llm.{key}")

    analysis = config["paper_mode_analysis"]
    _require_keys(
        analysis,
        "paper_mode_analysis",
        {
            "schema_version",
            "bits_per_token",
            "kv_cache_compression_ratio",
            "bandwidth_ghz",
            "noise_dbm_per_hz",
            "final_ea_output_tokens",
            "rho_tolerance",
            "score_tolerance_seconds",
            "threshold_max_iterations",
            "fig4",
            "fig5",
        },
    )
    if analysis["schema_version"] != 1:
        raise ValueError("paper_mode_analysis.schema_version must be 1")
    for key in (
        "bits_per_token",
        "kv_cache_compression_ratio",
        "bandwidth_ghz",
        "final_ea_output_tokens",
        "rho_tolerance",
        "score_tolerance_seconds",
    ):
        _positive_number(analysis[key], f"paper_mode_analysis.{key}")
    _finite_number(
        analysis["noise_dbm_per_hz"],
        "paper_mode_analysis.noise_dbm_per_hz",
    )
    _positive_int(
        analysis["threshold_max_iterations"],
        "paper_mode_analysis.threshold_max_iterations",
    )
    if float(analysis["rho_tolerance"]) > 1:
        raise ValueError("paper_mode_analysis.rho_tolerance must be <= 1")

    fig4 = analysis["fig4"]
    fig4_keys = {
        "default_aa_count",
        "default_ea_compute_tflops",
        "default_aa_compute_tflops",
        "default_beta",
        "panel_a_beta_min",
        "panel_a_beta_max",
        "panel_a_beta_points",
        "panel_a_snr_db_values",
        "panel_a_input_tokens_values",
        "panel_b_compute_min_tflops",
        "panel_b_compute_max_tflops",
        "panel_b_compute_points",
        "panel_b_snr_db_values",
        "panel_b_input_tokens",
        "panel_b_input_tokens_source",
        "panel_c_snr_min_db",
        "panel_c_snr_max_db",
        "panel_c_snr_points",
        "panel_c_aa_count_values",
        "panel_c_input_tokens_values",
        "panel_d_aa_count_values",
        "panel_d_snr_db_values",
        "panel_d_input_tokens_values",
    }
    _require_keys(fig4, "paper_mode_analysis.fig4", fig4_keys)
    _positive_int(
        fig4["default_aa_count"],
        "paper_mode_analysis.fig4.default_aa_count",
    )
    for key in (
        "default_ea_compute_tflops",
        "default_aa_compute_tflops",
        "panel_a_beta_min",
        "panel_a_beta_max",
        "panel_b_compute_min_tflops",
        "panel_b_compute_max_tflops",
        "panel_b_input_tokens",
    ):
        _positive_number(fig4[key], f"paper_mode_analysis.fig4.{key}")
    _nonnegative_number(
        fig4["default_beta"], "paper_mode_analysis.fig4.default_beta"
    )
    for low_key, high_key in (
        ("panel_a_beta_min", "panel_a_beta_max"),
        ("panel_b_compute_min_tflops", "panel_b_compute_max_tflops"),
        ("panel_c_snr_min_db", "panel_c_snr_max_db"),
    ):
        low = _finite_number(fig4[low_key], f"paper_mode_analysis.fig4.{low_key}")
        high = _finite_number(
            fig4[high_key], f"paper_mode_analysis.fig4.{high_key}"
        )
        if high <= low:
            raise ValueError(f"paper_mode_analysis.fig4.{high_key} must exceed {low_key}")
    for key in (
        "panel_a_beta_points",
        "panel_b_compute_points",
        "panel_c_snr_points",
    ):
        _positive_int(fig4[key], f"paper_mode_analysis.fig4.{key}")
    for key in (
        "panel_a_snr_db_values",
        "panel_b_snr_db_values",
        "panel_d_snr_db_values",
    ):
        _finite_increasing_values(
            fig4[key], f"paper_mode_analysis.fig4.{key}"
        )
    for key in (
        "panel_a_input_tokens_values",
        "panel_c_input_tokens_values",
        "panel_d_input_tokens_values",
    ):
        _positive_increasing_values(
            fig4[key], f"paper_mode_analysis.fig4.{key}"
        )
    for key in ("panel_c_aa_count_values", "panel_d_aa_count_values"):
        _positive_increasing_values(
            fig4[key], f"paper_mode_analysis.fig4.{key}", integers=True
        )
    _finite_number(
        fig4["panel_c_snr_min_db"],
        "paper_mode_analysis.fig4.panel_c_snr_min_db",
    )
    _finite_number(
        fig4["panel_c_snr_max_db"],
        "paper_mode_analysis.fig4.panel_c_snr_max_db",
    )
    if fig4["panel_b_input_tokens_source"] != "calibration-assumption":
        raise ValueError(
            "paper_mode_analysis.fig4.panel_b_input_tokens_source must be "
            "'calibration-assumption'"
        )

    fig5 = analysis["fig5"]
    fig5_keys = {
        "receiver_compute_tflops",
        "generated_tokens",
        "rho_min",
        "rho_max",
        "rho_points",
        "panel_a_snr_db",
        "xi_values",
        "theta_values",
        "panel_b_xi",
        "panel_b_snr_db_values",
        "panel_c_snr_db",
        "panel_c_alpha_min",
        "panel_c_alpha_max",
        "panel_c_alpha_points",
        "panel_c_xi_min",
        "panel_c_xi_max",
        "panel_c_xi_points",
    }
    _require_keys(fig5, "paper_mode_analysis.fig5", fig5_keys)
    for key in (
        "receiver_compute_tflops",
        "generated_tokens",
        "rho_min",
        "rho_max",
        "panel_c_alpha_min",
        "panel_c_alpha_max",
    ):
        _positive_number(fig5[key], f"paper_mode_analysis.fig5.{key}")
    rho_min = float(fig5["rho_min"])
    rho_max = float(fig5["rho_max"])
    if not 0 < rho_min < rho_max <= 1:
        raise ValueError(
            "paper_mode_analysis.fig5 rho range must satisfy "
            "0 < rho_min < rho_max <= 1"
        )
    for key in ("rho_points", "panel_c_alpha_points", "panel_c_xi_points"):
        _positive_int(fig5[key], f"paper_mode_analysis.fig5.{key}")
    for key in ("panel_a_snr_db", "panel_c_snr_db"):
        _finite_number(fig5[key], f"paper_mode_analysis.fig5.{key}")
    _nonnegative_number(
        fig5["panel_b_xi"], "paper_mode_analysis.fig5.panel_b_xi"
    )
    for key in ("panel_c_xi_min", "panel_c_xi_max"):
        _nonnegative_number(fig5[key], f"paper_mode_analysis.fig5.{key}")
    if float(fig5["panel_c_alpha_max"]) <= float(fig5["panel_c_alpha_min"]):
        raise ValueError(
            "paper_mode_analysis.fig5.panel_c_alpha_max must exceed "
            "panel_c_alpha_min"
        )
    if float(fig5["panel_c_xi_max"]) <= float(fig5["panel_c_xi_min"]):
        raise ValueError(
            "paper_mode_analysis.fig5.panel_c_xi_max must exceed panel_c_xi_min"
        )
    _finite_increasing_values(
        fig5["xi_values"],
        "paper_mode_analysis.fig5.xi_values",
        nonnegative=True,
    )
    _finite_increasing_values(
        fig5["theta_values"],
        "paper_mode_analysis.fig5.theta_values",
        nonnegative=True,
    )
    _finite_increasing_values(
        fig5["panel_b_snr_db_values"],
        "paper_mode_analysis.fig5.panel_b_snr_db_values",
    )


def validate_paper_jmsra_config(config: dict) -> None:
    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    required_sections = {"random_seed", "llm", "paper_jmsra"}
    missing = required_sections - set(config)
    if missing:
        raise ValueError(f"missing config sections: {sorted(missing)}")
    if isinstance(config["random_seed"], bool) or not isinstance(
        config["random_seed"], int
    ):
        raise ValueError("random_seed must be an integer")
    llm = config["llm"]
    llm_keys = {
        "layers",
        "heads",
        "head_dim",
        "hidden_dim",
        "ffn_dim",
        "vocab_size",
    }
    _require_keys(llm, "llm", llm_keys)
    for key in llm_keys:
        _positive_int(llm[key], f"llm.{key}")

    paper = config["paper_jmsra"]
    required_keys = {
        "bits_per_token",
        "kv_cache_compression_ratio",
        "bandwidth_ghz",
        "noise_dbm_per_hz",
        "distance_min_m",
        "distance_max_m",
        "path_loss_model",
        "aa_transmit_power_dbm_min",
        "aa_transmit_power_dbm_max",
        "alpha_base",
        "alpha_jitter_min",
        "alpha_jitter_max",
        "xi_multiplier_min",
        "xi_multiplier_max",
        "theta0_tokens",
        "tolerance_seconds",
        "min_rho",
        "fig6",
        "fig7",
    }
    _require_keys(paper, "paper_jmsra", required_keys)
    for key in (
        "bits_per_token",
        "kv_cache_compression_ratio",
        "bandwidth_ghz",
        "alpha_base",
        "alpha_jitter_min",
        "alpha_jitter_max",
        "xi_multiplier_min",
        "xi_multiplier_max",
        "theta0_tokens",
        "tolerance_seconds",
        "min_rho",
    ):
        _positive_number(paper[key], f"paper_jmsra.{key}")
    _finite_number(
        paper["noise_dbm_per_hz"], "paper_jmsra.noise_dbm_per_hz"
    )
    distance_min = _positive_number(
        paper["distance_min_m"], "paper_jmsra.distance_min_m"
    )
    distance_max = _positive_number(
        paper["distance_max_m"], "paper_jmsra.distance_max_m"
    )
    if distance_max < distance_min:
        raise ValueError("distance_max_m must be >= distance_min_m")
    if paper["path_loss_model"] != "30 + 35 log10(d)":
        raise ValueError(
            "path_loss_model must be '30 + 35 log10(d)'"
        )
    power_min = _finite_number(
        paper["aa_transmit_power_dbm_min"],
        "paper_jmsra.aa_transmit_power_dbm_min",
    )
    power_max = _finite_number(
        paper["aa_transmit_power_dbm_max"],
        "paper_jmsra.aa_transmit_power_dbm_max",
    )
    if power_max < power_min:
        raise ValueError(
            "aa_transmit_power_dbm_max must be >= aa_transmit_power_dbm_min"
        )
    for low_key, high_key in (
        ("alpha_jitter_min", "alpha_jitter_max"),
        ("xi_multiplier_min", "xi_multiplier_max"),
    ):
        if float(paper[high_key]) < float(paper[low_key]):
            raise ValueError(f"{high_key} must be >= {low_key}")
    if float(paper["min_rho"]) > 1:
        raise ValueError("min_rho must be <= 1")

    fig6 = paper["fig6"]
    fig6_keys = {
        "aa_count",
        "low_ea_compute_tflops",
        "high_ea_compute_tflops",
        "low_anchor_latency_seconds",
        "low_anchor_kv_count",
        "high_anchor_latency_seconds",
        "high_anchor_kv_count",
    }
    _require_keys(fig6, "paper_jmsra.fig6", fig6_keys)
    aa_count = _positive_int(
        fig6["aa_count"], "paper_jmsra.fig6.aa_count"
    )
    for key in (
        "low_ea_compute_tflops",
        "high_ea_compute_tflops",
        "low_anchor_latency_seconds",
        "high_anchor_latency_seconds",
    ):
        _positive_number(fig6[key], f"paper_jmsra.fig6.{key}")
    for key in ("low_anchor_kv_count", "high_anchor_kv_count"):
        value = fig6[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"paper_jmsra.fig6.{key} must be an integer")
        if not 0 <= value <= aa_count:
            raise ValueError(
                f"paper_jmsra.fig6.{key} must be between 0 and aa_count"
            )

    fig7 = paper["fig7"]
    fig7_keys = {
        "ea_compute_tflops",
        "bandwidth_scan_aa_count",
        "bandwidth_ghz_values",
        "aa_count_scan_bandwidth_ghz",
        "aa_count_values",
    }
    _require_keys(fig7, "paper_jmsra.fig7", fig7_keys)
    _positive_number(
        fig7["ea_compute_tflops"],
        "paper_jmsra.fig7.ea_compute_tflops",
    )
    _positive_int(
        fig7["bandwidth_scan_aa_count"],
        "paper_jmsra.fig7.bandwidth_scan_aa_count",
    )
    _positive_number(
        fig7["aa_count_scan_bandwidth_ghz"],
        "paper_jmsra.fig7.aa_count_scan_bandwidth_ghz",
    )
    _positive_increasing_values(
        fig7["bandwidth_ghz_values"],
        "paper_jmsra.fig7.bandwidth_ghz_values",
    )
    _positive_increasing_values(
        fig7["aa_count_values"],
        "paper_jmsra.fig7.aa_count_values",
        integers=True,
    )


def validate_paper_multi_round_config(config: dict) -> None:
    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    _require_keys(config, "config", {"random_seed", "llm", "paper_multi_round"})
    if isinstance(config["random_seed"], bool) or not isinstance(
        config["random_seed"], int
    ):
        raise ValueError("random_seed must be an integer")

    llm = config["llm"]
    llm_keys = {
        "layers",
        "heads",
        "head_dim",
        "hidden_dim",
        "ffn_dim",
        "vocab_size",
    }
    _require_keys(llm, "llm", llm_keys)
    for key in llm_keys:
        _positive_int(llm[key], f"llm.{key}")

    paper = config["paper_multi_round"]
    required_keys = {
        "schema_version",
        "rounds",
        "max_aas",
        "active_aa_counts",
        "active_aa_counts_source",
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
        "tolerance_seconds",
        "min_rho",
        "trend",
    }
    _require_keys(paper, "paper_multi_round", required_keys)
    if paper["schema_version"] != 1:
        raise ValueError("paper_multi_round.schema_version must be 1")
    rounds = _positive_int(paper["rounds"], "paper_multi_round.rounds")
    max_aas = _positive_int(paper["max_aas"], "paper_multi_round.max_aas")

    counts = paper["active_aa_counts"]
    if not isinstance(counts, list) or len(counts) != rounds:
        raise ValueError(
            "paper_multi_round.active_aa_counts length must equal rounds"
        )
    for value in counts:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= max_aas
        ):
            raise ValueError(
                "paper_multi_round.active_aa_counts values must be between "
                "1 and max_aas"
            )
    if paper["active_aa_counts_source"] != "figure-inferred-calibration":
        raise ValueError(
            "paper_multi_round.active_aa_counts_source must be "
            "'figure-inferred-calibration'"
        )

    for key in (
        "bits_per_token",
        "kv_cache_compression_ratio",
        "bandwidth_ghz",
        "distance_min_m",
        "distance_max_m",
        "ea_compute_tflops",
        "aa_compute_tflops_min",
        "aa_compute_tflops_max",
        "ea_initial_prompt_tokens",
        "ea_generated_tokens",
        "aa_sensory_input_tokens",
        "aa_generated_tokens",
        "ea_context_limit_tokens",
        "ea_kv_debt_window_tokens",
        "aa_context_limit_tokens",
        "aa_kv_debt_window_tokens",
        "tolerance_seconds",
        "min_rho",
    ):
        _positive_number(paper[key], f"paper_multi_round.{key}")
    for key in (
        "noise_dbm_per_hz",
        "ea_transmit_power_dbm",
        "aa_transmit_power_dbm_min",
        "aa_transmit_power_dbm_max",
    ):
        _finite_number(paper[key], f"paper_multi_round.{key}")

    for low_key, high_key in (
        ("distance_min_m", "distance_max_m"),
        ("aa_compute_tflops_min", "aa_compute_tflops_max"),
        ("aa_transmit_power_dbm_min", "aa_transmit_power_dbm_max"),
    ):
        if float(paper[high_key]) < float(paper[low_key]):
            raise ValueError(
                f"paper_multi_round.{high_key} must be >= {low_key}"
            )
    if paper["path_loss_model"] != "30 + 35 log10(d)":
        raise ValueError(
            "paper_multi_round.path_loss_model must be "
            "'30 + 35 log10(d)'"
        )
    if float(paper["min_rho"]) > 1:
        raise ValueError("paper_multi_round.min_rho must be <= 1")
    for debt_key, context_key in (
        ("ea_kv_debt_window_tokens", "ea_context_limit_tokens"),
        ("aa_kv_debt_window_tokens", "aa_context_limit_tokens"),
    ):
        if float(paper[debt_key]) > float(paper[context_key]):
            raise ValueError(
                f"paper_multi_round {debt_key} debt window must not exceed "
                f"{context_key}"
            )

    trend = paper["trend"]
    trend_keys = {
        "early_round_end",
        "deep_round_start",
        "deep_round_end",
        "pure_nl_peak_min_seconds",
        "pure_nl_peak_max_seconds",
        "deep_kv_share_min",
        "proposed_prefill_ratio_max",
        "proposed_total_ratio_max",
    }
    _require_keys(trend, "paper_multi_round.trend", trend_keys)
    early_end = _positive_int(
        trend["early_round_end"],
        "paper_multi_round.trend.early_round_end",
    )
    deep_start = _positive_int(
        trend["deep_round_start"],
        "paper_multi_round.trend.deep_round_start",
    )
    deep_end = _positive_int(
        trend["deep_round_end"],
        "paper_multi_round.trend.deep_round_end",
    )
    if not early_end < deep_start <= deep_end <= rounds:
        raise ValueError("paper_multi_round.trend round bounds are invalid")
    peak_min = _positive_number(
        trend["pure_nl_peak_min_seconds"],
        "paper_multi_round.trend.pure_nl_peak_min_seconds",
    )
    peak_max = _positive_number(
        trend["pure_nl_peak_max_seconds"],
        "paper_multi_round.trend.pure_nl_peak_max_seconds",
    )
    if peak_max <= peak_min:
        raise ValueError("paper_multi_round.trend peak range is invalid")
    for key in (
        "deep_kv_share_min",
        "proposed_prefill_ratio_max",
        "proposed_total_ratio_max",
    ):
        value = _positive_number(
            trend[key], f"paper_multi_round.trend.{key}"
        )
        if value > 1:
            raise ValueError(f"paper_multi_round.trend.{key} must be <= 1")


def _sample_paper_problem_vectors(
    paper: dict, aa_count: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    streams = [
        np.random.default_rng(child)
        for child in np.random.SeedSequence(seed).spawn(5)
    ]
    alpha_rng, xi_rng, distance_rng, fading_rng, power_rng = streams
    alphas = float(paper["alpha_base"]) * alpha_rng.uniform(
        float(paper["alpha_jitter_min"]),
        float(paper["alpha_jitter_max"]),
        size=aa_count,
    )
    xis = xi_rng.uniform(
        float(paper["xi_multiplier_min"]),
        float(paper["xi_multiplier_max"]),
        size=aa_count,
    ) * alphas
    distances = distance_rng.uniform(
        float(paper["distance_min_m"]),
        float(paper["distance_max_m"]),
        size=aa_count,
    )
    fading = fading_rng.exponential(scale=1.0, size=aa_count)
    powers_dbm = power_rng.uniform(
        float(paper["aa_transmit_power_dbm_min"]),
        float(paper["aa_transmit_power_dbm_max"]),
        size=aa_count,
    )
    return alphas, xis, distances, fading, powers_dbm


def make_paper_jmsra_problem(
    config: dict,
    aa_count: int,
    ea_compute_tflops: float,
    bandwidth_ghz: float,
    seed: int,
) -> JmsraProblem:
    validate_paper_jmsra_config(config)
    if aa_count <= 0:
        raise ValueError("aa_count must be positive")
    if ea_compute_tflops <= 0:
        raise ValueError("ea_compute_tflops must be positive")
    if bandwidth_ghz <= 0:
        raise ValueError("bandwidth_ghz must be positive")
    paper = config["paper_jmsra"]
    alphas, xis, distances, fading, powers_dbm = (
        _sample_paper_problem_vectors(paper, aa_count, seed)
    )
    return JmsraProblem(
        config=transformer_config(config),
        alphas=[float(value) for value in alphas],
        xis=[float(value) for value in xis],
        gammas=[float(paper["kv_cache_compression_ratio"])] * aa_count,
        powers_watt=[dbm_to_watt(float(value)) for value in powers_dbm],
        gains=[
            channel_gain(float(distance), float(fading_power))
            for distance, fading_power in zip(distances, fading)
        ],
        bandwidth_hz=ghz_to_hz(float(bandwidth_ghz)),
        noise_psd_w_per_hz=noise_dbm_per_hz_to_watt_per_hz(
            float(paper["noise_dbm_per_hz"])
        ),
        bits_per_token=float(paper["bits_per_token"]),
        ea_compute_flops=tflops_to_flops(float(ea_compute_tflops)),
        theta0=float(paper["theta0_tokens"]),
        tolerance_seconds=float(paper["tolerance_seconds"]),
        min_rho=float(paper["min_rho"]),
    )


def make_paper_jmsra_topology(
    config: dict, aa_count: int, seed: int
) -> list[dict]:
    validate_paper_jmsra_config(config)
    if aa_count <= 0:
        raise ValueError("aa_count must be positive")
    paper = config["paper_jmsra"]
    _, _, distances, fading, powers_dbm = _sample_paper_problem_vectors(
        paper, aa_count, seed
    )
    angle_rng = np.random.default_rng(np.random.SeedSequence([seed, 6]))
    angles = angle_rng.uniform(0.0, 2.0 * math.pi, size=aa_count)
    return [
        {
            "agent_index": index,
            "x_m": float(distance * math.cos(angle)),
            "y_m": float(distance * math.sin(angle)),
            "distance_m": float(distance),
            "fading_power": float(fading_power),
            "gain": channel_gain(float(distance), float(fading_power)),
            "power_watt": dbm_to_watt(float(power_dbm)),
        }
        for index, (distance, angle, fading_power, power_dbm) in enumerate(
            zip(distances, angles, fading, powers_dbm)
        )
    ]
