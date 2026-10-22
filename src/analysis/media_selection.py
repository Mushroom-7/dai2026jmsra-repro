import math
from dataclasses import dataclass

from src.models.transformer import (
    TransformerConfig,
    autoregressive_latency_seconds,
    inference_latency_seconds,
    prefill_latency_seconds,
)
from src.utils.latency import kv_cache_bits, token_bits


def _validate_rate_inputs(
    bandwidth_hz: float, snr_db: float, rho: float
) -> None:
    if not math.isfinite(bandwidth_hz) or bandwidth_hz <= 0:
        raise ValueError("bandwidth_hz must be finite and positive")
    if not math.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    if not math.isfinite(rho) or not 0 < rho <= 1:
        raise ValueError("rho must be in (0, 1]")


def nominal_snr_rate_bps(
    bandwidth_hz: float, snr_db: float, rho: float = 1.0
) -> float:
    _validate_rate_inputs(bandwidth_hz, snr_db, rho)
    snr_linear = 10.0 ** (snr_db / 10.0)
    effective_bandwidth = rho * bandwidth_hz
    return effective_bandwidth * math.log2(1.0 + snr_linear / rho)


def allocated_link_snr_rate_bps(
    bandwidth_hz: float, snr_db: float, rho: float = 1.0
) -> float:
    _validate_rate_inputs(bandwidth_hz, snr_db, rho)
    snr_linear = 10.0 ** (snr_db / 10.0)
    return rho * bandwidth_hz * math.log2(1.0 + snr_linear)


@dataclass(frozen=True)
class StaticE2EScenario:
    config: TransformerConfig
    aa_count: int
    input_tokens: float
    beta: float
    final_output_tokens: float
    ea_compute_flops: float
    aa_compute_flops: float
    bandwidth_hz: float
    snr_db: float
    bits_per_token: float
    compression_ratio: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.aa_count, bool)
            or not isinstance(self.aa_count, int)
            or self.aa_count <= 0
        ):
            raise ValueError("aa_count must be a positive integer")
        for name in (
            "input_tokens",
            "final_output_tokens",
            "ea_compute_flops",
            "aa_compute_flops",
            "bandwidth_hz",
            "bits_per_token",
            "compression_ratio",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.beta) or self.beta < 0:
            raise ValueError("beta must be finite and nonnegative")
        if not math.isfinite(self.snr_db):
            raise ValueError("snr_db must be finite")


def _workload(
    scenario: StaticE2EScenario,
) -> tuple[float, float, float, float, float, float, float]:
    s = scenario.input_tokens
    alpha_01 = scenario.beta * s
    s_i1 = alpha_01
    alpha_i1 = scenario.beta * s_i1
    s_i2 = s
    alpha_i2 = scenario.beta * s_i2
    s_02 = scenario.aa_count * alpha_i2
    return s, alpha_01, s_i1, alpha_i1, s_i2, alpha_i2, s_02


def static_nl_e2e_latency_seconds(scenario: StaticE2EScenario) -> float:
    s, alpha_01, s_i1, alpha_i1, s_i2, alpha_i2, s_02 = _workload(
        scenario
    )
    ea_1 = inference_latency_seconds(
        scenario.config, alpha_01, 0.0, s, scenario.ea_compute_flops
    )
    aa_1 = inference_latency_seconds(
        scenario.config, alpha_i1, 0.0, s_i1, scenario.aa_compute_flops
    )
    aa_2 = inference_latency_seconds(
        scenario.config,
        alpha_i2,
        s_i1 + alpha_i1,
        s_i2,
        scenario.aa_compute_flops,
    )
    ea_2 = inference_latency_seconds(
        scenario.config,
        scenario.final_output_tokens,
        s + alpha_01,
        s_02,
        scenario.ea_compute_flops,
    )
    broadcast_rate = allocated_link_snr_rate_bps(
        scenario.bandwidth_hz, scenario.snr_db
    )
    uplink_rate = allocated_link_snr_rate_bps(
        scenario.bandwidth_hz,
        scenario.snr_db,
        1.0 / scenario.aa_count,
    )
    broadcast = token_bits(alpha_01, scenario.bits_per_token) / broadcast_rate
    uplink = token_bits(alpha_i2, scenario.bits_per_token) / uplink_rate
    return ea_1 + ea_2 + broadcast + aa_1 + aa_2 + uplink


def static_kv_e2e_latency_seconds(scenario: StaticE2EScenario) -> float:
    s, alpha_01, _, alpha_i1, s_i2, alpha_i2, _ = _workload(scenario)
    ea_1 = inference_latency_seconds(
        scenario.config, alpha_01, 0.0, s, scenario.ea_compute_flops
    )
    aa_1 = autoregressive_latency_seconds(
        scenario.config,
        alpha_i1,
        s + alpha_01,
        scenario.aa_compute_flops,
    )
    aa_2 = inference_latency_seconds(
        scenario.config,
        alpha_i2,
        s + alpha_01 + alpha_i1,
        s_i2,
        scenario.aa_compute_flops,
    )
    eta = s + alpha_01 + scenario.aa_count * (
        alpha_i1 + s_i2 + alpha_i2
    )
    ea_2 = autoregressive_latency_seconds(
        scenario.config,
        scenario.final_output_tokens,
        eta,
        scenario.ea_compute_flops,
    )
    broadcast_bits = kv_cache_bits(
        scenario.config, alpha_01, s, scenario.compression_ratio
    )
    uplink_bits = kv_cache_bits(
        scenario.config,
        alpha_i2,
        alpha_i1 + s_i2,
        scenario.compression_ratio,
    )
    broadcast_rate = allocated_link_snr_rate_bps(
        scenario.bandwidth_hz, scenario.snr_db
    )
    uplink_rate = allocated_link_snr_rate_bps(
        scenario.bandwidth_hz,
        scenario.snr_db,
        1.0 / scenario.aa_count,
    )
    broadcast = broadcast_bits / broadcast_rate
    uplink = uplink_bits / uplink_rate
    return ea_1 + ea_2 + broadcast + aa_1 + aa_2 + uplink


def static_latency_ratio(scenario: StaticE2EScenario) -> float:
    return static_nl_e2e_latency_seconds(
        scenario
    ) / static_kv_e2e_latency_seconds(scenario)


def _validate_single_link_inputs(
    alpha: float,
    xi: float,
    theta: float,
    compute_flops: float,
    bandwidth_hz: float,
    snr_db: float,
    rho: float,
    bits_per_token: float,
    compression_ratio: float,
) -> None:
    for name, value in (("alpha", alpha), ("xi", xi), ("theta", theta)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    for name, value in (
        ("compute_flops", compute_flops),
        ("bits_per_token", bits_per_token),
        ("compression_ratio", compression_ratio),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    _validate_rate_inputs(bandwidth_hz, snr_db, rho)


def single_link_mode_score(
    config: TransformerConfig,
    alpha: float,
    xi: float,
    theta: float,
    compute_flops: float,
    bandwidth_hz: float,
    snr_db: float,
    rho: float,
    bits_per_token: float,
    compression_ratio: float,
) -> float:
    _validate_single_link_inputs(
        alpha,
        xi,
        theta,
        compute_flops,
        bandwidth_hz,
        snr_db,
        rho,
        bits_per_token,
        compression_ratio,
    )
    rate = nominal_snr_rate_bps(bandwidth_hz, snr_db, rho)
    nl_latency = prefill_latency_seconds(
        config, alpha, theta, compute_flops
    ) + token_bits(alpha, bits_per_token) / rate
    kv_latency = kv_cache_bits(
        config, alpha, xi, compression_ratio
    ) / rate
    return nl_latency - kv_latency


def bandwidth_threshold(
    config: TransformerConfig,
    alpha: float,
    xi: float,
    theta: float,
    compute_flops: float,
    bandwidth_hz: float,
    snr_db: float,
    bits_per_token: float,
    compression_ratio: float,
    rho_tolerance: float,
    score_tolerance_seconds: float,
    max_iterations: int,
) -> float | None:
    for name, value in (
        ("rho_tolerance", rho_tolerance),
        ("score_tolerance_seconds", score_tolerance_seconds),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations <= 0
    ):
        raise ValueError("max_iterations must be a positive integer")

    def score_at(rho: float) -> float:
        return single_link_mode_score(
            config,
            alpha,
            xi,
            theta,
            compute_flops,
            bandwidth_hz,
            snr_db,
            rho,
            bits_per_token,
            compression_ratio,
        )

    high = 1.0
    high_score = score_at(high)
    if high_score <= 0:
        return None

    low = min(rho_tolerance, 1e-9)
    low_score = score_at(low)
    if low_score >= 0:
        raise ValueError("single-link score must be negative at the lower endpoint")

    for _ in range(max_iterations):
        midpoint = (low + high) / 2.0
        midpoint_score = score_at(midpoint)
        if abs(midpoint_score) <= score_tolerance_seconds:
            return midpoint
        if midpoint_score < 0:
            low = midpoint
        else:
            high = midpoint
        if high - low <= rho_tolerance:
            return (low + high) / 2.0

    raise ValueError("bandwidth threshold solver exhausted max_iterations")
