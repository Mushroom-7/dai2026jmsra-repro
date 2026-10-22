from dataclasses import dataclass

import numpy as np

from src.channels.wireless_channel import uplink_rate_bps
from src.models.transformer import TransformerConfig
from src.utils.latency import (
    ea_prefill_latency_for_modes,
    kv_cache_bits,
    token_bits,
)


_RHO_BISECTION_ITERATIONS = 48


@dataclass(frozen=True)
class JmsraProblem:
    config: TransformerConfig
    alphas: list[float]
    xis: list[float]
    gammas: list[float]
    powers_watt: list[float]
    gains: list[float]
    bandwidth_hz: float
    noise_psd_w_per_hz: float
    bits_per_token: float
    ea_compute_flops: float
    theta0: float
    tolerance_seconds: float = 1e-6
    min_rho: float = 1e-9

    @property
    def size(self) -> int:
        return len(self.alphas)


@dataclass(frozen=True)
class JmsraResult:
    modes: list[int]
    rho: list[float]
    latency_seconds: float
    transmission_latency_seconds: float


@dataclass(frozen=True)
class JmsraTracePoint:
    kv_count: int
    latency_seconds: float
    modes: list[int]


@dataclass(frozen=True)
class JmsraTraceResult:
    result: JmsraResult
    selected_direction: str
    forward_path: list[JmsraTracePoint]
    backward_path: list[JmsraTracePoint]


def _validate_problem(problem: JmsraProblem) -> None:
    lengths = {
        len(problem.alphas),
        len(problem.xis),
        len(problem.gammas),
        len(problem.powers_watt),
        len(problem.gains),
    }
    if len(lengths) != 1:
        raise ValueError("all per-agent vectors must have equal length")
    if problem.size == 0:
        raise ValueError("problem must contain at least one AA")


def _data_bits(problem: JmsraProblem, mode: int, index: int) -> float:
    if mode == 0:
        return token_bits(problem.alphas[index], problem.bits_per_token)
    if mode == 1:
        return kv_cache_bits(
            problem.config,
            problem.alphas[index],
            problem.xis[index],
            problem.gammas[index],
        )
    raise ValueError("mode must be 0 for NL or 1 for KV")


def _required_rho_for_latency(
    problem: JmsraProblem,
    bits: float,
    index: int,
    target_latency: float,
) -> float:
    if bits == 0:
        return 0.0
    low = problem.min_rho
    high = 1.0
    high_rate = uplink_rate_bps(
        problem.powers_watt[index],
        problem.gains[index],
        problem.bandwidth_hz,
        problem.noise_psd_w_per_hz,
        high,
    )
    if bits / high_rate > target_latency:
        return float("inf")
    for _ in range(80):
        mid = (low + high) / 2.0
        rate = uplink_rate_bps(
            problem.powers_watt[index],
            problem.gains[index],
            problem.bandwidth_hz,
            problem.noise_psd_w_per_hz,
            mid,
        )
        if bits / rate <= target_latency:
            high = mid
        else:
            low = mid
    return high


def _required_rhos_for_latency(
    problem: JmsraProblem,
    data_bits: list[float],
    target_latency: float,
) -> list[float]:
    bits = np.asarray(data_bits, dtype=float)
    powers = np.asarray(problem.powers_watt, dtype=float)
    gains = np.asarray(problem.gains, dtype=float)
    full_band_rate = problem.bandwidth_hz * np.log2(
        1.0
        + powers
        * gains
        / (problem.bandwidth_hz * problem.noise_psd_w_per_hz)
    )
    infeasible = bits / full_band_rate > target_latency
    active = (bits > 0.0) & ~infeasible
    low = np.full(problem.size, problem.min_rho, dtype=float)
    high = np.ones(problem.size, dtype=float)
    for _ in range(_RHO_BISECTION_ITERATIONS):
        mid = (low + high) / 2.0
        effective_bandwidth = mid * problem.bandwidth_hz
        rates = effective_bandwidth * np.log2(
            1.0
            + powers
            * gains
            / (effective_bandwidth * problem.noise_psd_w_per_hz)
        )
        meets_target = bits / rates <= target_latency
        high = np.where(active & meets_target, mid, high)
        low = np.where(active & ~meets_target, mid, low)
    result = np.where(bits == 0.0, 0.0, high)
    result = np.where(infeasible, float("inf"), result)
    return [float(value) for value in result]


def _required_rhos_batch(
    problem: JmsraProblem,
    data_bits: np.ndarray,
    target_latencies: np.ndarray,
) -> np.ndarray:
    powers = np.asarray(problem.powers_watt, dtype=float)
    gains = np.asarray(problem.gains, dtype=float)
    full_band_rate = problem.bandwidth_hz * np.log2(
        1.0
        + powers
        * gains
        / (problem.bandwidth_hz * problem.noise_psd_w_per_hz)
    )
    targets = target_latencies[:, np.newaxis]
    infeasible = data_bits / full_band_rate > targets
    active = (data_bits > 0.0) & ~infeasible
    low = np.full(data_bits.shape, problem.min_rho, dtype=float)
    high = np.ones(data_bits.shape, dtype=float)
    for _ in range(_RHO_BISECTION_ITERATIONS):
        mid = (low + high) / 2.0
        effective_bandwidth = mid * problem.bandwidth_hz
        rates = effective_bandwidth * np.log2(
            1.0
            + powers[np.newaxis, :]
            * gains[np.newaxis, :]
            / (effective_bandwidth * problem.noise_psd_w_per_hz)
        )
        meets_target = data_bits / rates <= targets
        high = np.where(active & meets_target, mid, high)
        low = np.where(active & ~meets_target, mid, low)
    result = np.where(data_bits == 0.0, 0.0, high)
    return np.where(infeasible, float("inf"), result)


def _allocate_bandwidth_batch(
    problem: JmsraProblem, modes_batch: list[list[int]]
) -> tuple[np.ndarray, np.ndarray]:
    _validate_problem(problem)
    if not modes_batch:
        raise ValueError("modes_batch must be nonempty")
    if any(len(modes) != problem.size for modes in modes_batch):
        raise ValueError("each modes vector must equal problem size")
    if any(mode not in (0, 1) for modes in modes_batch for mode in modes):
        raise ValueError("mode must be 0 for NL or 1 for KV")
    data_bits = np.asarray(
        [
            [
                _data_bits(problem, mode, index)
                for index, mode in enumerate(modes)
            ]
            for modes in modes_batch
        ],
        dtype=float,
    )
    full_rates = np.asarray(
        [
            uplink_rate_bps(
                problem.powers_watt[index],
                problem.gains[index],
                problem.bandwidth_hz,
                problem.noise_psd_w_per_hz,
                1.0,
            )
            for index in range(problem.size)
        ],
        dtype=float,
    )
    high = np.max(data_bits / full_rates, axis=1)
    high = np.maximum(high, problem.tolerance_seconds)
    while True:
        required = _required_rhos_batch(problem, data_bits, high)
        feasible = np.sum(required, axis=1) <= 1.0
        if bool(np.all(feasible)):
            break
        high = np.where(feasible, high, high * 2.0)

    low = np.zeros_like(high)
    best_rho = required.copy()
    while bool(np.any(high - low > problem.tolerance_seconds)):
        active_rows = high - low > problem.tolerance_seconds
        trial = np.where(active_rows, (low + high) / 2.0, high)
        required = _required_rhos_batch(problem, data_bits, trial)
        feasible = np.sum(required, axis=1) <= 1.0
        move_high = active_rows & feasible
        move_low = active_rows & ~feasible
        high = np.where(move_high, trial, high)
        low = np.where(move_low, trial, low)
        best_rho = np.where(move_high[:, np.newaxis], required, best_rho)
    return high, best_rho


def _calc_latencies_batch(
    problem: JmsraProblem, modes_batch: list[list[int]]
) -> tuple[list[float], list[list[float]]]:
    transmission, allocations = _allocate_bandwidth_batch(
        problem, modes_batch
    )
    prefill = np.asarray(
        [
            ea_prefill_latency_for_modes(
                problem.config,
                modes,
                problem.alphas,
                problem.theta0,
                problem.ea_compute_flops,
            )
            for modes in modes_batch
        ],
        dtype=float,
    )
    return (
        [float(value) for value in prefill + transmission],
        [
            [float(value) for value in allocation]
            for allocation in allocations
        ],
    )


def allocate_bandwidth(
    problem: JmsraProblem, modes: list[int]
) -> tuple[float, list[float]]:
    _validate_problem(problem)
    if len(modes) != problem.size:
        raise ValueError("modes length must equal problem size")
    data_bits = [
        _data_bits(problem, mode, index)
        for index, mode in enumerate(modes)
    ]
    high = max(
        bits
        / uplink_rate_bps(
            problem.powers_watt[index],
            problem.gains[index],
            problem.bandwidth_hz,
            problem.noise_psd_w_per_hz,
            1.0,
        )
        for index, bits in enumerate(data_bits)
    )
    high = max(high, problem.tolerance_seconds)
    while True:
        required = _required_rhos_for_latency(problem, data_bits, high)
        if sum(required) <= 1.0:
            break
        high *= 2.0

    low = 0.0
    best_rho = required
    while high - low > problem.tolerance_seconds:
        trial = (low + high) / 2.0
        required = _required_rhos_for_latency(problem, data_bits, trial)
        if sum(required) <= 1.0:
            high = trial
            best_rho = required
        else:
            low = trial
    return high, best_rho


def calc_latency(
    problem: JmsraProblem, modes: list[int]
) -> tuple[float, list[float]]:
    prefill = ea_prefill_latency_for_modes(
        problem.config,
        modes,
        problem.alphas,
        problem.theta0,
        problem.ea_compute_flops,
    )
    transmission, rho = allocate_bandwidth(problem, modes)
    return prefill + transmission, rho


def _trace_point(modes: list[int], latency: float) -> JmsraTracePoint:
    return JmsraTracePoint(
        kv_count=sum(modes),
        latency_seconds=latency,
        modes=list(modes),
    )


def _greedy_search_with_trace(
    problem: JmsraProblem, initial_value: int, target_value: int
) -> tuple[list[int], float, list[float], list[JmsraTracePoint]]:
    modes = [initial_value] * problem.size
    candidates = set(range(problem.size))
    current_latency, current_rho = calc_latency(problem, modes)
    path = [_trace_point(modes, current_latency)]
    while candidates:
        candidate_indices = sorted(candidates)
        trial_modes_batch = []
        for index in candidate_indices:
            trial_modes = list(modes)
            trial_modes[index] = target_value
            trial_modes_batch.append(trial_modes)
        trial_latencies, trial_allocations = _calc_latencies_batch(
            problem, trial_modes_batch
        )
        scored = []
        for index, trial_latency, trial_rho in zip(
            candidate_indices, trial_latencies, trial_allocations
        ):
            scored.append(
                (
                    current_latency - trial_latency,
                    index,
                    trial_latency,
                    trial_rho,
                )
            )
        improvement, index, next_latency, next_rho = max(
            scored, key=lambda item: item[0]
        )
        if improvement <= 0:
            break
        modes[index] = target_value
        candidates.remove(index)
        current_latency = next_latency
        current_rho = next_rho
        path.append(_trace_point(modes, current_latency))
    return modes, current_latency, current_rho, path


def _greedy_search(
    problem: JmsraProblem, initial_value: int, target_value: int
) -> tuple[list[int], float, list[float]]:
    modes, latency, rho, _ = _greedy_search_with_trace(
        problem, initial_value, target_value
    )
    return modes, latency, rho


def solve_jmsra(problem: JmsraProblem) -> JmsraResult:
    _validate_problem(problem)
    forward_modes, forward_latency, forward_rho = _greedy_search(
        problem, 0, 1
    )
    backward_modes, backward_latency, backward_rho = _greedy_search(
        problem, 1, 0
    )
    if forward_latency <= backward_latency:
        transmission, rho = allocate_bandwidth(problem, forward_modes)
        return JmsraResult(
            forward_modes, rho, forward_latency, transmission
        )
    transmission, rho = allocate_bandwidth(problem, backward_modes)
    return JmsraResult(backward_modes, rho, backward_latency, transmission)


def solve_jmsra_with_trace(problem: JmsraProblem) -> JmsraTraceResult:
    _validate_problem(problem)
    forward_modes, forward_latency, _, forward_path = (
        _greedy_search_with_trace(problem, 0, 1)
    )
    backward_modes, backward_latency, _, backward_path = (
        _greedy_search_with_trace(problem, 1, 0)
    )
    if forward_latency <= backward_latency:
        transmission, rho = allocate_bandwidth(problem, forward_modes)
        return JmsraTraceResult(
            JmsraResult(forward_modes, rho, forward_latency, transmission),
            "forward",
            forward_path,
            backward_path,
        )
    transmission, rho = allocate_bandwidth(problem, backward_modes)
    return JmsraTraceResult(
        JmsraResult(backward_modes, rho, backward_latency, transmission),
        "backward",
        forward_path,
        backward_path,
    )
