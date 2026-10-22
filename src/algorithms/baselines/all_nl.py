from dataclasses import dataclass

from src.algorithms.jmsra import JmsraProblem, calc_latency
from src.channels.wireless_channel import uplink_rate_bps
from src.utils.latency import ea_prefill_latency_for_modes, token_bits


@dataclass(frozen=True)
class BaselineResult:
    name: str
    modes: list[int]
    rho: list[float]
    latency_seconds: float


def _average_latency(problem: JmsraProblem, modes: list[int]) -> float:
    prefill = ea_prefill_latency_for_modes(
        problem.config,
        modes,
        problem.alphas,
        problem.theta0,
        problem.ea_compute_flops,
    )
    rho = [1.0 / problem.size] * problem.size
    latencies = []
    for index, mode in enumerate(modes):
        bits = (
            token_bits(problem.alphas[index], problem.bits_per_token)
            if mode == 0
            else 0
        )
        rate = uplink_rate_bps(
            problem.powers_watt[index],
            problem.gains[index],
            problem.bandwidth_hz,
            problem.noise_psd_w_per_hz,
            rho[index],
        )
        latencies.append(bits / rate)
    return prefill + max(latencies)


def run_all_nl(
    problem: JmsraProblem, optimized: bool = False
) -> BaselineResult:
    modes = [0] * problem.size
    if optimized:
        latency, rho = calc_latency(problem, modes)
    else:
        rho = [1.0 / problem.size] * problem.size
        latency = _average_latency(problem, modes)
    return BaselineResult(
        "All-NL-Opt" if optimized else "All-NL", modes, rho, latency
    )
