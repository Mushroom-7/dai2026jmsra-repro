import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.algorithms.jmsra import JmsraProblem, solve_jmsra
from src.channels.wireless_channel import broadcast_rate_bps, channel_gain
from src.models.transformer import (
    TransformerConfig,
    autoregressive_latency_seconds,
    prefill_latency_seconds,
)
from src.utils.data_gen import (
    transformer_config,
    validate_paper_multi_round_config,
)
from src.utils.latency import (
    ea_prefill_latency_for_modes,
    kv_cache_bits,
    token_bits,
)
from src.utils.units import (
    dbm_to_watt,
    ghz_to_hz,
    noise_dbm_per_hz_to_watt_per_hz,
    tflops_to_flops,
)


def _validate_nonnegative_finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _validate_positive_finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


@dataclass(frozen=True)
class DialogueState:
    ea_context_tokens: float
    ea_kv_debt_tokens: float
    aa_context_tokens: tuple[float, ...]
    aa_kv_debt_tokens: tuple[float, ...]

    def __post_init__(self) -> None:
        _validate_nonnegative_finite(
            self.ea_context_tokens, "ea_context_tokens"
        )
        _validate_nonnegative_finite(
            self.ea_kv_debt_tokens, "ea_kv_debt_tokens"
        )
        if len(self.aa_context_tokens) != len(self.aa_kv_debt_tokens):
            raise ValueError("AA context and debt vectors must have same length")
        for value in self.aa_context_tokens:
            _validate_nonnegative_finite(value, "AA context values")
        for value in self.aa_kv_debt_tokens:
            _validate_nonnegative_finite(value, "AA debt values")


@dataclass(frozen=True)
class TransferUpdate:
    context_tokens: float
    debt_tokens: float
    transferred_context_tokens: float


@dataclass(frozen=True)
class BroadcastDecision:
    mode: int
    nl_latency_seconds: float
    kv_latency_seconds: float
    rate_bps: float


@dataclass(frozen=True)
class AgentProfile:
    agent_index: int
    compute_tflops: float
    compute_flops: float
    transmit_power_dbm: float
    transmit_power_watt: float


@dataclass(frozen=True)
class RoundChannels:
    distances_m: tuple[float, ...]
    fading_powers: tuple[float, ...]
    gains: tuple[float, ...]


@dataclass(frozen=True)
class RoundResult:
    round_index: int
    active_agents: tuple[int, ...]
    ea_mode: int
    aa_modes: tuple[int, ...]
    aa_bandwidth: tuple[float, ...]
    broadcast_rate_bps: float
    broadcast_nl_latency_seconds: float
    broadcast_kv_latency_seconds: float
    jmsra_latency_seconds: float
    communication_latency_seconds: float
    ours_ea_prefill_seconds: float
    ours_ea_autoregressive_seconds: float
    ours_ea_total_seconds: float
    pure_nl_ea_prefill_seconds: float
    pure_nl_ea_autoregressive_seconds: float
    pure_nl_ea_total_seconds: float
    ea_autoregressive_context_tokens: float
    aa_debt_before_return: tuple[float, ...]
    channels: RoundChannels
    state_before: DialogueState
    state_after: DialogueState


@dataclass(frozen=True)
class ExperimentResult:
    agent_profiles: tuple[AgentProfile, ...]
    rounds: tuple[RoundResult, ...]


@dataclass(frozen=True)
class MultiRoundExperiment:
    config: TransformerConfig
    random_seed: int
    active_aa_counts: tuple[int, ...]
    max_aas: int
    bits_per_token: float
    compression_ratio: float
    bandwidth_hz: float
    noise_psd_w_per_hz: float
    distance_min_m: float
    distance_max_m: float
    ea_compute_flops: float
    aa_compute_tflops_min: float
    aa_compute_tflops_max: float
    ea_transmit_power_watt: float
    aa_transmit_power_dbm_min: float
    aa_transmit_power_dbm_max: float
    ea_initial_prompt_tokens: float
    ea_generated_tokens: float
    aa_sensory_input_tokens: float
    aa_generated_tokens: float
    ea_context_limit_tokens: float
    ea_kv_debt_window_tokens: float
    aa_context_limit_tokens: float
    aa_kv_debt_window_tokens: float
    tolerance_seconds: float
    min_rho: float

    def __post_init__(self) -> None:
        if len(self.active_aa_counts) == 0:
            raise ValueError("active_aa_counts must be nonempty")
        if any(
            count < 1 or count > self.max_aas
            for count in self.active_aa_counts
        ):
            raise ValueError("active_aa_counts must be within max_aas")


def accumulate_debt(current: float, increment: float, window: float) -> float:
    current_value = _validate_nonnegative_finite(current, "current")
    increment_value = _validate_nonnegative_finite(increment, "increment")
    window_value = _validate_positive_finite(window, "window")
    return min(current_value + increment_value, window_value)


def settle_sender_debt(
    current: float,
    alpha_tokens: float,
    mode: int,
    window: float,
) -> float:
    if isinstance(mode, bool) or mode not in (0, 1):
        raise ValueError("mode must be 0 or 1")
    if mode == 1:
        return 0.0
    return accumulate_debt(current, alpha_tokens, window)


def receive_transfer(
    context_tokens: float,
    debt_tokens: float,
    alpha_tokens: float,
    mode: int,
    context_limit_tokens: float,
) -> TransferUpdate:
    context = _validate_nonnegative_finite(
        context_tokens, "context_tokens"
    )
    debt = _validate_nonnegative_finite(debt_tokens, "debt_tokens")
    alpha = _validate_nonnegative_finite(alpha_tokens, "alpha_tokens")
    context_limit = _validate_positive_finite(
        context_limit_tokens, "context_limit_tokens"
    )
    if isinstance(mode, bool) or mode not in (0, 1):
        raise ValueError("mode must be 0 or 1")
    transferred = alpha
    return TransferUpdate(
        context_tokens=min(context + transferred, context_limit),
        debt_tokens=debt if mode == 0 else 0.0,
        transferred_context_tokens=transferred,
    )


def select_ea_broadcast_mode(
    config: TransformerConfig,
    alpha_tokens: float,
    debt_tokens: float,
    receiver_context_tokens: Sequence[float],
    receiver_compute_flops: Sequence[float],
    rate_bps: float,
    bits_per_token: float,
    compression_ratio: float,
) -> BroadcastDecision:
    if not receiver_context_tokens:
        raise ValueError("receiver vectors must be nonempty")
    if len(receiver_context_tokens) != len(receiver_compute_flops):
        raise ValueError("receiver vectors must have same length")
    alpha = _validate_nonnegative_finite(alpha_tokens, "alpha_tokens")
    debt = _validate_nonnegative_finite(debt_tokens, "debt_tokens")
    rate = _validate_positive_finite(rate_bps, "rate_bps")
    token_width = _validate_positive_finite(
        bits_per_token, "bits_per_token"
    )
    compression = _validate_positive_finite(
        compression_ratio, "compression_ratio"
    )
    nl_transmission = token_bits(alpha, token_width) / rate
    nl_latency = max(
        prefill_latency_seconds(
            config,
            alpha,
            _validate_nonnegative_finite(theta, "receiver context"),
            _validate_positive_finite(compute, "receiver compute"),
        )
        + nl_transmission
        for theta, compute in zip(
            receiver_context_tokens, receiver_compute_flops
        )
    )
    kv_latency = kv_cache_bits(
        config, alpha, debt, compression
    ) / rate
    return BroadcastDecision(
        mode=int(kv_latency < nl_latency),
        nl_latency_seconds=nl_latency,
        kv_latency_seconds=kv_latency,
        rate_bps=rate,
    )


def build_experiment(config: dict) -> MultiRoundExperiment:
    validate_paper_multi_round_config(config)
    paper = config["paper_multi_round"]
    return MultiRoundExperiment(
        config=transformer_config(config),
        random_seed=int(config["random_seed"]),
        active_aa_counts=tuple(int(value) for value in paper["active_aa_counts"]),
        max_aas=int(paper["max_aas"]),
        bits_per_token=float(paper["bits_per_token"]),
        compression_ratio=float(paper["kv_cache_compression_ratio"]),
        bandwidth_hz=ghz_to_hz(float(paper["bandwidth_ghz"])),
        noise_psd_w_per_hz=noise_dbm_per_hz_to_watt_per_hz(
            float(paper["noise_dbm_per_hz"])
        ),
        distance_min_m=float(paper["distance_min_m"]),
        distance_max_m=float(paper["distance_max_m"]),
        ea_compute_flops=tflops_to_flops(
            float(paper["ea_compute_tflops"])
        ),
        aa_compute_tflops_min=float(paper["aa_compute_tflops_min"]),
        aa_compute_tflops_max=float(paper["aa_compute_tflops_max"]),
        ea_transmit_power_watt=dbm_to_watt(
            float(paper["ea_transmit_power_dbm"])
        ),
        aa_transmit_power_dbm_min=float(
            paper["aa_transmit_power_dbm_min"]
        ),
        aa_transmit_power_dbm_max=float(
            paper["aa_transmit_power_dbm_max"]
        ),
        ea_initial_prompt_tokens=float(paper["ea_initial_prompt_tokens"]),
        ea_generated_tokens=float(paper["ea_generated_tokens"]),
        aa_sensory_input_tokens=float(paper["aa_sensory_input_tokens"]),
        aa_generated_tokens=float(paper["aa_generated_tokens"]),
        ea_context_limit_tokens=float(paper["ea_context_limit_tokens"]),
        ea_kv_debt_window_tokens=float(
            paper["ea_kv_debt_window_tokens"]
        ),
        aa_context_limit_tokens=float(paper["aa_context_limit_tokens"]),
        aa_kv_debt_window_tokens=float(
            paper["aa_kv_debt_window_tokens"]
        ),
        tolerance_seconds=float(paper["tolerance_seconds"]),
        min_rho=float(paper["min_rho"]),
    )


def _sample_environment(
    experiment: MultiRoundExperiment,
) -> tuple[tuple[AgentProfile, ...], tuple[RoundChannels, ...]]:
    streams = [
        np.random.default_rng(seed)
        for seed in np.random.SeedSequence(experiment.random_seed).spawn(4)
    ]
    compute_rng, power_rng, distance_rng, fading_rng = streams
    compute_tflops = compute_rng.uniform(
        experiment.aa_compute_tflops_min,
        experiment.aa_compute_tflops_max,
        size=experiment.max_aas,
    )
    power_dbm = power_rng.uniform(
        experiment.aa_transmit_power_dbm_min,
        experiment.aa_transmit_power_dbm_max,
        size=experiment.max_aas,
    )
    profiles = tuple(
        AgentProfile(
            agent_index=index,
            compute_tflops=float(compute),
            compute_flops=tflops_to_flops(float(compute)),
            transmit_power_dbm=float(power),
            transmit_power_watt=dbm_to_watt(float(power)),
        )
        for index, (compute, power) in enumerate(
            zip(compute_tflops, power_dbm)
        )
    )
    shape = (len(experiment.active_aa_counts), experiment.max_aas)
    distances = distance_rng.uniform(
        experiment.distance_min_m,
        experiment.distance_max_m,
        size=shape,
    )
    fading = fading_rng.exponential(scale=1.0, size=shape)
    channels = tuple(
        RoundChannels(
            distances_m=tuple(float(value) for value in distances[row]),
            fading_powers=tuple(float(value) for value in fading[row]),
            gains=tuple(
                channel_gain(float(distance), float(fading_power))
                for distance, fading_power in zip(
                    distances[row], fading[row]
                )
            ),
        )
        for row in range(shape[0])
    )
    return profiles, channels


def _run_round(
    experiment: MultiRoundExperiment,
    profiles: tuple[AgentProfile, ...],
    channels: RoundChannels,
    state: DialogueState,
    round_index: int,
    active_count: int,
) -> RoundResult:
    active = tuple(range(active_count))
    active_gains = [channels.gains[index] for index in active]
    broadcast_rate = broadcast_rate_bps(
        experiment.ea_transmit_power_watt,
        active_gains,
        experiment.bandwidth_hz,
        experiment.noise_psd_w_per_hz,
    )
    broadcast = select_ea_broadcast_mode(
        experiment.config,
        experiment.ea_generated_tokens,
        state.ea_kv_debt_tokens,
        [state.aa_context_tokens[index] for index in active],
        [profiles[index].compute_flops for index in active],
        broadcast_rate,
        experiment.bits_per_token,
        experiment.compression_ratio,
    )

    aa_context = list(state.aa_context_tokens)
    aa_debt = list(state.aa_kv_debt_tokens)
    broadcast_payload = experiment.ea_generated_tokens
    ea_debt_after_broadcast = settle_sender_debt(
        state.ea_kv_debt_tokens,
        experiment.ea_generated_tokens,
        broadcast.mode,
        experiment.ea_kv_debt_window_tokens,
    )
    for index in active:
        aa_context[index] = min(
            aa_context[index]
            + broadcast_payload
            + experiment.aa_sensory_input_tokens
            + experiment.aa_generated_tokens,
            experiment.aa_context_limit_tokens,
        )
        aa_debt[index] = accumulate_debt(
            aa_debt[index],
            experiment.aa_sensory_input_tokens,
            experiment.aa_kv_debt_window_tokens,
        )
    debt_before_return = tuple(aa_debt[index] for index in active)

    problem = JmsraProblem(
        config=experiment.config,
        alphas=[experiment.aa_generated_tokens] * active_count,
        xis=list(debt_before_return),
        gammas=[experiment.compression_ratio] * active_count,
        powers_watt=[profiles[index].transmit_power_watt for index in active],
        gains=active_gains,
        bandwidth_hz=experiment.bandwidth_hz,
        noise_psd_w_per_hz=experiment.noise_psd_w_per_hz,
        bits_per_token=experiment.bits_per_token,
        ea_compute_flops=experiment.ea_compute_flops,
        theta0=state.ea_context_tokens,
        tolerance_seconds=experiment.tolerance_seconds,
        min_rho=experiment.min_rho,
    )
    jmsra = solve_jmsra(problem)

    alphas = [experiment.aa_generated_tokens] * active_count
    ours_prefill = ea_prefill_latency_for_modes(
        experiment.config,
        jmsra.modes,
        alphas,
        state.ea_context_tokens,
        experiment.ea_compute_flops,
    )
    received_payload = tuple(
        alpha if mode == 0 else debt + alpha
        for mode, debt, alpha in zip(
            jmsra.modes, debt_before_return, alphas
        )
    )
    ea_ar_context = min(
        state.ea_context_tokens + sum(alphas),
        experiment.ea_context_limit_tokens,
    )
    ours_autoregressive = autoregressive_latency_seconds(
        experiment.config,
        experiment.ea_generated_tokens,
        ea_ar_context,
        experiment.ea_compute_flops,
    )
    pure_nl_prefill = prefill_latency_seconds(
        experiment.config,
        sum(alphas),
        state.ea_context_tokens,
        experiment.ea_compute_flops,
    )
    pure_nl_context = min(
        state.ea_context_tokens + sum(alphas),
        experiment.ea_context_limit_tokens,
    )
    pure_nl_autoregressive = autoregressive_latency_seconds(
        experiment.config,
        experiment.ea_generated_tokens,
        pure_nl_context,
        experiment.ea_compute_flops,
    )

    for local_index, mode in enumerate(jmsra.modes):
        index = active[local_index]
        aa_debt[index] = settle_sender_debt(
            aa_debt[index],
            experiment.aa_generated_tokens,
            mode,
            experiment.aa_kv_debt_window_tokens,
        )
    ea_debt = accumulate_debt(
        ea_debt_after_broadcast,
        sum(received_payload),
        experiment.ea_kv_debt_window_tokens,
    )
    state_after = DialogueState(
        ea_context_tokens=min(
            ea_ar_context + experiment.ea_generated_tokens,
            experiment.ea_context_limit_tokens,
        ),
        ea_kv_debt_tokens=ea_debt,
        aa_context_tokens=tuple(aa_context),
        aa_kv_debt_tokens=tuple(aa_debt),
    )
    return RoundResult(
        round_index=round_index,
        active_agents=active,
        ea_mode=broadcast.mode,
        aa_modes=tuple(jmsra.modes),
        aa_bandwidth=tuple(jmsra.rho),
        broadcast_rate_bps=broadcast_rate,
        broadcast_nl_latency_seconds=broadcast.nl_latency_seconds,
        broadcast_kv_latency_seconds=broadcast.kv_latency_seconds,
        jmsra_latency_seconds=jmsra.latency_seconds,
        communication_latency_seconds=jmsra.transmission_latency_seconds,
        ours_ea_prefill_seconds=ours_prefill,
        ours_ea_autoregressive_seconds=ours_autoregressive,
        ours_ea_total_seconds=ours_prefill + ours_autoregressive,
        pure_nl_ea_prefill_seconds=pure_nl_prefill,
        pure_nl_ea_autoregressive_seconds=pure_nl_autoregressive,
        pure_nl_ea_total_seconds=(
            pure_nl_prefill + pure_nl_autoregressive
        ),
        ea_autoregressive_context_tokens=ea_ar_context,
        aa_debt_before_return=debt_before_return,
        channels=channels,
        state_before=state,
        state_after=state_after,
    )


def run_experiment(experiment: MultiRoundExperiment) -> ExperimentResult:
    profiles, channels = _sample_environment(experiment)
    state = DialogueState(
        ea_context_tokens=experiment.ea_initial_prompt_tokens,
        ea_kv_debt_tokens=experiment.ea_initial_prompt_tokens,
        aa_context_tokens=(0.0,) * experiment.max_aas,
        aa_kv_debt_tokens=(0.0,) * experiment.max_aas,
    )
    rounds = []
    for offset, active_count in enumerate(experiment.active_aa_counts):
        result = _run_round(
            experiment,
            profiles,
            channels[offset],
            state,
            round_index=offset + 1,
            active_count=active_count,
        )
        rounds.append(result)
        state = result.state_after
    return ExperimentResult(profiles, tuple(rounds))
