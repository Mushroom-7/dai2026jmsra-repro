import math

import pytest

from src.algorithms.jmsra import (
    JmsraProblem,
    _calc_latencies_batch,
    _required_rho_for_latency,
    _required_rhos_for_latency,
    allocate_bandwidth,
    calc_latency,
    solve_jmsra,
    solve_jmsra_with_trace,
)
from src.channels.wireless_channel import channel_gain
from src.models.transformer import TransformerConfig
from src.utils.units import (
    dbm_to_watt,
    ghz_to_hz,
    noise_dbm_per_hz_to_watt_per_hz,
    tflops_to_flops,
)


def small_problem():
    return JmsraProblem(
        config=TransformerConfig(
            layers=2,
            heads=2,
            head_dim=8,
            hidden_dim=32,
            ffn_dim=64,
            vocab_size=128,
        ),
        alphas=[128, 256, 384],
        xis=[256, 512, 768],
        gammas=[2, 2, 2],
        powers_watt=[
            dbm_to_watt(20),
            dbm_to_watt(20),
            dbm_to_watt(20),
        ],
        gains=[channel_gain(5), channel_gain(7), channel_gain(9)],
        bandwidth_hz=ghz_to_hz(0.1),
        noise_psd_w_per_hz=noise_dbm_per_hz_to_watt_per_hz(-140),
        bits_per_token=16,
        ea_compute_flops=tflops_to_flops(1),
        theta0=1024,
        tolerance_seconds=1e-7,
    )


def test_allocate_bandwidth_returns_feasible_allocation():
    problem = small_problem()
    latency, rho = allocate_bandwidth(problem, [0, 1, 0])
    assert math.isfinite(latency)
    assert all(value >= 0 for value in rho)
    assert sum(rho) <= 1.000001


def test_vectorized_required_rhos_match_scalar_reference():
    problem = small_problem()
    bits = [2048.0, 16384.0, 32768.0]
    target_latency = 0.05
    expected = [
        _required_rho_for_latency(
            problem, value, index, target_latency
        )
        for index, value in enumerate(bits)
    ]
    actual = _required_rhos_for_latency(
        problem, bits, target_latency
    )
    assert actual == pytest.approx(expected, abs=1e-12)


def test_calc_latency_returns_total_latency_and_allocation():
    problem = small_problem()
    latency, rho = calc_latency(problem, [0, 0, 0])
    assert math.isfinite(latency)
    assert len(rho) == 3


def test_batch_latencies_match_individual_calculations():
    problem = small_problem()
    modes_batch = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]]
    expected = [calc_latency(problem, modes) for modes in modes_batch]
    latencies, allocations = _calc_latencies_batch(problem, modes_batch)
    assert latencies == pytest.approx([item[0] for item in expected])
    for actual, (_, reference) in zip(allocations, expected):
        assert actual == pytest.approx(reference, abs=1e-12)


def test_solve_jmsra_returns_valid_modes_and_bandwidth():
    problem = small_problem()
    result = solve_jmsra(problem)
    assert len(result.modes) == 3
    assert set(result.modes).issubset({0, 1})
    assert all(value >= 0 for value in result.rho)
    assert sum(result.rho) <= 1.000001
    assert math.isfinite(result.latency_seconds)


def test_solve_jmsra_with_trace_preserves_final_result_schema():
    problem = small_problem()
    traced = solve_jmsra_with_trace(problem)
    plain = solve_jmsra(problem)
    assert traced.result.modes == plain.modes
    assert traced.result.rho == plain.rho
    assert traced.result.latency_seconds == plain.latency_seconds
    assert traced.selected_direction in {"forward", "backward"}


def test_jmsra_trace_paths_are_ordered_and_finite():
    problem = small_problem()
    traced = solve_jmsra_with_trace(problem)
    assert traced.forward_path
    assert traced.backward_path
    assert traced.forward_path[0].kv_count == 0
    assert traced.backward_path[0].kv_count == problem.size
    for path in (traced.forward_path, traced.backward_path):
        for point in path:
            assert len(point.modes) == problem.size
            assert set(point.modes).issubset({0, 1})
            assert math.isfinite(point.latency_seconds)
