import math

import pytest

from src.analysis.media_selection import (
    StaticE2EScenario,
    allocated_link_snr_rate_bps,
    bandwidth_threshold,
    nominal_snr_rate_bps,
    single_link_mode_score,
    static_kv_e2e_latency_seconds,
    static_latency_ratio,
    static_nl_e2e_latency_seconds,
)
from src.models.transformer import (
    TransformerConfig,
    autoregressive_latency_seconds,
    inference_latency_seconds,
    prefill_latency_seconds,
)
from src.utils.latency import kv_cache_bits, token_bits
from src.utils.units import ghz_to_hz, tflops_to_flops


def llama7b_config() -> TransformerConfig:
    return TransformerConfig(32, 32, 128, 4096, 11008, 32000)


def scenario(**overrides) -> StaticE2EScenario:
    values = {
        "config": llama7b_config(),
        "aa_count": 5,
        "input_tokens": 512.0,
        "beta": 1.0,
        "final_output_tokens": 100.0,
        "ea_compute_flops": tflops_to_flops(10),
        "aa_compute_flops": tflops_to_flops(10),
        "bandwidth_hz": ghz_to_hz(2),
        "snr_db": 10.0,
        "bits_per_token": 16.0,
        "compression_ratio": 2.0,
    }
    values.update(overrides)
    return StaticE2EScenario(**values)


def test_nominal_snr_rate_uses_full_band_snr_reference():
    full = nominal_snr_rate_bps(2e9, 10, 1.0)
    half = nominal_snr_rate_bps(2e9, 10, 0.5)
    assert full == pytest.approx(2e9 * math.log2(11))
    assert half == pytest.approx(1e9 * math.log2(21))
    assert full > half > 0


def test_allocated_link_snr_rate_keeps_post_allocation_snr_fixed():
    full = allocated_link_snr_rate_bps(2e9, 10, 1.0)
    half = allocated_link_snr_rate_bps(2e9, 10, 0.5)
    assert full == pytest.approx(2e9 * math.log2(11))
    assert half == pytest.approx(1e9 * math.log2(11))
    assert half == pytest.approx(full / 2)


def test_static_e2e_latencies_are_positive_and_ratio_is_direct():
    item = scenario()
    nl = static_nl_e2e_latency_seconds(item)
    kv = static_kv_e2e_latency_seconds(item)
    assert nl > 0
    assert kv > 0
    assert static_latency_ratio(item) == pytest.approx(nl / kv)


def test_static_e2e_respects_compute_and_channel_changes():
    base = scenario()
    fast_ea = scenario(ea_compute_flops=tflops_to_flops(20))
    high_snr = scenario(snr_db=20)
    assert static_nl_e2e_latency_seconds(fast_ea) < (
        static_nl_e2e_latency_seconds(base)
    )
    assert static_kv_e2e_latency_seconds(high_snr) < (
        static_kv_e2e_latency_seconds(base)
    )


def test_static_nl_e2e_matches_explicit_equation_8_components():
    item = scenario()
    broadcast_rate = allocated_link_snr_rate_bps(ghz_to_hz(2), 10)
    uplink_rate = allocated_link_snr_rate_bps(ghz_to_hz(2), 10, 0.2)
    expected = sum(
        (
            inference_latency_seconds(
                item.config, 512, 0, 512, item.ea_compute_flops
            ),
            inference_latency_seconds(
                item.config, 100, 1024, 2560, item.ea_compute_flops
            ),
            token_bits(512, 16) / broadcast_rate,
            inference_latency_seconds(
                item.config, 512, 0, 512, item.aa_compute_flops
            ),
            inference_latency_seconds(
                item.config, 512, 1024, 512, item.aa_compute_flops
            ),
            token_bits(512, 16) / uplink_rate,
        )
    )
    assert static_nl_e2e_latency_seconds(item) == pytest.approx(expected)


def test_static_kv_e2e_matches_explicit_equation_17_components():
    item = scenario()
    broadcast_rate = allocated_link_snr_rate_bps(ghz_to_hz(2), 10)
    uplink_rate = allocated_link_snr_rate_bps(ghz_to_hz(2), 10, 0.2)
    expected = sum(
        (
            inference_latency_seconds(
                item.config, 512, 0, 512, item.ea_compute_flops
            ),
            autoregressive_latency_seconds(
                item.config, 100, 8704, item.ea_compute_flops
            ),
            kv_cache_bits(item.config, 512, 512, 2) / broadcast_rate,
            autoregressive_latency_seconds(
                item.config, 512, 1024, item.aa_compute_flops
            ),
            inference_latency_seconds(
                item.config, 512, 1536, 512, item.aa_compute_flops
            ),
            kv_cache_bits(item.config, 512, 1024, 2) / uplink_rate,
        )
    )
    assert static_kv_e2e_latency_seconds(item) == pytest.approx(expected)


def test_nl_token_volume_is_smaller_than_kv_cache_volume():
    assert token_bits(512, 16) < kv_cache_bits(llama7b_config(), 512, 3000, 2)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("aa_count", 0),
        ("input_tokens", -1),
        ("beta", -1),
        ("bandwidth_hz", 0),
        ("compression_ratio", 0),
    ],
)
def test_static_scenario_rejects_invalid_values(field, value):
    with pytest.raises(ValueError, match=field):
        scenario(**{field: value})


def score(alpha=512, xi=3000, theta=0, snr_db=5, rho=0.5):
    return single_link_mode_score(
        llama7b_config(),
        alpha,
        xi,
        theta,
        tflops_to_flops(1),
        ghz_to_hz(2),
        snr_db,
        rho,
        16,
        2,
    )


def test_single_link_score_is_direct_marginal_latency_difference():
    rate = nominal_snr_rate_bps(ghz_to_hz(2), 5, 0.5)
    nl = prefill_latency_seconds(
        llama7b_config(), 512, 0, tflops_to_flops(1)
    ) + token_bits(512, 16) / rate
    kv = kv_cache_bits(llama7b_config(), 512, 3000, 2) / rate
    assert score() == pytest.approx(nl - kv)


def test_single_link_score_is_strictly_increasing_in_rho():
    values = [score(rho=index / 100) for index in range(1, 101)]
    assert all(left < right for left, right in zip(values, values[1:]))


def test_bandwidth_threshold_satisfies_score_and_paper_ordering():
    common = {
        "config": llama7b_config(),
        "alpha": 512,
        "theta": 0,
        "compute_flops": tflops_to_flops(1),
        "bandwidth_hz": ghz_to_hz(2),
        "bits_per_token": 16,
        "compression_ratio": 2,
        "rho_tolerance": 1e-9,
        "score_tolerance_seconds": 1e-6,
        "max_iterations": 100,
    }
    low_xi = bandwidth_threshold(xi=0, snr_db=5, **common)
    high_xi = bandwidth_threshold(xi=6000, snr_db=5, **common)
    poor_channel = bandwidth_threshold(xi=3000, snr_db=5, **common)
    good_channel = bandwidth_threshold(xi=3000, snr_db=15, **common)

    assert low_xi is not None and high_xi is not None
    assert poor_channel is not None and good_channel is not None
    assert abs(score(xi=3000, snr_db=5, rho=poor_channel)) <= 1e-6
    assert low_xi < high_xi
    assert good_channel < poor_channel
