from src.models.transformer import TransformerConfig
from src.utils.latency import (
    ea_prefill_latency_for_modes,
    kv_cache_bits,
    token_bits,
    transmission_latency_seconds,
)
from src.utils.units import tflops_to_flops


def llama7b_config():
    return TransformerConfig(
        layers=32,
        heads=32,
        head_dim=128,
        hidden_dim=4096,
        ffn_dim=11008,
        vocab_size=32000,
    )


def test_token_and_kv_data_volume_formulas():
    cfg = llama7b_config()
    assert token_bits(alpha=1024, bits_per_token=16) == 16384
    small = kv_cache_bits(
        cfg, alpha=512, unshared_context=1024, compression_ratio=2
    )
    large = kv_cache_bits(
        cfg, alpha=1024, unshared_context=2048, compression_ratio=2
    )
    compressed = kv_cache_bits(
        cfg, alpha=512, unshared_context=1024, compression_ratio=4
    )
    assert large > small
    assert compressed == small / 2


def test_transmission_latency_is_bits_over_rate():
    assert transmission_latency_seconds(bits=2000, rate_bps=1000) == 2


def test_ea_prefill_latency_uses_only_nl_modes_as_input_tokens():
    cfg = llama7b_config()
    compute = tflops_to_flops(10)
    all_nl = ea_prefill_latency_for_modes(
        cfg, [0, 0, 0], [100, 200, 300], theta0=1000, compute_flops=compute
    )
    mixed = ea_prefill_latency_for_modes(
        cfg, [0, 1, 1], [100, 200, 300], theta0=1000, compute_flops=compute
    )
    all_kv = ea_prefill_latency_for_modes(
        cfg, [1, 1, 1], [100, 200, 300], theta0=1000, compute_flops=compute
    )
    assert all_nl > mixed > all_kv
