from src.models.transformer import (
    TransformerConfig,
    autoregressive_latency_seconds,
    inference_latency_seconds,
    prefill_latency_seconds,
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


def test_prefill_latency_increases_with_input_and_history():
    cfg = llama7b_config()
    compute = tflops_to_flops(10)
    base = prefill_latency_seconds(
        cfg, input_tokens=512, historical_tokens=1024, compute_flops=compute
    )
    longer_input = prefill_latency_seconds(
        cfg, input_tokens=1024, historical_tokens=1024, compute_flops=compute
    )
    longer_history = prefill_latency_seconds(
        cfg, input_tokens=512, historical_tokens=4096, compute_flops=compute
    )
    assert longer_input > base
    assert longer_history > base


def test_autoregressive_latency_increases_with_output_and_context():
    cfg = llama7b_config()
    compute = tflops_to_flops(10)
    base = autoregressive_latency_seconds(
        cfg, output_tokens=256, context_tokens=1024, compute_flops=compute
    )
    more_output = autoregressive_latency_seconds(
        cfg, output_tokens=512, context_tokens=1024, compute_flops=compute
    )
    more_context = autoregressive_latency_seconds(
        cfg, output_tokens=256, context_tokens=2048, compute_flops=compute
    )
    assert more_output > base
    assert more_context > base


def test_inference_latency_matches_prefill_plus_autoregressive():
    cfg = llama7b_config()
    compute = tflops_to_flops(10)
    total = inference_latency_seconds(
        cfg,
        output_tokens=512,
        historical_tokens=1024,
        input_tokens=256,
        compute_flops=compute,
    )
    expected = prefill_latency_seconds(
        cfg, 256, 1024, compute
    ) + autoregressive_latency_seconds(cfg, 512, 1280, compute)
    assert total == expected
