from src.models.transformer import TransformerConfig, prefill_latency_seconds


def token_bits(alpha: float, bits_per_token: float) -> float:
    if alpha < 0 or bits_per_token <= 0:
        raise ValueError(
            "alpha must be nonnegative and bits_per_token must be positive"
        )
    return bits_per_token * alpha


def kv_cache_bits(
    config: TransformerConfig,
    alpha: float,
    unshared_context: float,
    compression_ratio: float,
) -> float:
    if alpha < 0 or unshared_context < 0 or compression_ratio <= 0:
        raise ValueError(
            "token counts must be nonnegative and compression_ratio must be positive"
        )
    return (
        32.0
        * config.layers
        * config.heads
        * config.head_dim
        * (unshared_context + alpha)
        / compression_ratio
    )


def transmission_latency_seconds(bits: float, rate_bps: float) -> float:
    if bits < 0 or rate_bps <= 0:
        raise ValueError("bits must be nonnegative and rate_bps must be positive")
    return bits / rate_bps


def ea_prefill_latency_for_modes(
    config: TransformerConfig,
    modes: list[int],
    alphas: list[float],
    theta0: float,
    compute_flops: float,
) -> float:
    if len(modes) != len(alphas):
        raise ValueError("modes and alphas must have the same length")
    nl_input_tokens = sum(
        alpha for mode, alpha in zip(modes, alphas) if mode == 0
    )
    return prefill_latency_seconds(
        config, nl_input_tokens, theta0, compute_flops
    )
