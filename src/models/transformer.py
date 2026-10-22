from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerConfig:
    layers: int
    heads: int
    head_dim: int
    hidden_dim: int
    ffn_dim: int
    vocab_size: int

    @property
    def k1(self) -> float:
        return 2.0 * self.layers * self.heads * self.head_dim

    @property
    def k2(self) -> float:
        return (
            8.0 * self.layers * self.heads * self.head_dim * self.hidden_dim
            + 4.0 * self.layers * self.hidden_dim * self.ffn_dim
        )

    @property
    def k3(self) -> float:
        return 2.0 * self.hidden_dim * self.vocab_size


def prefill_latency_seconds(
    config: TransformerConfig,
    input_tokens: float,
    historical_tokens: float,
    compute_flops: float,
) -> float:
    if input_tokens < 0 or historical_tokens < 0 or compute_flops <= 0:
        raise ValueError(
            "token counts must be nonnegative and compute_flops must be positive"
        )
    flops = (
        config.layers
        * (
            6.0
            * config.heads
            * config.head_dim
            * config.hidden_dim
            * input_tokens
            + 4.0
            * config.heads
            * config.head_dim
            * (input_tokens + historical_tokens)
            * input_tokens
            + (
                2.0
                * config.heads
                * config.head_dim
                * config.hidden_dim
                * input_tokens
                + 4.0
                * config.hidden_dim
                * config.ffn_dim
                * input_tokens
            )
        )
        + 2.0 * config.hidden_dim * config.vocab_size
    )
    return flops / compute_flops


def autoregressive_latency_seconds(
    config: TransformerConfig,
    output_tokens: float,
    context_tokens: float,
    compute_flops: float,
) -> float:
    if output_tokens < 0 or context_tokens < 0 or compute_flops <= 0:
        raise ValueError(
            "token counts must be nonnegative and compute_flops must be positive"
        )
    flops_per_token = (
        config.layers
        * (
            6.0 * config.heads * config.head_dim * config.hidden_dim
            + 4.0
            * config.heads
            * config.head_dim
            * (context_tokens + (1.0 + output_tokens) / 2.0)
            + (
                2.0 * config.heads * config.head_dim * config.hidden_dim
                + 4.0 * config.hidden_dim * config.ffn_dim
            )
        )
        + 2.0 * config.hidden_dim * config.vocab_size
    )
    return output_tokens * flops_per_token / compute_flops


def inference_latency_seconds(
    config: TransformerConfig,
    output_tokens: float,
    historical_tokens: float,
    input_tokens: float,
    compute_flops: float,
) -> float:
    prefill = prefill_latency_seconds(
        config, input_tokens, historical_tokens, compute_flops
    )
    autoregressive = autoregressive_latency_seconds(
        config,
        output_tokens,
        historical_tokens + input_tokens,
        compute_flops,
    )
    return prefill + autoregressive
