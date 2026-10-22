import math

import pytest

from src.analysis.multi_round import (
    DialogueState,
    accumulate_debt,
    build_experiment,
    receive_transfer,
    run_experiment,
    select_ea_broadcast_mode,
    settle_sender_debt,
)
from src.models.transformer import (
    TransformerConfig,
    autoregressive_latency_seconds,
    prefill_latency_seconds,
)
from src.utils.data_gen import load_config
from src.utils.latency import kv_cache_bits


@pytest.fixture
def llm_config():
    return TransformerConfig(32, 32, 128, 4096, 11008, 32000)


@pytest.fixture
def tiny_experiment():
    config = load_config("configs/paper_multi_round.yaml")
    paper = config["paper_multi_round"]
    paper["rounds"] = 3
    paper["max_aas"] = 2
    paper["active_aa_counts"] = [2, 1, 2]
    paper["ea_context_limit_tokens"] = 12000
    paper["ea_kv_debt_window_tokens"] = 6000
    paper["aa_context_limit_tokens"] = 5000
    paper["aa_kv_debt_window_tokens"] = 2000
    paper["trend"]["early_round_end"] = 1
    paper["trend"]["deep_round_start"] = 2
    paper["trend"]["deep_round_end"] = 3
    return build_experiment(config)


def test_nl_reception_adds_only_alpha_and_does_not_clear_debt():
    update = receive_transfer(
        context_tokens=1000,
        debt_tokens=3000,
        alpha_tokens=1024,
        mode=0,
        context_limit_tokens=5000,
    )
    assert update.context_tokens == 2024
    assert update.debt_tokens == 3000
    assert update.transferred_context_tokens == 1024


def test_kv_reception_adds_only_new_alpha_then_clears_debt():
    update = receive_transfer(
        context_tokens=1000,
        debt_tokens=3000,
        alpha_tokens=1024,
        mode=1,
        context_limit_tokens=5000,
    )
    assert update.context_tokens == 2024
    assert update.debt_tokens == 0
    assert update.transferred_context_tokens == 1024


def test_accumulated_debt_is_window_capped():
    assert accumulate_debt(9000, 2048, 10240) == 10240


def test_nl_carries_current_generation_into_future_kv_debt():
    assert settle_sender_debt(3000, 1024, 0, 3500) == 3500


def test_kv_clears_debt_including_current_generation():
    assert settle_sender_debt(3000, 1024, 1, 3500) == 0


@pytest.mark.parametrize("mode", [-1, 2])
def test_receive_transfer_rejects_invalid_modes(mode):
    with pytest.raises(ValueError, match="mode"):
        receive_transfer(0, 0, 1024, mode, 4096)


@pytest.mark.parametrize(
    ("current", "increment", "window"),
    [(-1, 1, 10), (1, -1, 10), (1, 1, 0), (math.inf, 1, 10)],
)
def test_accumulate_debt_rejects_invalid_values(current, increment, window):
    with pytest.raises(ValueError):
        accumulate_debt(current, increment, window)


def test_dialogue_state_requires_matching_nonnegative_aa_vectors():
    with pytest.raises(ValueError, match="same length"):
        DialogueState(0, 0, (0, 0), (0,))
    with pytest.raises(ValueError, match="nonnegative"):
        DialogueState(0, 0, (0, -1), (0, 0))


def test_dialogue_state_is_immutable():
    state = DialogueState(1024, 1024, (0,), (0,))
    with pytest.raises(AttributeError):
        state.ea_context_tokens = 0


def test_broadcast_decision_matches_direct_min_max_cost(llm_config):
    decision = select_ea_broadcast_mode(
        config=llm_config,
        alpha_tokens=1024,
        debt_tokens=1024,
        receiver_context_tokens=[0, 2000],
        receiver_compute_flops=[5e12, 15e12],
        rate_bps=8e9,
        bits_per_token=16,
        compression_ratio=2,
    )
    expected_nl = max(
        prefill_latency_seconds(llm_config, 1024, theta, compute)
        + 16 * 1024 / 8e9
        for theta, compute in zip([0, 2000], [5e12, 15e12])
    )
    expected_kv = kv_cache_bits(llm_config, 1024, 1024, 2) / 8e9
    assert decision.nl_latency_seconds == pytest.approx(expected_nl)
    assert decision.kv_latency_seconds == pytest.approx(expected_kv)
    assert decision.mode == int(expected_kv < expected_nl)


@pytest.mark.parametrize(
    ("contexts", "computes", "message"),
    [([], [], "nonempty"), ([0], [5e12, 6e12], "same length")],
)
def test_broadcast_decision_rejects_invalid_receivers(
    llm_config, contexts, computes, message
):
    with pytest.raises(ValueError, match=message):
        select_ea_broadcast_mode(
            llm_config,
            1024,
            1024,
            contexts,
            computes,
            8e9,
            16,
            2,
        )


def test_broadcast_decision_can_choose_each_mode(llm_config):
    nl = select_ea_broadcast_mode(
        llm_config, 1024, 10240, [0], [15e12], 1e8, 16, 2
    )
    kv = select_ea_broadcast_mode(
        llm_config, 1024, 0, [100000], [5e12], 1e12, 16, 2
    )
    assert nl.mode == 0
    assert kv.mode == 1


def test_round_engine_applies_jmsra_modes_and_state_limits(tiny_experiment):
    result = run_experiment(tiny_experiment)
    assert len(result.rounds) == 3
    assert all(
        sum(item.aa_bandwidth) <= 1.0 + 1e-6 for item in result.rounds
    )
    assert all(
        value <= 5000
        for item in result.rounds
        for value in item.state_after.aa_context_tokens
    )
    assert all(
        value <= 2000
        for item in result.rounds
        for value in item.state_after.aa_kv_debt_tokens
    )
    assert all(
        item.state_after.ea_context_tokens <= 12000
        for item in result.rounds
    )


def test_waiting_agent_state_is_unchanged(tiny_experiment):
    result = run_experiment(tiny_experiment)
    before = result.rounds[1].state_before
    after = result.rounds[1].state_after
    waiting_index = 1
    assert (
        after.aa_context_tokens[waiting_index]
        == before.aa_context_tokens[waiting_index]
    )
    assert (
        after.aa_kv_debt_tokens[waiting_index]
        == before.aa_kv_debt_tokens[waiting_index]
    )


def test_active_agent_context_includes_local_input_and_output(
    tiny_experiment,
):
    first = run_experiment(tiny_experiment).rounds[0]
    assert first.state_after.aa_context_tokens[0] >= (
        first.state_before.aa_context_tokens[0]
        + tiny_experiment.aa_sensory_input_tokens
        + tiny_experiment.aa_generated_tokens
    )


def test_ea_compute_breakdown_uses_transformer_primitives(tiny_experiment):
    first = run_experiment(tiny_experiment).rounds[0]
    nl_tokens = sum(
        tiny_experiment.aa_generated_tokens
        for mode in first.aa_modes
        if mode == 0
    )
    expected_prefill = prefill_latency_seconds(
        tiny_experiment.config,
        nl_tokens,
        first.state_before.ea_context_tokens,
        tiny_experiment.ea_compute_flops,
    )
    expected_autoregressive = autoregressive_latency_seconds(
        tiny_experiment.config,
        tiny_experiment.ea_generated_tokens,
        first.ea_autoregressive_context_tokens,
        tiny_experiment.ea_compute_flops,
    )
    assert first.ours_ea_prefill_seconds == pytest.approx(expected_prefill)
    assert first.ours_ea_autoregressive_seconds == pytest.approx(
        expected_autoregressive
    )
    assert first.ours_ea_total_seconds == pytest.approx(
        expected_prefill + expected_autoregressive
    )


def test_ea_generated_output_enters_next_round_context(tiny_experiment):
    first = run_experiment(tiny_experiment).rounds[0]
    expected = min(
        first.state_before.ea_context_tokens
        + len(first.active_agents) * tiny_experiment.aa_generated_tokens
        + tiny_experiment.ea_generated_tokens,
        tiny_experiment.ea_context_limit_tokens,
    )
    assert first.state_after.ea_context_tokens == expected


def test_experiment_is_deterministic(tiny_experiment):
    assert run_experiment(tiny_experiment) == run_experiment(tiny_experiment)
