import json

from simulations.single_round import run_single_round


def test_single_round_writes_result_json(tmp_path):
    output = tmp_path / "single_round.json"
    result = run_single_round("configs/sim_params.yaml", str(output))
    assert output.exists()
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert "jmsra" in saved
    assert "baselines" in saved
    assert saved["jmsra"]["latency_seconds"] > 0
    assert (
        result["jmsra"]["latency_seconds"]
        == saved["jmsra"]["latency_seconds"]
    )
