import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.baselines.all_kv import run_all_kv
from src.algorithms.baselines.all_kv_opt import run_all_kv_opt
from src.algorithms.baselines.all_nl import run_all_nl
from src.algorithms.jmsra import solve_jmsra
from src.utils.data_gen import load_config, make_single_round_problem


def _baseline_to_dict(result):
    return {
        "name": result.name,
        "modes": result.modes,
        "rho": result.rho,
        "latency_seconds": result.latency_seconds,
    }


def run_single_round(
    config_path: str = "configs/sim_params.yaml",
    output_path: str = "results/logs/single_round.json",
) -> dict:
    config = load_config(config_path)
    problem = make_single_round_problem(config)
    jmsra = solve_jmsra(problem)
    baselines = [
        run_all_nl(problem),
        run_all_nl(problem, optimized=True),
        run_all_kv(problem),
        run_all_kv_opt(problem),
    ]
    result = {
        "jmsra": {
            "modes": jmsra.modes,
            "rho": jmsra.rho,
            "latency_seconds": jmsra.latency_seconds,
            "transmission_latency_seconds": (
                jmsra.transmission_latency_seconds
            ),
        },
        "baselines": [_baseline_to_dict(item) for item in baselines],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    run_single_round()
