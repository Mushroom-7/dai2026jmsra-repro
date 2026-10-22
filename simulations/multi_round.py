import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.jmsra import JmsraProblem, solve_jmsra
from src.channels.wireless_channel import channel_gain
from src.utils.data_gen import load_config, transformer_config
from src.utils.units import (
    dbm_to_watt,
    ghz_to_hz,
    noise_dbm_per_hz_to_watt_per_hz,
    tflops_to_flops,
)


def run_multi_round(
    config_path: str = "configs/sim_params.yaml",
    output_path: str = "results/logs/multi_round.json",
) -> dict:
    config = load_config(config_path)
    rng = np.random.default_rng(config["random_seed"] + 1)
    multi_round = config["multi_round"]
    communication = config["communication"]
    max_aas = int(multi_round["max_aas"])
    aa_kv_debt = np.zeros(max_aas)
    theta0 = float(multi_round["ea_initial_prompt_tokens"])
    rounds = []
    for round_index in range(1, int(multi_round["rounds"]) + 1):
        active_count = int(
            rng.integers(max(2, max_aas // 4), max_aas + 1)
        )
        active = sorted(
            rng.choice(max_aas, size=active_count, replace=False).tolist()
        )
        alphas = [float(multi_round["aa_generated_tokens"])] * active_count
        aa_kv_debt[active] = np.minimum(
            aa_kv_debt[active]
            + float(multi_round["aa_sensory_input_tokens"]),
            float(multi_round["aa_kv_debt_window_tokens"]),
        )
        xis = [float(aa_kv_debt[index]) for index in active]
        distances = rng.uniform(5.0, 10.0, size=active_count)
        fading = rng.exponential(scale=1.0, size=active_count)
        powers = rng.uniform(
            float(config["single_round"]["aa_transmit_power_dbm_min"]),
            float(config["single_round"]["aa_transmit_power_dbm_max"]),
            size=active_count,
        )
        problem = JmsraProblem(
            config=transformer_config(config),
            alphas=alphas,
            xis=xis,
            gammas=[
                float(communication["kv_cache_compression_ratio"])
            ]
            * active_count,
            powers_watt=[
                dbm_to_watt(float(value)) for value in powers
            ],
            gains=[
                channel_gain(float(distance), float(power))
                for distance, power in zip(distances, fading)
            ],
            bandwidth_hz=ghz_to_hz(
                float(communication["bandwidth_ghz"])
            ),
            noise_psd_w_per_hz=noise_dbm_per_hz_to_watt_per_hz(
                float(communication["noise_dbm_per_hz"])
            ),
            bits_per_token=float(communication["bits_per_token"]),
            ea_compute_flops=tflops_to_flops(
                float(multi_round["ea_compute_tflops"])
            ),
            theta0=theta0,
            tolerance_seconds=float(
                config["jmsra"]["tolerance_seconds"]
            ),
            min_rho=float(config["jmsra"]["min_rho"]),
        )
        result = solve_jmsra(problem)
        for local_index, mode in enumerate(result.modes):
            if mode == 1:
                aa_kv_debt[active[local_index]] = 0.0
        theta0 = min(
            theta0
            + sum(
                alpha
                for alpha, mode in zip(alphas, result.modes)
                if mode == 0
            ),
            float(multi_round["ea_context_limit_tokens"]),
        )
        rounds.append(
            {
                "round": round_index,
                "active_agents": active,
                "modes": result.modes,
                "rho": result.rho,
                "latency_seconds": result.latency_seconds,
                "theta0": theta0,
            }
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {"rounds": rounds}
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


if __name__ == "__main__":
    run_multi_round()
