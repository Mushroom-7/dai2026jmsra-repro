import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_single_round(
    input_path: str = "results/logs/single_round.json",
    output_path: str = "results/figures/single_round.png",
) -> str:
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    names = [item["name"] for item in data["baselines"]] + ["JMSRA"]
    latencies = [
        item["latency_seconds"] for item in data["baselines"]
    ] + [data["jmsra"]["latency_seconds"]]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(
        names,
        latencies,
        color=["#6b7280", "#9ca3af", "#ef4444", "#f97316", "#2563eb"],
    )
    axis.set_ylabel("Latency (s)")
    axis.set_title("Single-round latency comparison")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return str(output)


def plot_multi_round(
    input_path: str = "results/logs/multi_round.json",
    output_path: str = "results/figures/multi_round.png",
) -> str:
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    rounds = [item["round"] for item in data["rounds"]]
    latencies = [item["latency_seconds"] for item in data["rounds"]]
    kv_counts = [sum(item["modes"]) for item in data["rounds"]]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, latency_axis = plt.subplots(figsize=(9, 4.5))
    latency_axis.plot(rounds, latencies, color="#2563eb", label="Latency")
    latency_axis.set_xlabel("Dialogue round")
    latency_axis.set_ylabel("Latency (s)")
    kv_axis = latency_axis.twinx()
    kv_axis.plot(rounds, kv_counts, color="#f97316", label="KV agents")
    kv_axis.set_ylabel("AAs using KV")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return str(output)


if __name__ == "__main__":
    plot_single_round()
    plot_multi_round()
