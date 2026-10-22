import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


_METRIC_KEYS = (
    "ours_ea_prefill_seconds",
    "ours_ea_autoregressive_seconds",
    "ours_ea_total_seconds",
    "pure_nl_ea_total_seconds",
)


def _finite_number(value, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _load_result(input_path: str) -> dict:
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("unsupported schema_version")
    rounds = data.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise ValueError("rounds must be a nonempty list")
    for offset, item in enumerate(rounds, start=1):
        if not isinstance(item, dict) or item.get("round") != offset:
            raise ValueError("round entries must be sequential mappings")
        for key in _METRIC_KEYS:
            _finite_number(item.get(key), f"rounds[{offset - 1}].{key}")
        state = item.get("state_after")
        if not isinstance(state, dict):
            raise ValueError(f"rounds[{offset - 1}].state_after is required")
        _finite_number(
            state.get("ea_context_tokens"),
            f"rounds[{offset - 1}].state_after.ea_context_tokens",
        )

    schedule = data.get("schedule")
    if not isinstance(schedule, list) or len(schedule) != 21:
        raise ValueError("schedule must contain 21 rows")
    for row_index, row in enumerate(schedule):
        if not isinstance(row, list) or len(row) != len(rounds):
            raise ValueError("schedule row lengths must match rounds")
        for value in row:
            if isinstance(value, bool) or value not in (-1, 0, 1):
                raise ValueError(
                    f"schedule[{row_index}] contains an invalid mode"
                )
    return data


def plot_paper_multi_round(
    input_path: str = "results/logs/paper_multi_round.json",
    output_path: str = "results/figures/paper_fig8_multi_round.png",
) -> str:
    data = _load_result(input_path)
    rounds = data["rounds"]
    round_numbers = np.asarray([item["round"] for item in rounds])
    schedule = np.asarray(data["schedule"], dtype=int)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, (schedule_axis, latency_axis) = plt.subplots(
        1,
        2,
        figsize=(13.5, 5.2),
        gridspec_kw={"width_ratios": [1.0, 1.18]},
    )
    colors = ["#f3f4f6", "#12aeb5", "#e4571e"]
    color_map = ListedColormap(colors)
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], color_map.N)
    schedule_axis.imshow(
        schedule,
        cmap=color_map,
        norm=norm,
        interpolation="nearest",
        aspect="auto",
        extent=(0.5, len(rounds) + 0.5, 20.5, -0.5),
    )
    schedule_axis.set_xlabel("Dialogue Rounds")
    schedule_axis.set_ylabel("Agent Index")
    schedule_axis.set_xticks([2, 6, 10, 14, 18, 22, 26, 30, 34, 38])
    schedule_axis.set_yticks([0, 5, 10, 15, 20])
    schedule_axis.grid(False)
    schedule_axis.legend(
        handles=[
            Patch(facecolor=colors[1], label="NL Mode"),
            Patch(facecolor=colors[2], label="KV Mode"),
            Patch(facecolor=colors[0], edgecolor="#9ca3af", label="Waiting"),
        ],
        loc="lower left",
        frameon=True,
        framealpha=1.0,
        edgecolor="#4b5563",
        fontsize=9,
    )

    pure_nl = [item["pure_nl_ea_total_seconds"] for item in rounds]
    ours_prefill = [item["ours_ea_prefill_seconds"] for item in rounds]
    ours_ar = [item["ours_ea_autoregressive_seconds"] for item in rounds]
    ours_total = [item["ours_ea_total_seconds"] for item in rounds]
    contexts_k = [
        item["state_after"]["ea_context_tokens"] / 1000.0
        for item in rounds
    ]
    marker_step = max(1, len(rounds) // 10)
    latency_axis.plot(
        round_numbers,
        pure_nl,
        color="#858585",
        linestyle="--",
        linewidth=2.0,
        marker="D",
        markevery=marker_step,
        markersize=4,
        label="Pure-NL: EA Total Latency",
    )
    latency_axis.plot(
        round_numbers,
        ours_prefill,
        color="#2996c7",
        linewidth=1.8,
        marker="^",
        markevery=marker_step,
        markersize=4,
        label="Ours: EA Prefill",
    )
    latency_axis.plot(
        round_numbers,
        ours_ar,
        color="#dc4a3d",
        linewidth=1.8,
        marker="o",
        markerfacecolor="none",
        markevery=marker_step,
        markersize=4,
        label="Ours: EA Autoregressive",
    )
    latency_axis.plot(
        round_numbers,
        ours_total,
        color="#111827",
        linewidth=2.0,
        marker="s",
        markevery=marker_step,
        markersize=4,
        label="Ours: EA Total Latency",
    )
    latency_axis.set_xlabel("Dialogue Rounds")
    latency_axis.set_ylabel("Computation Latency (s)")
    latency_axis.set_xlim(1, len(rounds))
    latency_axis.set_xticks([1, 5, 9, 13, 17, 21, 25, 29, 33, 37])
    latency_axis.set_ylim(bottom=0)
    latency_axis.grid(True, color="#d1d5db", linewidth=0.7, alpha=0.7)

    context_axis = latency_axis.twinx()
    context_axis.plot(
        round_numbers,
        contexts_k,
        color="#74aa35",
        linestyle=":",
        linewidth=2.0,
        marker="D",
        markevery=marker_step,
        markersize=3.5,
        label="EA Context Length",
    )
    context_axis.set_ylabel("Context Length (K)", color="#74aa35")
    context_axis.tick_params(axis="y", colors="#74aa35")
    context_axis.spines["right"].set_color("#74aa35")
    context_axis.set_ylim(bottom=0)

    lines = latency_axis.get_lines() + context_axis.get_lines()
    latency_axis.legend(
        lines,
        [line.get_label() for line in lines],
        loc="upper left",
        bbox_to_anchor=(0.02, 0.96),
        frameon=True,
        framealpha=0.95,
        edgecolor="#4b5563",
        fontsize=8.5,
    )
    schedule_axis.text(
        0.5,
        -0.17,
        "(a)",
        transform=schedule_axis.transAxes,
        ha="center",
        va="top",
    )
    latency_axis.text(
        0.5,
        -0.17,
        "(b)",
        transform=latency_axis.transAxes,
        ha="center",
        va="top",
    )
    figure.subplots_adjust(
        left=0.07, right=0.94, bottom=0.18, top=0.97, wspace=0.23
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return str(output)


if __name__ == "__main__":
    plot_paper_multi_round()
