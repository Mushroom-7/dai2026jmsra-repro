import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _required_mapping(mapping: dict, key: str, context: str) -> dict:
    if key not in mapping:
        raise ValueError(f"missing {context}.{key}")
    value = mapping[key]
    if not isinstance(value, dict):
        raise ValueError(f"{context}.{key} must be a mapping")
    return value


def _validate_case(
    case: dict, context: str, *, require_topology: bool = False
) -> None:
    jmsra = _required_mapping(case, "jmsra", context)
    for key in ("kv_count", "latency_seconds", "forward_path", "backward_path"):
        if key not in jmsra:
            raise ValueError(f"missing {context}.jmsra.{key}")
    for key in ("forward_path", "backward_path"):
        if not isinstance(jmsra[key], list) or not jmsra[key]:
            raise ValueError(f"{context}.jmsra.{key} must be a nonempty list")
    baselines = case.get("baselines")
    if not isinstance(baselines, list) or not baselines:
        raise ValueError(f"{context}.baselines must be a nonempty list")
    for index, baseline in enumerate(baselines):
        if not isinstance(baseline, dict):
            raise ValueError(f"{context}.baselines[{index}] must be a mapping")
        for key in ("name", "latency_seconds"):
            if key not in baseline:
                raise ValueError(
                    f"missing {context}.baselines[{index}].{key}"
                )
    if require_topology:
        topology = case.get("topology")
        if not isinstance(topology, list) or not topology:
            raise ValueError(f"{context}.topology must be a nonempty list")
        aa_count = case.get("aa_count")
        if not isinstance(aa_count, int) or aa_count <= 0:
            raise ValueError(f"{context}.aa_count must be a positive integer")
        if len(topology) != aa_count:
            raise ValueError(
                f"{context}.topology length must match {context}.aa_count"
            )
        modes = jmsra.get("modes")
        rho_values = jmsra.get("rho")
        if not isinstance(modes, list) or len(modes) != aa_count:
            raise ValueError(f"{context}.jmsra.modes length must match aa_count")
        if not isinstance(rho_values, list) or len(rho_values) != aa_count:
            raise ValueError(f"{context}.jmsra.rho length must match aa_count")
        if jmsra["kv_count"] != sum(modes):
            raise ValueError(f"{context}.jmsra.kv_count must match modes")
        for index, node in enumerate(topology):
            if not isinstance(node, dict):
                raise ValueError(
                    f"{context}.topology[{index}] must be a mapping"
                )
            for key in (
                "agent_index",
                "x_m",
                "y_m",
                "distance_m",
                "mode",
                "media",
                "rho",
            ):
                if key not in node:
                    raise ValueError(
                        f"missing {context}.topology[{index}].{key}"
                    )
            if node["agent_index"] != index:
                raise ValueError(
                    f"{context}.topology[{index}].agent_index must equal {index}"
                )
            for key in ("x_m", "y_m", "distance_m", "rho"):
                value = node[key]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError(
                        f"{context}.topology[{index}].{key} must be finite"
                    )
            if not 5 <= node["distance_m"] <= 10:
                raise ValueError(
                    f"{context}.topology[{index}].distance_m must be in [5, 10]"
                )
            if not math.isclose(
                math.hypot(node["x_m"], node["y_m"]),
                node["distance_m"],
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"{context}.topology[{index}] coordinate radius must match "
                    "distance_m"
                )
            if type(node["mode"]) is not int or node["mode"] not in (0, 1):
                raise ValueError(
                    f"{context}.topology[{index}].mode must be 0 or 1"
                )
            expected_media = "KV" if node["mode"] == 1 else "NL"
            if node["media"] != expected_media:
                raise ValueError(
                    f"{context}.topology[{index}].media must match mode"
                )
            if not 0 < node["rho"] <= 1:
                raise ValueError(
                    f"{context}.topology[{index}].rho must be in (0, 1]"
                )
            if node["mode"] != modes[index]:
                raise ValueError(
                    f"{context}.topology[{index}].mode must match jmsra.modes"
                )
            if not math.isclose(node["rho"], rho_values[index]):
                raise ValueError(
                    f"{context}.topology[{index}].rho must match jmsra.rho"
                )


def _validate_plot_data(data: dict) -> None:
    if not isinstance(data, dict):
        raise ValueError("calibration log must contain a mapping")
    metadata = _required_mapping(data, "metadata", "calibration log")
    if metadata.get("schema_version") != 2:
        raise ValueError("calibration log metadata.schema_version must be 2")
    fig6 = _required_mapping(data, "fig6", "calibration log")
    for name in ("c0_5_tflops", "c0_35_tflops"):
        case = _required_mapping(fig6, name, "fig6")
        _validate_case(case, f"fig6.{name}", require_topology=True)
    fig7 = _required_mapping(data, "fig7", "calibration log")
    for name in ("bandwidth_scan", "aa_count_scan"):
        scan = fig7.get(name)
        if not isinstance(scan, list) or not scan:
            raise ValueError(f"fig7.{name} must be a nonempty list")
        for index, case in enumerate(scan):
            if not isinstance(case, dict):
                raise ValueError(f"fig7.{name}[{index}] must be a mapping")
            _validate_case(case, f"fig7.{name}[{index}]")


def _baseline_latency(case: dict, name: str) -> float:
    for baseline in case["baselines"]:
        if baseline["name"] == name:
            return baseline["latency_seconds"]
    raise ValueError(f"missing baseline {name}")


def _plot_fig6(data: dict, output_dir: Path) -> str:
    output = output_dir / "paper_fig6_latency_paths.png"
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    cases = [
        ("C0 = 5 TFLOPS", data["fig6"]["c0_5_tflops"]),
        ("C0 = 35 TFLOPS", data["fig6"]["c0_35_tflops"]),
    ]
    for axis, (title, case) in zip(axes, cases):
        forward = case["jmsra"]["forward_path"]
        backward = case["jmsra"]["backward_path"]
        axis.plot(
            [point["kv_count"] for point in forward],
            [point["latency_seconds"] for point in forward],
            marker="o",
            label="JMSRA Forward",
        )
        axis.plot(
            [point["kv_count"] for point in backward],
            [point["latency_seconds"] for point in backward],
            marker="s",
            label="JMSRA Backward",
        )
        axis.scatter(
            [case["jmsra"]["kv_count"]],
            [case["jmsra"]["latency_seconds"]],
            color="#111827",
            label="Selected",
            zorder=3,
        )
        axis.axhline(
            _baseline_latency(case, "All-NL"),
            color="#9ca3af",
            linestyle="--",
            label="All-NL",
        )
        axis.axhline(
            _baseline_latency(case, "All-KV"),
            color="#f97316",
            linestyle="--",
            label="All-KV",
        )
        axis.set_title(title)
        axis.set_xlabel("AAs using KV")
        axis.set_ylabel("Marginal latency (s)")
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return str(output)


def _plot_fig6_topology(case: dict, title: str, output: Path) -> str:
    figure, axis = plt.subplots(figsize=(5.2, 5.2))
    topology = case["topology"]
    nl_nodes = [node for node in topology if node["mode"] == 0]
    kv_nodes = [node for node in topology if node["mode"] == 1]
    for node in topology:
        color = "#2563eb" if node["mode"] == 1 else "#f97316"
        axis.plot([0, node["x_m"]], [0, node["y_m"]], color=color, alpha=0.25)
    if nl_nodes:
        axis.scatter(
            [node["x_m"] for node in nl_nodes],
            [node["y_m"] for node in nl_nodes],
            marker="o",
            s=70,
            color="#f97316",
            edgecolor="#111827",
            linewidth=0.7,
            label="NL Mode",
        )
    if kv_nodes:
        axis.scatter(
            [node["x_m"] for node in kv_nodes],
            [node["y_m"] for node in kv_nodes],
            marker="s",
            s=70,
            color="#2563eb",
            edgecolor="#111827",
            linewidth=0.7,
            label="KV Mode",
        )
    axis.scatter(
        [0],
        [0],
        marker="*",
        s=220,
        color="#111827",
        label="EA",
        zorder=4,
    )
    axis.add_patch(
        plt.Circle((0, 0), 5, fill=False, color="#9ca3af", linestyle="--")
    )
    axis.add_patch(
        plt.Circle((0, 0), 10, fill=False, color="#9ca3af", linestyle=":")
    )
    axis.set_xlim(-10.5, 10.5)
    axis.set_ylim(-10.5, 10.5)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Distance X (m)")
    axis.set_ylabel("Distance Y (m)")
    axis.set_title(title)
    axis.grid(True, color="#e5e7eb", linewidth=0.8)
    axis.legend(loc="upper right", fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return str(output)


def _plot_scan(
    data: list[dict], x_key: str, output: Path, xlabel: str
) -> str:
    x_values = [item[x_key] for item in data]
    names = ["All-NL", "All-KV", "All-NL-Opt", "All-KV-Opt"]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for name in names:
        axis.plot(
            x_values,
            [_baseline_latency(item, name) for item in data],
            marker="o",
            label=name,
        )
    axis.plot(
        x_values,
        [item["jmsra"]["latency_seconds"] for item in data],
        marker="s",
        linewidth=2.0,
        label="JMSRA",
    )
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Marginal latency (s)")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return str(output)


def plot_paper_jmsra(
    input_path: str = "results/logs/paper_jmsra_calibration.json",
    output_dir: str = "results/figures",
) -> list[str]:
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    _validate_plot_data(data)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    return [
        _plot_fig6(data, output),
        _plot_fig6_topology(
            data["fig6"]["c0_5_tflops"],
            "C0 = 5 TFLOPS",
            output / "paper_fig6_topology_c0_5_tflops.png",
        ),
        _plot_fig6_topology(
            data["fig6"]["c0_35_tflops"],
            "C0 = 35 TFLOPS",
            output / "paper_fig6_topology_c0_35_tflops.png",
        ),
        _plot_scan(
            data["fig7"]["bandwidth_scan"],
            "bandwidth_ghz",
            output / "paper_fig7_bandwidth_scan.png",
            "System bandwidth (GHz)",
        ),
        _plot_scan(
            data["fig7"]["aa_count_scan"],
            "aa_count",
            output / "paper_fig7_aa_count_scan.png",
            "Number of AAs",
        ),
    ]


if __name__ == "__main__":
    plot_paper_jmsra()
