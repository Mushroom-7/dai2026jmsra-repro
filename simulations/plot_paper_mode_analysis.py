import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _required_mapping(mapping: dict, key: str, context: str) -> dict:
    if key not in mapping:
        raise ValueError(f"missing {context}.{key}")
    value = mapping[key]
    if not isinstance(value, dict):
        raise ValueError(f"{context}.{key} must be a mapping")
    return value


def _finite_values(values, context: str) -> list[float]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{context} must be a nonempty list")
    parsed = []
    for index, value in enumerate(values):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{context}[{index}] must be finite")
        parsed.append(float(value))
    return parsed


def _finite_field(mapping: dict, key: str, context: str) -> float:
    if key not in mapping:
        raise ValueError(f"missing {context}.{key}")
    value = mapping[key]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{context}.{key} must be finite")
    return float(value)


def _validate_line_panel(
    panel: dict,
    context: str,
    value_key: str,
    *,
    numeric_fields: tuple[str, ...],
    system_field: bool = False,
) -> None:
    x_values = _finite_values(panel.get("x_values"), f"{context}.x_values")
    series = panel.get("series")
    if not isinstance(series, list) or not series:
        raise ValueError(f"{context}.series must be a nonempty list")
    for index, item in enumerate(series):
        item_context = f"{context}.series[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{item_context} must be a mapping")
        values = _finite_values(item.get(value_key), f"{item_context}.{value_key}")
        if len(values) != len(x_values):
            raise ValueError(
                f"{item_context}.{value_key} length must match x_values"
            )
        for key in numeric_fields:
            _finite_field(item, key, item_context)
        if system_field and item.get("system") not in {
            "fixed_ea",
            "homogeneous",
        }:
            raise ValueError(
                f"{item_context}.system must be 'fixed_ea' or 'homogeneous'"
            )


def _validate_threshold_panel(panel: dict) -> None:
    alpha_values = _finite_values(
        panel.get("alpha_values"), "fig5.panel_c.alpha_values"
    )
    xi_values = _finite_values(
        panel.get("xi_values"), "fig5.panel_c.xi_values"
    )
    surfaces = panel.get("surfaces")
    if not isinstance(surfaces, list) or len(surfaces) != 2:
        raise ValueError("fig5.panel_c.surfaces must contain exactly 2 items")
    theta_values = []
    for surface_index, surface in enumerate(surfaces):
        context = f"fig5.panel_c.surfaces[{surface_index}]"
        if not isinstance(surface, dict):
            raise ValueError(f"{context} must be a mapping")
        theta_values.append(_finite_field(surface, "theta", context))
        thresholds = surface.get("thresholds")
        if not isinstance(thresholds, list) or len(thresholds) != len(xi_values):
            raise ValueError(f"{context}.thresholds length must match xi_values")
        for row_index, row in enumerate(thresholds):
            if not isinstance(row, list) or len(row) != len(alpha_values):
                raise ValueError(
                    f"{context}.thresholds row length must match alpha_values"
                )
            for column_index, value in enumerate(row):
                if value is None:
                    continue
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not 0 < value <= 1
                ):
                    raise ValueError(
                        f"{context}.thresholds[{row_index}]"
                        f"[{column_index}] must be null or in (0, 1]"
                    )
    if len(set(theta_values)) != len(theta_values):
        raise ValueError("fig5.panel_c.surfaces theta values must be distinct")


def _validate_plot_data(data: dict) -> None:
    if not isinstance(data, dict):
        raise ValueError("analysis log must contain a mapping")
    metadata = _required_mapping(data, "metadata", "analysis log")
    if metadata.get("schema_version") != 1:
        raise ValueError("analysis log metadata.schema_version must be 1")

    fig4 = _required_mapping(data, "fig4", "analysis log")
    fig4_fields = {
        "panel_a": (("snr_db", "input_tokens"), False),
        "panel_b": (("snr_db",), True),
        "panel_c": (("aa_count", "input_tokens"), False),
        "panel_d": (("snr_db", "input_tokens"), False),
    }
    for name, (numeric_fields, system_field) in fig4_fields.items():
        panel = _required_mapping(fig4, name, "fig4")
        _validate_line_panel(
            panel,
            f"fig4.{name}",
            "ratios",
            numeric_fields=numeric_fields,
            system_field=system_field,
        )

    fig5 = _required_mapping(data, "fig5", "analysis log")
    fig5_fields = {
        "panel_a": ("xi", "theta"),
        "panel_b": ("snr_db", "theta"),
    }
    for name, numeric_fields in fig5_fields.items():
        panel = _required_mapping(fig5, name, "fig5")
        _validate_line_panel(
            panel,
            f"fig5.{name}",
            "scores",
            numeric_fields=numeric_fields,
        )
    panel_c = _required_mapping(fig5, "panel_c", "fig5")
    _validate_threshold_panel(panel_c)


def _line_style(series: dict) -> dict:
    alternate = series.get("input_tokens") == 2048 or (
        series.get("system") == "homogeneous"
    )
    return {
        "marker": "^" if alternate else "o",
        "markevery": 4,
        "markersize": 3.5,
        "linewidth": 1.5,
        "linestyle": "--" if alternate else "-",
    }


def _fig4_label(panel_name: str, series: dict) -> str:
    if panel_name == "panel_a":
        return f"SNR={series['snr_db']:g} dB, s={series['input_tokens']:g}"
    if panel_name == "panel_b":
        system = "Fixed EA" if series["system"] == "fixed_ea" else "Homogeneous"
        return f"{system}, SNR={series['snr_db']:g} dB"
    if panel_name == "panel_c":
        return f"I={series['aa_count']}, s={series['input_tokens']:g}"
    return f"SNR={series['snr_db']:g} dB, s={series['input_tokens']:g}"


def _plot_fig4(data: dict, output_dir: Path) -> str:
    output = output_dir / "paper_fig4_latency_ratio.png"
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.8))
    panel_settings = {
        "panel_a": (axes[0, 0], "Output-input ratio", "(a)"),
        "panel_b": (axes[0, 1], "Computation capability of AA (TFLOPS)", "(b)"),
        "panel_c": (axes[1, 0], "SNR (dB)", "(c)"),
        "panel_d": (axes[1, 1], "Number of AAs", "(d)"),
    }
    colors = ("#E4572E", "#7B2CBF", "#4D9221", "#2A6FBB", "#B07C17", "#008F7A")

    for panel_name, (axis, xlabel, letter) in panel_settings.items():
        panel = data["fig4"][panel_name]
        x_values = panel["x_values"]
        color_field = "aa_count" if panel_name == "panel_c" else "snr_db"
        color_values = sorted(
            {series[color_field] for series in panel["series"]}
        )
        color_map = {
            value: colors[index % len(colors)]
            for index, value in enumerate(color_values)
        }
        all_ratios = [
            value for series in panel["series"] for value in series["ratios"]
        ]
        upper = max(all_ratios) * 1.08
        lower = max(0.0, min(all_ratios) * 0.88)
        axis.axhspan(lower, 1, color="#FDE2E2", alpha=0.7, zorder=0)
        axis.axhspan(1, upper, color="#E5F3E1", alpha=0.7, zorder=0)
        for series in panel["series"]:
            axis.plot(
                x_values,
                series["ratios"],
                color=color_map[series[color_field]],
                label=_fig4_label(panel_name, series),
                **_line_style(series),
            )
        axis.axhline(1, color="#374151", linestyle=(0, (4, 3)), linewidth=1.1)
        axis.set_ylim(lower, upper)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(r"Total latency ratio $t_{NL}/t_{KV}$")
        axis.grid(True, color="#D1D5DB", linewidth=0.6, alpha=0.8)
        axis.legend(fontsize=7.2, loc="best", framealpha=0.92)
        axis.text(
            0.5,
            -0.22,
            letter,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=10,
        )
        if panel_name == "panel_a":
            axis.set_xscale("log")

    figure.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.97,
        bottom=0.09,
        hspace=0.38,
        wspace=0.22,
    )
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return str(output)


def _fig5_curve_label(panel_name: str, series: dict) -> str:
    variable = (
        f"xi={series['xi']:g}"
        if panel_name == "panel_a"
        else f"SNR={series['snr_db']:g} dB"
    )
    return f"{variable}, theta={series['theta']:g}"


def _plot_fig5(data: dict, output_dir: Path) -> str:
    output = output_dir / "paper_fig5_single_link_selection.png"
    figure = plt.figure(figsize=(15.2, 4.9))
    grid = figure.add_gridspec(1, 3, width_ratios=(1, 1, 1.12), wspace=0.3)
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    colors = ("#E4572E", "#4D9221", "#1677C8")

    for panel_name, axis, letter in zip(
        ("panel_a", "panel_b"), axes, ("(a)", "(b)")
    ):
        panel = data["fig5"][panel_name]
        for index, series in enumerate(panel["series"]):
            color_index = index % 3
            axis.plot(
                panel["x_values"],
                series["scores"],
                color=colors[color_index],
                linestyle="--" if series["theta"] > 0 else "-",
                linewidth=1.8,
                label=_fig5_curve_label(panel_name, series),
            )
        axis.axhspan(-10, 0, color="#FDE2E2", alpha=0.7, zorder=0)
        axis.axhspan(0, 10, color="#E5F3E1", alpha=0.7, zorder=0)
        axis.axhline(0, color="#374151", linestyle=(0, (4, 3)), linewidth=1.1)
        axis.set_xlim(0, 1)
        # The published Fig. 5 clips the rho -> 0 tail to keep crossings legible.
        axis.set_ylim(-10, 10)
        axis.set_xlabel(r"Bandwidth allocation coefficient $\rho$")
        axis.set_ylabel(r"Mode selection function $f(\rho)$ (s)")
        axis.grid(True, color="#D1D5DB", linewidth=0.6, alpha=0.8)
        axis.legend(fontsize=7.2, loc="lower right", framealpha=0.92)
        axis.set_title(letter, fontsize=10, pad=8)

    panel_c = data["fig5"]["panel_c"]
    axis_3d = figure.add_subplot(grid[0, 2], projection="3d")
    alpha_grid, xi_grid = np.meshgrid(
        panel_c["alpha_values"], panel_c["xi_values"]
    )
    cmaps = ("Blues", "YlOrBr")
    legend_handles = []
    for surface, cmap in zip(panel_c["surfaces"], cmaps):
        thresholds = np.asarray(
            [
                [np.nan if value is None else value for value in row]
                for row in surface["thresholds"]
            ],
            dtype=float,
        )
        axis_3d.plot_surface(
            alpha_grid,
            xi_grid,
            thresholds,
            cmap=cmap,
            alpha=0.78,
            linewidth=0,
            antialiased=True,
        )
        legend_handles.append(
            Patch(
                facecolor=plt.get_cmap(cmap)(0.65),
                label=rf"$\theta={surface['theta']:g}$",
                alpha=0.78,
            )
        )
    axis_3d.set_xlabel("Generated tokens", labelpad=8)
    axis_3d.set_ylabel("Unshared length", labelpad=8)
    axis_3d.set_zlabel(r"Threshold $\rho^*$", labelpad=7)
    axis_3d.set_zlim(0, 1)
    axis_3d.view_init(elev=27, azim=-132)
    axis_3d.set_title("(c)", fontsize=10, pad=8)
    axis_3d.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=8,
    )

    figure.subplots_adjust(left=0.055, right=0.98, top=0.92, bottom=0.15)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return str(output)


def plot_paper_mode_analysis(
    input_path: str = "results/logs/paper_mode_analysis.json",
    output_dir: str = "results/figures",
) -> list[str]:
    input_file = Path(input_path)
    if not input_file.is_absolute():
        input_file = PROJECT_ROOT / input_file
    data = json.loads(input_file.read_text(encoding="utf-8"))
    _validate_plot_data(data)

    output = Path(output_dir)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    return [_plot_fig4(data, output), _plot_fig5(data, output)]


if __name__ == "__main__":
    plot_paper_mode_analysis()
