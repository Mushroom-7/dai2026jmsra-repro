import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.media_selection import (
    StaticE2EScenario,
    bandwidth_threshold,
    single_link_mode_score,
    static_latency_ratio,
)
from src.utils.data_gen import (
    load_config,
    transformer_config,
    validate_paper_mode_analysis_config,
)
from src.utils.units import ghz_to_hz, tflops_to_flops


def _as_floats(values) -> list[float]:
    return [float(value) for value in values]


def _strictly_increasing(values: list[float]) -> bool:
    return all(left < right for left, right in zip(values, values[1:]))


def _strictly_decreasing(values: list[float]) -> bool:
    return all(left > right for left, right in zip(values, values[1:]))


def _has_crossover(values: list[float], boundary: float = 1.0) -> bool:
    return min(values) < boundary < max(values)


def _is_nonmonotonic(values: list[float]) -> bool:
    differences = [right - left for left, right in zip(values, values[1:])]
    return any(value > 0 for value in differences) and any(
        value < 0 for value in differences
    )


def _has_single_valley(values: list[float]) -> bool:
    if len(values) < 3:
        return False
    minimum_index = values.index(min(values))
    if minimum_index == 0 or minimum_index == len(values) - 1:
        return False
    return _strictly_decreasing(values[: minimum_index + 1]) and (
        _strictly_increasing(values[minimum_index:])
    )


def _scenario(config: dict, **overrides) -> StaticE2EScenario:
    analysis = config["paper_mode_analysis"]
    fig4 = analysis["fig4"]
    values = {
        "config": transformer_config(config),
        "aa_count": int(fig4["default_aa_count"]),
        "input_tokens": 512.0,
        "beta": float(fig4["default_beta"]),
        "final_output_tokens": float(analysis["final_ea_output_tokens"]),
        "ea_compute_flops": tflops_to_flops(
            float(fig4["default_ea_compute_tflops"])
        ),
        "aa_compute_flops": tflops_to_flops(
            float(fig4["default_aa_compute_tflops"])
        ),
        "bandwidth_hz": ghz_to_hz(float(analysis["bandwidth_ghz"])),
        "snr_db": 10.0,
        "bits_per_token": float(analysis["bits_per_token"]),
        "compression_ratio": float(
            analysis["kv_cache_compression_ratio"]
        ),
    }
    values.update(overrides)
    return StaticE2EScenario(**values)


def _ratio_series(config: dict, x_values, scenario_factory, **parameters) -> dict:
    ratios = [
        float(static_latency_ratio(scenario_factory(float(value))))
        for value in x_values
    ]
    return {
        **parameters,
        "ratios": ratios,
        "crosses_boundary": _has_crossover(ratios),
    }


def _fig4_panels(config: dict) -> dict:
    fig4 = config["paper_mode_analysis"]["fig4"]

    beta_values = _as_floats(
        np.geomspace(
            fig4["panel_a_beta_min"],
            fig4["panel_a_beta_max"],
            fig4["panel_a_beta_points"],
        )
    )
    panel_a_series = []
    for input_tokens in fig4["panel_a_input_tokens_values"]:
        for snr_db in fig4["panel_a_snr_db_values"]:
            panel_a_series.append(
                _ratio_series(
                    config,
                    beta_values,
                    lambda beta, s=float(input_tokens), snr=float(snr_db): (
                        _scenario(
                            config,
                            beta=beta,
                            input_tokens=s,
                            snr_db=snr,
                        )
                    ),
                    input_tokens=float(input_tokens),
                    snr_db=float(snr_db),
                )
            )
    panel_a = {
        "x_name": "beta",
        "x_values": beta_values,
        "series": panel_a_series,
        "has_crossover": any(
            series["crosses_boundary"] for series in panel_a_series
        ),
    }

    compute_values = _as_floats(
        np.linspace(
            fig4["panel_b_compute_min_tflops"],
            fig4["panel_b_compute_max_tflops"],
            fig4["panel_b_compute_points"],
        )
    )
    panel_b_series = []
    for snr_db in fig4["panel_b_snr_db_values"]:
        panel_b_series.append(
            _ratio_series(
                config,
                compute_values,
                lambda compute, snr=float(snr_db): _scenario(
                    config,
                    input_tokens=float(fig4["panel_b_input_tokens"]),
                    snr_db=snr,
                    aa_compute_flops=tflops_to_flops(compute),
                ),
                system="fixed_ea",
                ea_compute_tflops=float(fig4["default_ea_compute_tflops"]),
                snr_db=float(snr_db),
            )
        )
        panel_b_series.append(
            _ratio_series(
                config,
                compute_values,
                lambda compute, snr=float(snr_db): _scenario(
                    config,
                    input_tokens=float(fig4["panel_b_input_tokens"]),
                    snr_db=snr,
                    ea_compute_flops=tflops_to_flops(compute),
                    aa_compute_flops=tflops_to_flops(compute),
                ),
                system="homogeneous",
                snr_db=float(snr_db),
            )
        )
    panel_b = {
        "x_name": "compute_tflops",
        "x_values": compute_values,
        "series": panel_b_series,
        "has_crossover": any(
            series["crosses_boundary"] for series in panel_b_series
        ),
    }

    snr_values = _as_floats(
        np.linspace(
            fig4["panel_c_snr_min_db"],
            fig4["panel_c_snr_max_db"],
            fig4["panel_c_snr_points"],
        )
    )
    panel_c_series = []
    for input_tokens in fig4["panel_c_input_tokens_values"]:
        for aa_count in fig4["panel_c_aa_count_values"]:
            panel_c_series.append(
                _ratio_series(
                    config,
                    snr_values,
                    lambda snr, s=float(input_tokens), count=int(aa_count): (
                        _scenario(
                            config,
                            input_tokens=s,
                            aa_count=count,
                            snr_db=snr,
                        )
                    ),
                    input_tokens=float(input_tokens),
                    aa_count=int(aa_count),
                )
            )
    panel_c = {
        "x_name": "snr_db",
        "x_values": snr_values,
        "series": panel_c_series,
        "has_crossover": any(
            series["crosses_boundary"] for series in panel_c_series
        ),
    }

    aa_count_values = [int(value) for value in fig4["panel_d_aa_count_values"]]
    panel_d_series = []
    for input_tokens in fig4["panel_d_input_tokens_values"]:
        for snr_db in fig4["panel_d_snr_db_values"]:
            panel_d_series.append(
                _ratio_series(
                    config,
                    aa_count_values,
                    lambda count, s=float(input_tokens), snr=float(snr_db): (
                        _scenario(
                            config,
                            input_tokens=s,
                            aa_count=int(count),
                            snr_db=snr,
                        )
                    ),
                    input_tokens=float(input_tokens),
                    snr_db=float(snr_db),
                )
            )
    panel_d = {
        "x_name": "aa_count",
        "x_values": aa_count_values,
        "series": panel_d_series,
        "has_crossover": any(
            series["crosses_boundary"] for series in panel_d_series
        ),
    }
    return {
        "panel_a": panel_a,
        "panel_b": panel_b,
        "panel_c": panel_c,
        "panel_d": panel_d,
    }


def _single_link_arguments(config: dict) -> dict:
    analysis = config["paper_mode_analysis"]
    fig5 = analysis["fig5"]
    return {
        "config": transformer_config(config),
        "compute_flops": tflops_to_flops(
            float(fig5["receiver_compute_tflops"])
        ),
        "bandwidth_hz": ghz_to_hz(float(analysis["bandwidth_ghz"])),
        "bits_per_token": float(analysis["bits_per_token"]),
        "compression_ratio": float(
            analysis["kv_cache_compression_ratio"]
        ),
    }


def _threshold(config: dict, *, alpha, xi, theta, snr_db) -> float | None:
    analysis = config["paper_mode_analysis"]
    value = bandwidth_threshold(
        **_single_link_arguments(config),
        alpha=float(alpha),
        xi=float(xi),
        theta=float(theta),
        snr_db=float(snr_db),
        rho_tolerance=float(analysis["rho_tolerance"]),
        score_tolerance_seconds=float(analysis["score_tolerance_seconds"]),
        max_iterations=int(analysis["threshold_max_iterations"]),
    )
    return None if value is None else float(value)


def _score_series(
    config: dict, rho_values, *, alpha, xi, theta, snr_db
) -> dict:
    scores = [
        float(
            single_link_mode_score(
                **_single_link_arguments(config),
                alpha=float(alpha),
                xi=float(xi),
                theta=float(theta),
                snr_db=float(snr_db),
                rho=float(rho),
            )
        )
        for rho in rho_values
    ]
    return {
        "alpha": float(alpha),
        "xi": float(xi),
        "theta": float(theta),
        "snr_db": float(snr_db),
        "scores": scores,
        "threshold": _threshold(
            config,
            alpha=alpha,
            xi=xi,
            theta=theta,
            snr_db=snr_db,
        ),
    }


def _fig5_panels(config: dict) -> dict:
    fig5 = config["paper_mode_analysis"]["fig5"]
    rho_values = _as_floats(
        np.linspace(fig5["rho_min"], fig5["rho_max"], fig5["rho_points"])
    )
    alpha = float(fig5["generated_tokens"])

    panel_a_series = [
        _score_series(
            config,
            rho_values,
            alpha=alpha,
            xi=xi,
            theta=theta,
            snr_db=fig5["panel_a_snr_db"],
        )
        for theta in fig5["theta_values"]
        for xi in fig5["xi_values"]
    ]
    panel_a = {
        "x_name": "rho",
        "x_values": rho_values,
        "series": panel_a_series,
    }

    panel_b_series = [
        _score_series(
            config,
            rho_values,
            alpha=alpha,
            xi=fig5["panel_b_xi"],
            theta=theta,
            snr_db=snr_db,
        )
        for theta in fig5["theta_values"]
        for snr_db in fig5["panel_b_snr_db_values"]
    ]
    panel_b = {
        "x_name": "rho",
        "x_values": rho_values,
        "series": panel_b_series,
    }

    alpha_values = _as_floats(
        np.linspace(
            fig5["panel_c_alpha_min"],
            fig5["panel_c_alpha_max"],
            fig5["panel_c_alpha_points"],
        )
    )
    xi_values = _as_floats(
        np.linspace(
            fig5["panel_c_xi_min"],
            fig5["panel_c_xi_max"],
            fig5["panel_c_xi_points"],
        )
    )
    surfaces = []
    for theta in fig5["theta_values"]:
        thresholds = [
            [
                _threshold(
                    config,
                    alpha=surface_alpha,
                    xi=xi,
                    theta=theta,
                    snr_db=fig5["panel_c_snr_db"],
                )
                for surface_alpha in alpha_values
            ]
            for xi in xi_values
        ]
        surfaces.append(
            {
                "theta": float(theta),
                "snr_db": float(fig5["panel_c_snr_db"]),
                "thresholds": thresholds,
            }
        )
    panel_c = {
        "alpha_values": alpha_values,
        "xi_values": xi_values,
        "surfaces": surfaces,
    }
    return {"panel_a": panel_a, "panel_b": panel_b, "panel_c": panel_c}


def _all_present(values) -> bool:
    return all(value is not None for value in values)


def _fig4_checks(fig4: dict) -> dict:
    panels = list(fig4.values())
    all_ratios = [
        ratio
        for panel in panels
        for series in panel["series"]
        for ratio in series["ratios"]
    ]
    homogeneous = [
        series
        for series in fig4["panel_b"]["series"]
        if series["system"] == "homogeneous"
    ]
    panel_d_low_snr = [
        series
        for series in fig4["panel_d"]["series"]
        if series["snr_db"] == 0
    ]
    panel_d_high_snr = [
        series
        for series in fig4["panel_d"]["series"]
        if series["snr_db"] > 0
    ]
    return {
        "fig4_has_nl_and_kv_regions": (
            min(all_ratios) < 1 < max(all_ratios)
            and all(panel["has_crossover"] for panel in panels)
        ),
        "fig4_a_nonmonotonic": any(
            _is_nonmonotonic(series["ratios"])
            for series in fig4["panel_a"]["series"]
        ),
        "fig4_c_increases_with_snr": all(
            _strictly_increasing(series["ratios"])
            for series in fig4["panel_c"]["series"]
        ),
        "fig4_d_increases_with_aa_count": (
            bool(panel_d_low_snr)
            and bool(panel_d_high_snr)
            and all(
                _has_single_valley(series["ratios"])
                for series in panel_d_low_snr
            )
            and all(
                _strictly_increasing(series["ratios"])
                for series in panel_d_high_snr
            )
        ),
        "fig4_b_homogeneous_moves_toward_nl": all(
            _strictly_decreasing(series["ratios"])
            for series in homogeneous
        ),
    }


def _thresholds_by(series: list[dict], fixed_name: str, fixed_value) -> list:
    return [
        item["threshold"]
        for item in series
        if item[fixed_name] == float(fixed_value)
    ]


def _surface_alpha_decreases(surface: dict) -> bool:
    for row in surface["thresholds"]:
        present_indices = [
            index for index, value in enumerate(row) if value is not None
        ]
        if len(present_indices) < max(2, math.ceil(0.7 * len(row))):
            return False
        first_present = present_indices[0]
        if any(value is None for value in row[first_present:]):
            return False
        present = row[first_present:]
        decreases = sum(
            left > right for left, right in zip(present, present[1:])
        )
        if (
            present[-1] >= present[0]
            or decreases / (len(present) - 1) < 0.9
        ):
            return False
    return True


def _fig5_checks(fig5: dict) -> dict:
    panel_a_series = fig5["panel_a"]["series"]
    panel_b_series = fig5["panel_b"]["series"]
    theta_values = sorted({item["theta"] for item in panel_a_series})
    xi_values = sorted({item["xi"] for item in panel_a_series})
    snr_values = sorted({item["snr_db"] for item in panel_b_series})

    xi_groups = [
        _thresholds_by(panel_a_series, "theta", theta)
        for theta in theta_values
    ]
    snr_groups = [
        _thresholds_by(panel_b_series, "theta", theta)
        for theta in theta_values
    ]
    theta_pairs = [
        _thresholds_by(panel_a_series, "xi", xi) for xi in xi_values
    ]
    return {
        "fig5_scores_increase_with_rho": all(
            _strictly_increasing(series["scores"])
            for series in panel_a_series + panel_b_series
        ),
        "fig5_xi_increases_threshold": all(
            _all_present(group) and _strictly_increasing(group)
            for group in xi_groups
        ),
        "fig5_snr_decreases_threshold": all(
            _all_present(group) and _strictly_decreasing(group)
            for group in snr_groups
        ),
        "fig5_theta_decreases_threshold": all(
            _all_present(pair) and _strictly_decreasing(pair)
            for pair in theta_pairs
        ),
        "fig5_alpha_decreases_threshold": all(
            _surface_alpha_decreases(surface)
            for surface in fig5["panel_c"]["surfaces"]
        ),
    }


def run_paper_mode_analysis(
    config_path: str = "configs/paper_mode_analysis.yaml",
    output_path: str = "results/logs/paper_mode_analysis.json",
) -> dict:
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = PROJECT_ROOT / config_file
    config = load_config(str(config_file))
    validate_paper_mode_analysis_config(config)
    analysis = config["paper_mode_analysis"]
    fig4_config = analysis["fig4"]

    fig4 = _fig4_panels(config)
    fig5 = _fig5_panels(config)
    checks = {**_fig4_checks(fig4), **_fig5_checks(fig5)}
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(f"paper trend checks failed: {', '.join(failures)}")

    result = {
        "metadata": {
            "schema_version": int(analysis["schema_version"]),
            "config": config,
            "assumptions": {
                "fig4_b_input_tokens": fig4_config["panel_b_input_tokens"],
                "fig4_b_input_tokens_source": fig4_config[
                    "panel_b_input_tokens_source"
                ],
            },
        },
        "fig4": fig4,
        "fig5": fig5,
        "checks": checks,
    }
    output = Path(output_path)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    run_paper_mode_analysis()
