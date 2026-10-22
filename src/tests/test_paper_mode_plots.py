import json
from copy import deepcopy
from pathlib import Path

import pytest

from simulations.paper_mode_analysis import run_paper_mode_analysis
from simulations.plot_paper_mode_analysis import plot_paper_mode_analysis


def test_plotter_generates_fig4_and_fig5_pngs(tmp_path):
    log_path = tmp_path / "analysis.json"
    figure_dir = tmp_path / "figures"
    run_paper_mode_analysis(output_path=str(log_path))
    outputs = plot_paper_mode_analysis(str(log_path), str(figure_dir))

    assert {Path(path).name for path in outputs} == {
        "paper_fig4_latency_ratio.png",
        "paper_fig5_single_link_selection.png",
    }
    assert all(Path(path).stat().st_size > 0 for path in outputs)


def test_plotter_rejects_unsupported_schema(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text(
        json.dumps({"metadata": {"schema_version": 2}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema_version must be 1"):
        plot_paper_mode_analysis(str(path), str(tmp_path / "figures"))


def test_plotter_validates_consumed_series_metadata(tmp_path):
    path = tmp_path / "analysis.json"
    data = run_paper_mode_analysis(output_path=str(path))
    del data["fig4"]["panel_b"]["series"][0]["system"]
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match=r"panel_b.*system"):
        plot_paper_mode_analysis(str(path), str(tmp_path / "figures"))


def test_plotter_rejects_extra_threshold_surface(tmp_path):
    path = tmp_path / "analysis.json"
    data = run_paper_mode_analysis(output_path=str(path))
    surfaces = data["fig5"]["panel_c"]["surfaces"]
    surfaces.append(deepcopy(surfaces[0]))
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="surfaces must contain exactly 2"):
        plot_paper_mode_analysis(str(path), str(tmp_path / "figures"))
