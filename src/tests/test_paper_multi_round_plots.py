import json
from pathlib import Path

import matplotlib.image as mpimg
import pytest

from simulations.paper_multi_round import run_paper_multi_round
from simulations.plot_paper_multi_round import plot_paper_multi_round


@pytest.fixture(scope="module")
def plot_source(tmp_path_factory):
    source = tmp_path_factory.mktemp("paper-multi-round-plot") / "result.json"
    run_paper_multi_round("configs/paper_multi_round.yaml", str(source))
    return source


def test_plotter_rejects_unsupported_schema(tmp_path):
    source = tmp_path / "bad.json"
    source.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        plot_paper_multi_round(str(source), str(tmp_path / "bad.png"))


def test_plotter_rejects_invalid_schedule_value(plot_source, tmp_path):
    data = json.loads(plot_source.read_text(encoding="utf-8"))
    data["schedule"][0][0] = 2
    source = tmp_path / "bad-schedule.json"
    source.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="schedule"):
        plot_paper_multi_round(str(source), str(tmp_path / "bad.png"))


def test_plotter_creates_nonblank_two_panel_figure(plot_source, tmp_path):
    target = tmp_path / "fig8.png"
    assert Path(
        plot_paper_multi_round(str(plot_source), str(target))
    ).exists()
    pixels = mpimg.imread(target)
    assert pixels.std() > 0.01
    assert pixels.shape[1] > pixels.shape[0]
