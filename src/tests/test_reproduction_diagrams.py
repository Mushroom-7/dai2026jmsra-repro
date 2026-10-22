import importlib
import importlib.util
from pathlib import Path

import matplotlib.image as mpimg
import pytest


MODULE_NAME = "simulations.render_reproduction_diagrams"
ROOT = Path(__file__).resolve().parents[2]


def _load_diagram_module():
    spec = importlib.util.find_spec(MODULE_NAME)
    assert spec is not None, f"{MODULE_NAME} must exist"
    return importlib.import_module(MODULE_NAME)


def test_diagram_models_are_valid_and_cover_all_views():
    module = _load_diagram_module()
    diagrams = module.build_diagrams()

    assert {diagram.slug for diagram in diagrams} == {
        "system-architecture",
        "data-flow",
        "reproduction-workflow",
    }
    for diagram in diagrams:
        module.validate_diagram(diagram)
        assert len(diagram.nodes) >= 8
        assert len(diagram.edges) >= 7


def test_mermaid_uses_the_shared_nodes_and_edges():
    module = _load_diagram_module()

    for diagram in module.build_diagrams():
        output = module.render_mermaid(diagram)
        assert output.startswith("flowchart ")
        for node in diagram.nodes:
            assert node.id in output
        for edge in diagram.edges:
            assert edge.source in output
            assert edge.target in output


def test_mermaid_group_ids_do_not_collide_with_node_ids():
    module = _load_diagram_module()

    for diagram in module.build_diagrams():
        output = module.render_mermaid(diagram)
        for group in {node.group for node in diagram.nodes}:
            assert f"subgraph cluster_{group}" in output


def test_system_architecture_covers_experiment_computation_flow():
    module = _load_diagram_module()
    diagram = next(
        item
        for item in module.build_diagrams()
        if item.slug == "system-architecture"
    )
    node_ids = {node.id for node in diagram.nodes}

    assert {
        "parameters",
        "scenario",
        "transformer",
        "channel",
        "candidates",
        "media_selection",
        "jmsra",
        "state",
        "fig45",
        "fig67",
        "fig8",
        "metrics",
        "json",
        "plots",
        "checks",
    }.issubset(node_ids)
    assert {
        "suite",
        "cli",
        "run_manager",
        "run_dir",
        "manifest",
    }.isdisjoint(node_ids)


@pytest.mark.parametrize(
    "slug",
    ["system-architecture", "data-flow", "reproduction-workflow"],
)
def test_png_render_is_high_resolution_and_nonblank(tmp_path, slug):
    module = _load_diagram_module()
    assert hasattr(module, "render_png"), "render_png must exist"
    diagram = next(
        item for item in module.build_diagrams() if item.slug == slug
    )
    output = tmp_path / f"{slug}.png"

    module.render_png(diagram, output)

    pixels = mpimg.imread(output)
    assert max(pixels.shape[:2]) >= 2400
    assert pixels.std() > 0.01


def test_summary_report_covers_scope_evidence_and_diagrams():
    report_path = ROOT / "docs" / "reproduction-summary-report.md"
    assert report_path.exists(), "summary report must exist"
    report = report_path.read_text(encoding="utf-8")

    for heading in [
        "执行摘要",
        "复现范围",
        "系统框图",
        "数据流图",
        "工作流",
        "参数对照",
        "实验对照",
        "完成度",
        "研究创新",
    ]:
        assert heading in report
    assert report.count("```mermaid") == 3
    assert "LLaMA-7B" in report
    assert "不加载模型权重" in report
    assert "趋势复现" in report
