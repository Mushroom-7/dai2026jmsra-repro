import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch


ROOT = Path(__file__).resolve().parents[1]

GROUP_TITLES = {
    "input": "论文参数与配置",
    "compute": "计算与智能体",
    "communication": "无线通信",
    "decision": "介质选择与资源分配",
    "state": "多轮状态",
    "experiment": "复现实验",
    "evidence": "证据与输出",
    "process": "处理过程",
    "verification": "核验",
    "research": "研究扩展",
}

GROUP_COLORS = {
    "input": ("#EAF2F8", "#2E6F9E"),
    "compute": ("#E8F5E9", "#347A46"),
    "communication": ("#FFF3E0", "#B66516"),
    "decision": ("#F3E8F7", "#7B438F"),
    "state": ("#FDECEC", "#A94343"),
    "experiment": ("#E7F4F3", "#277A75"),
    "evidence": ("#F1F3F5", "#4D5964"),
    "process": ("#EEF3F9", "#486B8C"),
    "verification": ("#FFF7DB", "#99731B"),
    "research": ("#E9F5EE", "#327052"),
}


@dataclass(frozen=True)
class Node:
    id: str
    label: str
    group: str
    position: tuple[float, float]
    size: tuple[float, float] = (2.7, 0.95)


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    label: str = ""
    dashed: bool = False


@dataclass(frozen=True)
class Diagram:
    slug: str
    title: str
    direction: str
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]


def _nodes(*items: tuple) -> tuple[Node, ...]:
    return tuple(Node(*item) for item in items)


def _edges(*items: tuple) -> tuple[Edge, ...]:
    return tuple(Edge(*item) for item in items)


def _system_architecture() -> Diagram:
    return Diagram(
        slug="system-architecture",
        title="实验主体代码系统框图：从参数设置到结果输出",
        direction="LR",
        nodes=_nodes(
            ("parameters", "实验参数设置\nTable 2--4 / 三份 YAML", "input", (1.5, 5.7)),
            ("scenario", "配置加载与场景生成\ndata_gen.py", "process", (4.7, 5.7)),
            ("transformer", "Transformer 计算原语\nFLOP / Prefill / AR / KV", "compute", (8.0, 8.0)),
            ("channel", "无线信道原语\n路径损耗 / 衰落 / 速率", "communication", (8.0, 3.4)),
            ("candidates", "NL/KV 候选时延\n计算 + 传输 + 聚合", "process", (11.3, 5.7)),
            ("media_selection", "介质选择分析\n时延比 / 单链路阈值", "decision", (14.6, 8.0)),
            ("jmsra", "JMSRA 与基线\n模式选择 + 带宽分配", "decision", (14.6, 3.4)),
            ("state", "多轮状态更新\n上下文 / KV 债务 / 窗口", "state", (17.8, 0.8)),
            ("fig45", "Fig.4--5 分析实验\n参数扫描 / 选择边界", "experiment", (17.8, 8.6)),
            ("fig67", "Fig.6--7 单轮实验\n路径 / 带宽 / AA 数扫描", "experiment", (17.8, 5.7)),
            ("fig8", "Fig.8 多轮协作实验\nEA 广播 / AA 响应 / EA 聚合", "experiment", (17.8, 2.8)),
            ("metrics", "实验指标汇总\n时延 / 模式 / 带宽 / 上下文", "evidence", (21.0, 5.7)),
            ("json", "JSON 数值结果\n逐点数据 / 轮次状态", "evidence", (24.2, 8.2)),
            ("plots", "论文图表输出\nFig.4--8 PNG", "evidence", (24.2, 5.7)),
            ("checks", "论文结论核验\n趋势 / 锚点 / 曲线顺序", "verification", (24.2, 3.2)),
        ),
        edges=_edges(
            ("parameters", "scenario"),
            ("scenario", "transformer", "模型与负载"),
            ("scenario", "channel", "拓扑与信道样本"),
            ("transformer", "candidates", "计算时延"),
            ("channel", "candidates", "传输速率"),
            ("candidates", "media_selection", "NL/KV 时延"),
            ("candidates", "jmsra", "每 AA 候选"),
            ("media_selection", "fig45", "扫描"),
            ("jmsra", "fig67", "优化 / 基线"),
            ("media_selection", "fig8"),
            ("jmsra", "fig8"),
            ("fig8", "state", "更新并反馈下一轮", True),
            ("fig45", "metrics"),
            ("fig67", "metrics"),
            ("fig8", "metrics"),
            ("metrics", "json", "保存原始结果"),
            ("metrics", "plots", "绘图"),
            ("metrics", "checks", "对照论文"),
        ),
    )


def _data_flow() -> Diagram:
    return Diagram(
        slug="data-flow",
        title="单轮仿真数据流与 40 轮状态闭环",
        direction="LR",
        nodes=_nodes(
            ("config", "本轮配置\n活跃 AAs / Table 4 参数", "input", (1.5, 8.3)),
            ("previous", "上一轮状态\n上下文与 KV 债务", "state", (1.5, 4.8)),
            ("sample", "采样无线环境\n距离 / 衰落 / 增益", "process", (5.0, 8.3)),
            ("candidates", "计算 NL/KV\n广播候选时延", "process", (8.5, 8.3)),
            ("broadcast", "选择 EA 广播介质", "decision", (12.0, 8.3)),
            ("aa_infer", "AAs 接收并本地推理", "compute", (15.5, 8.3)),
            ("return_candidates", "形成 AAs 返回候选\nNL/KV 负载与时延", "process", (15.5, 4.8)),
            ("jmsra", "JMSRA 联合优化\n模式 + 带宽比例", "decision", (12.0, 4.8)),
            ("ea_infer", "EA 聚合与推理\nPrefill + Autoregressive", "compute", (8.5, 4.8)),
            ("update", "状态转移\n截断 / 累债 / 清债", "state", (8.5, 1.3)),
            ("metrics", "本轮指标\n时延 / 模式 / 带宽", "evidence", (5.0, 1.3)),
            ("json", "追加 JSON 轮次记录", "evidence", (1.5, 1.3)),
        ),
        edges=_edges(
            ("config", "sample", "环境范围"),
            ("previous", "candidates", "上下文与债务"),
            ("sample", "candidates", "广播速率"),
            ("candidates", "broadcast", "候选时延"),
            ("broadcast", "aa_infer", "NL 或 KV"),
            ("aa_infer", "return_candidates", "响应状态"),
            ("sample", "return_candidates", "上行速率"),
            ("return_candidates", "jmsra", "每 AA 候选"),
            ("jmsra", "ea_infer", "选定负载"),
            ("previous", "ea_infer", "EA 上下文"),
            ("ea_infer", "update", "生成结果"),
            ("aa_infer", "update", "AA 新状态"),
            ("update", "metrics", "状态与耗时"),
            ("metrics", "json", "序列化"),
            ("update", "previous", "下一轮反馈", True),
        ),
    )


def _reproduction_workflow() -> Diagram:
    return Diagram(
        slug="reproduction-workflow",
        title="论文数值复现、核验与研究扩展工作流",
        direction="TB",
        nodes=_nodes(
            ("paper", "论文公式、Table 2--4\n与 Fig.4--8 可见锚点", "input", (9.0, 9.0)),
            ("parameters", "公开参数映射\nYAML 配置", "process", (5.0, 7.2)),
            ("assumptions", "未公开信息登记\n输入长度 / 种子 / 活跃轨迹", "process", (13.0, 7.2)),
            ("fig45", "Fig.4--5\n解析时延与介质阈值", "experiment", (3.0, 5.1)),
            ("fig67", "Fig.6--7\nJMSRA 确定性校准", "experiment", (9.0, 5.1)),
            ("fig8", "Fig.8\n40 轮动态状态仿真", "experiment", (15.0, 5.1)),
            ("artifacts", "JSON 原始数值\nPNG 复现图表", "evidence", (5.0, 3.0)),
            ("compare", "论文锚点、曲线顺序\n与趋势检查", "verification", (9.0, 3.0)),
            ("tests", "pytest 回归测试\n配置 / 公式 / 绘图", "verification", (13.0, 3.0)),
            ("decision", "完成度判定\n证据边界与差异说明", "verification", (9.0, 1.1)),
            ("report", "总结报告与系统图", "evidence", (5.0, -0.8)),
            ("innovation", "研究扩展\n目标函数 / 信道 / 策略 / 实机", "research", (13.0, -0.8)),
        ),
        edges=_edges(
            ("paper", "parameters", "提取"),
            ("paper", "assumptions", "识别缺失信息"),
            ("parameters", "fig45", "参数化"),
            ("parameters", "fig67", "参数化"),
            ("parameters", "fig8", "参数化"),
            ("assumptions", "fig45", "显式假设"),
            ("assumptions", "fig67", "确定性种子"),
            ("assumptions", "fig8", "推断轨迹"),
            ("fig45", "artifacts", "输出"),
            ("fig67", "artifacts", "输出"),
            ("fig8", "artifacts", "输出"),
            ("artifacts", "compare", "数值证据"),
            ("fig45", "tests", "实现"),
            ("fig67", "tests", "实现"),
            ("fig8", "tests", "实现"),
            ("compare", "decision", "趋势与偏差"),
            ("tests", "decision", "回归结果"),
            ("decision", "report", "形成结论"),
            ("decision", "innovation", "建立基线"),
        ),
    )


def build_diagrams() -> tuple[Diagram, ...]:
    return (_system_architecture(), _data_flow(), _reproduction_workflow())


def validate_diagram(diagram: Diagram) -> None:
    ids = [node.id for node in diagram.nodes]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate node id in {diagram.slug}")
    known = set(ids)
    for node in diagram.nodes:
        if node.group not in GROUP_TITLES:
            raise ValueError(f"unknown group {node.group} in {diagram.slug}")
    for edge in diagram.edges:
        if edge.source not in known or edge.target not in known:
            raise ValueError(f"unknown edge endpoint in {diagram.slug}")


def _mermaid_label(label: str) -> str:
    return label.replace("\n", "<br/>").replace('"', "#quot;")


def render_mermaid(diagram: Diagram) -> str:
    validate_diagram(diagram)
    lines = [f"flowchart {diagram.direction}"]
    groups = dict.fromkeys(node.group for node in diagram.nodes)
    for group in groups:
        lines.append(
            f'    subgraph cluster_{group}["{GROUP_TITLES[group]}"]'
        )
        for node in diagram.nodes:
            if node.group == group:
                lines.append(
                    f'        {node.id}["{_mermaid_label(node.label)}"]'
                )
        lines.append("    end")
    for edge in diagram.edges:
        connector = "-.->" if edge.dashed else "-->"
        if edge.label:
            lines.append(
                f'    {edge.source} {connector}|"{_mermaid_label(edge.label)}"| '
                f"{edge.target}"
            )
        else:
            lines.append(f"    {edge.source} {connector} {edge.target}")
    for group in groups:
        fill, stroke = GROUP_COLORS[group]
        lines.append(
            f"    classDef {group} fill:{fill},stroke:{stroke},color:#17202A,stroke-width:1.5px"
        )
        node_ids = ",".join(
            node.id for node in diagram.nodes if node.group == group
        )
        lines.append(f"    class {node_ids} {group}")
    return "\n".join(lines) + "\n"


def write_mermaid_files(diagrams: tuple[Diagram, ...]) -> tuple[Path, ...]:
    output_dir = ROOT / "docs" / "diagrams"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for diagram in diagrams:
        output = output_dir / f"{diagram.slug}.mmd"
        output.write_text(render_mermaid(diagram), encoding="utf-8")
        outputs.append(output)
    return tuple(outputs)


def _configure_font() -> str:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in (
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ):
        if candidate in available:
            plt.rcParams["font.sans-serif"] = [candidate]
            plt.rcParams["axes.unicode_minus"] = False
            return candidate
    return "DejaVu Sans"


def _boundary_point(source: Node, target: Node) -> tuple[float, float]:
    dx = target.position[0] - source.position[0]
    dy = target.position[1] - source.position[1]
    if dx == 0 and dy == 0:
        raise ValueError(f"overlapping edge endpoints: {source.id}")
    x_scale = math.inf if dx == 0 else (source.size[0] / 2) / abs(dx)
    y_scale = math.inf if dy == 0 else (source.size[1] / 2) / abs(dy)
    scale = min(x_scale, y_scale)
    return (
        source.position[0] + dx * scale,
        source.position[1] + dy * scale,
    )


def _draw_edge(axis, edge: Edge, nodes: dict[str, Node]) -> None:
    source = nodes[edge.source]
    target = nodes[edge.target]
    start = _boundary_point(source, target)
    end = _boundary_point(target, source)
    color = "#67717B" if edge.dashed else "#34495E"
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.45,
        linestyle="--" if edge.dashed else "-",
        color=color,
        connectionstyle="arc3,rad=0.0",
        zorder=1,
    )
    axis.add_patch(arrow)
    if edge.label:
        midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        horizontal = abs(end[0] - start[0]) >= abs(end[1] - start[1])
        offset = (0.0, 0.16) if horizontal else (0.18, 0.0)
        axis.text(
            midpoint[0] + offset[0],
            midpoint[1] + offset[1],
            edge.label,
            ha="center",
            va="center",
            fontsize=7.4,
            color="#34495E",
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.9,
            },
            zorder=2,
        )


def _draw_node(axis, node: Node) -> None:
    fill, stroke = GROUP_COLORS[node.group]
    x, y = node.position
    width, height = node.size
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor=fill,
        edgecolor=stroke,
        linewidth=1.8,
        zorder=3,
    )
    axis.add_patch(patch)
    axis.text(
        x,
        y,
        node.label,
        ha="center",
        va="center",
        fontsize=9.6,
        color="#17202A",
        linespacing=1.25,
        zorder=4,
    )


def render_png(
    diagram: Diagram,
    output_path: str | Path,
    dpi: int = 200,
) -> str:
    validate_diagram(diagram)
    _configure_font()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(20, 12.5), dpi=dpi)
    nodes = {node.id: node for node in diagram.nodes}

    for edge in diagram.edges:
        _draw_edge(axis, edge, nodes)
    for node in diagram.nodes:
        _draw_node(axis, node)

    left = min(node.position[0] - node.size[0] / 2 for node in diagram.nodes)
    right = max(node.position[0] + node.size[0] / 2 for node in diagram.nodes)
    bottom = min(node.position[1] - node.size[1] / 2 for node in diagram.nodes)
    top = max(node.position[1] + node.size[1] / 2 for node in diagram.nodes)
    axis.set_xlim(left - 0.9, right + 0.9)
    axis.set_ylim(bottom - 1.15, top + 0.85)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")
    axis.set_title(diagram.title, fontsize=17, fontweight="bold", pad=18)

    groups = dict.fromkeys(node.group for node in diagram.nodes)
    legend_handles = [
        Patch(
            facecolor=GROUP_COLORS[group][0],
            edgecolor=GROUP_COLORS[group][1],
            label=GROUP_TITLES[group],
        )
        for group in groups
    ]
    axis.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=min(5, len(legend_handles)),
        frameon=False,
        fontsize=8.2,
    )
    figure.savefig(
        output,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.16,
    )
    plt.close(figure)
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(f"empty diagram output: {output}")
    return str(output)


def write_png_files(diagrams: tuple[Diagram, ...]) -> tuple[Path, ...]:
    output_dir = ROOT / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for diagram in diagrams:
        output = output_dir / f"{diagram.slug.replace('-', '_')}.png"
        render_png(diagram, output)
        outputs.append(output)
    return tuple(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render reproduction architecture diagrams."
    )
    parser.add_argument(
        "--format",
        choices=("all", "mermaid", "png"),
        default="all",
        help="Output format to generate.",
    )
    arguments = parser.parse_args()
    diagrams = build_diagrams()
    outputs = []
    if arguments.format in ("all", "mermaid"):
        outputs.extend(write_mermaid_files(diagrams))
    if arguments.format in ("all", "png"):
        outputs.extend(write_png_files(diagrams))
    for output in outputs:
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
