# Token/KV-Cache Communication Media Selection Reproduction

This project reproduces the analytical latency simulation model from the
Token/KV-cache communication media selection and resource allocation paper for
multi-agent collaboration.

## Scope

The implementation models Transformer inference latency, wireless transmission
latency, static NL/KV baselines, and the JMSRA joint media selection and
bandwidth allocation algorithm. It does not run real LLM inference.

## Environment

Use the project-local virtual environment for every command.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Simulations

### One-Command Paper Reproduction

Run Fig. 4 through Fig. 8, generate every paper plot, validate the built-in
paper trends, and run the complete regression suite with one command:

```powershell
.\.venv\Scripts\python.exe simulations\run_paper_reproduction.py --run-id baseline --with-tests
```

Without `--run-id`, the runner creates a UTC timestamp ID. Every run is
isolated below `results/runs/<run-id>/` and contains numerical logs, figures,
configuration snapshots, pytest output when requested, and
`run_manifest.json`. Existing run directories are never overwritten.

Run one experiment group or skip plotting for numerical parameter sweeps:

```powershell
.\.venv\Scripts\python.exe simulations\run_paper_reproduction.py --suite fig8 --run-id fig8-trial
.\.venv\Scripts\python.exe simulations\run_paper_reproduction.py --suite fig45 --run-id fig45-data-only --skip-plots
```

The suite manifest is `configs/reproduction_suite.yaml`. It references the
three adjustable parameter files:

- `configs/paper_mode_analysis.yaml` for Fig. 4 and Fig. 5
- `configs/paper_jmsra.yaml` for Fig. 6 and Fig. 7
- `configs/paper_multi_round.yaml` for Fig. 8

Use `--config` for another suite manifest and `--output-root` for another run
directory parent. Run `--help` for the complete command-line interface.

### General Simulations

```powershell
.\.venv\Scripts\python.exe simulations\single_round.py
.\.venv\Scripts\python.exe simulations\multi_round.py
.\.venv\Scripts\python.exe simulations\plot_results.py
```

### Paper Fig. 4 and Fig. 5 Analysis

```powershell
.\.venv\Scripts\python.exe simulations\paper_mode_analysis.py
.\.venv\Scripts\python.exe simulations\plot_paper_mode_analysis.py
```

This analytical reproduction evaluates the paper's complete two-round NL/KV
latency equations for Fig. 4 and the single-link marginal mode-selection
criterion for Fig. 5. The runner writes the raw values, threshold surfaces,
crossover summaries, and explicit paper-trend checks to:

- `results/logs/paper_mode_analysis.json`
- `results/figures/paper_fig4_latency_ratio.png`
- `results/figures/paper_fig5_single_link_selection.png`

The paper and its public TeX source do not disclose the input length used for
Fig. 4(b). This reproduction uses `s = 512` as an explicit calibration
assumption; the value and its source are recorded in both the YAML config and
JSON metadata.

### Paper JMSRA Calibration

```powershell
.\.venv\Scripts\python.exe simulations\paper_jmsra_calibration.py
.\.venv\Scripts\python.exe simulations\plot_paper_jmsra.py
```

The calibration targets Table 3, Fig. 6, and Fig. 7 from the paper. The
generated JSON reports deltas from the Fig. 6 paper anchors because the paper
does not disclose the exact random seed or topology samples.

Unlike the analytical Fig. 4/Fig. 5 reproduction above, Fig. 6/Fig. 7 use
deterministic stochastic calibration against the paper's reported and visible
anchors.

The deterministic calibration seed (`1477`) was selected from the fixed
integer range `[0, 4999]`. Candidates were first filtered against the visible
Fig. 6 and Fig. 7 baseline anchors, then ranked by the disclosed JMSRA latency
and KV-count anchors. The selection metadata is stored in
`configs/paper_jmsra.yaml`; it is a reproducible calibration procedure, not a
claim that the paper used the same random topology.

The Fig. 6 calibration output also records deterministic AA positions and the
selected NL/KV mode for each AA. The topology plots reproduce the paper's
spatial interpretation: AAs are sampled in the 5-10 m range around the EA, and
KV mode is favored more strongly when EA compute is limited.

### Paper Table 4 and Fig. 8 Multi-Round Analysis

```powershell
.\.venv\Scripts\python.exe simulations\paper_multi_round.py
.\.venv\Scripts\python.exe simulations\plot_paper_multi_round.py
```

This experiment runs the complete 40-round analytical dialogue state model,
including EA-to-AAs broadcast selection, AAs-to-EA JMSRA allocation, context
limits, KV-debt windows, and debt clearing. It writes:

- `results/logs/paper_multi_round.json`
- `results/figures/paper_fig8_multi_round.png`

The analytical LLaMA-7B profile does not execute model weights; it only supplies
the FLOP and KV-volume parameters. The paper does not disclose its random seed or exact active-AA trace.
The configured `figure-inferred-calibration` trace follows the contiguous
Waiting regions visible in Fig. 8(a), and both this trace and the deterministic
seed are labeled as assumptions in the JSON metadata.

The resulting EA behavior matches the main reported trends: the EA uses KV in
the first round and NL afterward, context saturates near 400K, Pure-NL peaks at
the paper's reported scale, and JMSRA suppresses deep-round EA computation.
The early rounds contain a larger NL share than deep rounds, but the generated
heatmap remains more KV-heavy than the published Fig. 8(a). It is therefore a
parameter- and equation-aligned trend reproduction, not a pointwise heatmap
match.

## Reproduction Matrix

| Paper content | Repository coverage | Evidence boundary |
| --- | --- | --- |
| Table 2, Fig. 4 | Complete analytical two-round E2E evaluation | Fig. 4(b) input length is an explicit calibration assumption |
| Table 2, Fig. 5 | Complete analytical single-link threshold evaluation | Equation and qualitative boundary reproduction |
| Table 3, Fig. 6 | Deterministic JMSRA calibration | Matches reported anchors within recorded deltas; original seed is unavailable |
| Table 3, Fig. 7 | Deterministic bandwidth and AA-count scans | Reproduces curve ordering and trends |
| Table 4, Fig. 8 | Complete 40-round analytical state and plotting pipeline | Active trace and seed are inferred; early heatmap is only trend-aligned |
| Real LLaMA execution and semantic quality | Not implemented | Outside the paper's numerical latency evaluation |

## Summary Report and System Diagrams

The Chinese [reproduction summary report](docs/reproduction-summary-report.md)
documents the paper-method mapping, parameter and experiment comparisons,
completion boundary, and research-extension options. It embeds Mermaid versions
of the system architecture, runtime data flow, and reproduction workflow.

Standalone high-resolution diagrams are available at:

- `results/figures/system_architecture.png`
- `results/figures/data_flow.png`
- `results/figures/reproduction_workflow.png`

Regenerate all Mermaid and PNG diagram artifacts with:

```powershell
.\.venv\Scripts\python.exe simulations\render_reproduction_diagrams.py
```

Individual legacy entry points write outputs to:

- `results/logs/`
- `results/figures/`

Paper calibration figure files:

- `paper_fig6_latency_paths.png`
- `paper_fig6_topology_c0_5_tflops.png`
- `paper_fig6_topology_c0_35_tflops.png`
- `paper_fig7_bandwidth_scan.png`
- `paper_fig7_aa_count_scan.png`

## Notes

The simulation framework targets paper-consistent trends. Paper calibration
reports anchor deltas instead of claiming exact figure matching when random
topology samples are not disclosed.
