# Script Entry Points

This directory contains reproducible command-line entry points used by the
T2exture experiments. Scripts are grouped by responsibility so reviewers can
find the correct command without reading the full experiment runner.

## Layout

```text
scripts/
  baselines/  official prior-method evaluators for Table 1 and Table 6
  formal/     preflight, experiment runner, metrics collection, deployment profiling
  real/       real-sequence benchmark evaluation and no-reference metric backfill
  figures/    paper-figure asset export utilities
```

## Common Commands

Train, evaluate, and visualize the three release checkpoints:

```powershell
conda run -n t2exture python -B scripts/formal/train_release_models.py --variants s l g
```

Preflight the formal synthetic protocol:

```powershell
conda run -n t2exture python -B scripts/formal/preflight.py --manifest configs/formal/experiment_manifest.yaml --config train.yaml
```

Collect table-ready metrics from `outputs/final`:

```powershell
conda run -n t2exture python -B scripts/formal/collect_metrics.py --outputs-root outputs/final --csv outputs/final/_summary/metrics_summary.csv --markdown outputs/final/_summary/metrics_summary.md --per-table-dir outputs/final/_summary/by_table
```

Run the Ours-L component ablation table, scheduling only the middle rows:

```powershell
conda run -n t2exture python -B scripts/formal/run_experiments.py --table07
```

Export main-figure assets from a selected sequence:

```powershell
conda run -n t2exture python -B scripts/figures/export_main_figure_assets.py --data-root datasets --scene spot --left-id 41 --target-id 46 --right-id 51 --passive-ids 43 44 45 47 48 49 --pretrained pretrained/amt-l.pth --checkpoint outputs/final/table03_passive_context/context-per-side-03/best.pt --prediction-png outputs/final/table03_passive_context/context-per-side-03/test/pred/spot/041_046_051.png --backbone amt-l --output-dir outputs/main_figure_assets/spot_041_046_051 --device cuda
```

For the full formal command list, use `configs/formal/experiment_manifest.yaml`.
