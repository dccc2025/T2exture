# T2exture Agent Instructions

## Runtime Principles

- The workstation is configured for high-performance long-running experiments. Use full available performance when running planned training, evaluation, pseudo-flow generation, visualization, and metric jobs.
- Prefer time-efficient execution. Parallelize independent CPU/I/O/documentation/setup tasks when this does not compete with a running GPU training job.
- Do not run multiple heavy GPU training jobs on the same GPU at the same time unless the user explicitly asks for that tradeoff.
- Quality, fairness, and reproducibility override raw speed. Do not change split, ROI size, crop, loss weights, metric definitions, checkpoint selection, or baseline membership without user confirmation.

## Experiment Fairness

- Formal synthetic experiments use the current fixed split and, after it is generated, `dataset_roi` as the formal `--data-root`.
- Keep source data and derived data separate. Do not overwrite original `dataset/`, `dataset/flow/s10`, `pretrained/`, or third-party code while generating ROI data or experiment artifacts.
- Evaluate final metrics on full ROI frames. Training may use the configured crop, but validation/test metrics and qualitative artifacts must use the same full-resolution ROI protocol across methods.
- Every method in the same table must use the same split, ROI, input resolution, metric implementation, and artifact contract.

## Artifact Requirements

- Preserve every formal experiment under `outputs/final/<table_id>/<exp_id>/`.
- Keep `best.pt`, `last.pt`, `config.json`, `pred/`, `err/`, `metrics.json`, `scene.csv`, `frame.csv`, `manifest.json`, `vis/png/`, and `vis/video/` whenever applicable.
- Do not rely on scalar metrics alone. Keep visual outputs for auditability before treating an experiment as usable.

## Confirmation Boundaries

- Continue automatically for routine code, docs, metric checks, wrappers, visualization, and non-destructive data generation.
- Ask before destructive data changes, overwriting source caches, external uploads, changing the formal experiment protocol, lowering crop/resolution after OOM, or using an uncertain baseline source.
- If results look abnormal or could affect the paper's main conclusion, stop and report the evidence before continuing.
