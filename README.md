# T2exture

AMT-based texture-frame interpolation for the synthetic Active-HADAR dataset.

## Formal Data Root

Formal experiments must use `dataset_roi` as `--data-root`.

```text
dataset_roi/
  train.txt
  valid.txt
  test.txt
  roi_manifest.json
  sim/SCENE/
    texture/001.npy
    passive/001.npy
  flow/s10/SCENE/001_002_011.npz
```

Raw full-frame data under `dataset/` is treated as a source cache only. New
pseudo-flow labels should follow the AMT-style derived-data layout:

```text
DATA_ROOT/flow/sXX/SCENE/LEFT_TARGET_RIGHT.npz
```

Each flow file stores:

```text
flow0: [2, H, W] target-to-left-endpoint flow
flow1: [2, H, W] target-to-right-endpoint flow
```

Training concatenates them as `[4, H, W]` for AMT's `MultipleFlowLoss`.

## Training

Windows PowerShell:

```powershell
conda run -n gflow python -B train.py --data-root dataset_roi --pretrained pretrained/amt-l.pth --backbone amt-l --output-dir outputs/final/table01_prior/ours-l --config train.yaml
```

`train.yaml` is the default formal recipe. It requires `dataset_roi` by default,
uses `active_stride: 10`, and writes checkpoints plus `config.json` under the
chosen output directory.

For local debugging on a non-ROI root, use a separate config with:

```yaml
require_dataset_roi: false
```

## Evaluation And Visualization

```powershell
conda run -n gflow python -B eval.py --data-root dataset_roi --pretrained pretrained/amt-l.pth --backbone amt-l --checkpoint outputs/final/table01_prior/ours-l/best.pt --split test --output-dir outputs/final/table01_prior/ours-l/test --config train.yaml
conda run -n gflow python -B vis.py --eval-dir outputs/final/table01_prior/ours-l/test --method-label Ours-L
```

Evaluation writes `pred/`, `err/`, `metrics.json`, `scene.csv`, `frame.csv`, and
`manifest.json`. Visualization writes `vis/png/` and `vis/video/`; the formal
default style is grayscale for inputs, predictions, GT, and absolute-error
panels.

## Flow Labels

Generate LiteFlowNet pseudo-flow labels for a stride-specific flow set:

```powershell
conda run -n gflow python -B -m flow_generation.generate_liteflownet_flow --data-root dataset_roi --active-stride 5 --splits train valid test --device cuda
```

When `--flow-set` is omitted, the generator writes to `flow/sXX` based on
`--active-stride`, for example `active_stride=5` writes `flow/s05`.

## Tests

```powershell
python -B -m pytest tests/test_smoke.py -q -p no:cacheprovider
```
