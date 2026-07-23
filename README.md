# T2exture

AMT fine-tuning for texture-frame interpolation using active texture endpoints and passive-frame context.

## Dataset Layout

The runnable synthetic dataset should be scene-centric:

```text
DATA_ROOT/
  train.txt
  valid.txt
  test.txt
  sim/SCENE/
    texture/001.npy
    passive/001.npy
    flow/001_002_011.npz
```

`texture` and `passive` frames are grayscale `H x W` arrays. `flow` stores LiteFlowNet pseudo labels following AMT's target-to-endpoint convention:

```text
flow0: [2, H, W] target-to-left-endpoint flow
flow1: [2, H, W] target-to-right-endpoint flow
```

Training concatenates them as `[4, H, W]` for AMT's `MultipleFlowLoss`.

## Training

```bash
bash scripts/train.sh DATA_ROOT pretrained/amt-l.pth outputs/runs/experiment_01
```

`train.yaml` is the default recipe. It sets crop size, iteration schedule, learning rates, layer-wise LR decay, validation interval, and loss weights.

## Flow Labels

If a legacy cache exists under `DATA_ROOT/pseudo_flow/liteflownet_default_train_valid`, link it into the scene layout without duplicating storage:

```bash
python -m flow_generation.link_legacy_pseudo_flow --data-root DATA_ROOT
```

Generate missing LiteFlowNet labels with:

```bash
python -m flow_generation.generate_liteflownet_flow --data-root DATA_ROOT --splits train valid test
```

The generator mirrors AMT's target-to-endpoint flow convention but writes the T2exture scene-centric npz layout.

## Tests

```bash
python -m unittest tests.test_smoke -v
```

Use the training environment for metric tests:

```bash
conda run -n gflow python -m unittest tests.test_smoke -v
```
