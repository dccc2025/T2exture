# T2exture

T2exture is a method-code release for texture-frame interpolation in
active/passive thermal sequences. It adapts AMT-S, AMT-L, and AMT-G backbones to
single-channel texture frames and injects passive-frame context into the AMT
decoder through time-conditioned adapters.

This branch intentionally contains only the core method implementation:

```text
model/            T2exture AMT-S/L/G wrappers and conditioning modules
losses/           Charbonnier, census-style CSS, and pseudo-flow losses
data.py           dataset loader for texture/passive frames and pseudo flow
train.py          two-stage T2exture fine-tuning entry point
config.py         training configuration helpers
train.yaml        default training recipe
environment.yaml  conda environment specification
README.md         this file
```

Datasets, pretrained checkpoints, third-party AMT code, baseline runners,
evaluation scripts, generated outputs, and internal experiment plans are not
included in this method-code branch.

## Environment

Create and activate the environment:

```bash
conda env create -f environment.yaml
conda activate t2exture
```

The provided environment installs PyTorch 2.6 with CUDA 12.4 wheels. If your
system requires another CUDA build, install the matching PyTorch package and
keep the remaining dependencies unchanged.

## External Files

Place the official AMT implementation under:

```text
third_party/AMT_official/
```

The wrappers expect the original AMT network files:

```text
third_party/AMT_official/networks/AMT-S.py
third_party/AMT_official/networks/AMT-L.py
third_party/AMT_official/networks/AMT-G.py
```

Place AMT checkpoints under `pretrained/`, for example:

```text
pretrained/amt-s.pth
pretrained/amt-l.pth
pretrained/amt-g.pth
```

## Dataset Format

Training expects a dataset root such as `datasets/`:

```text
datasets/
  train.txt
  valid.txt
  sim/<scene>/
    texture/001.npy
    passive/001.npy
  flow/s10/<scene>/001_002_011.npz
```

`texture` and `passive` frames are single-channel NumPy arrays. Each pseudo-flow
file stores target-to-endpoint optical flow:

```text
flow0: [2, H, W] target-to-left-anchor flow
flow1: [2, H, W] target-to-right-anchor flow
```

## Training

Run T2exture-L training:

```bash
conda run -n t2exture python -B train.py \
  --data-root datasets \
  --pretrained pretrained/amt-l.pth \
  --backbone amt-l \
  --output-dir outputs/ours-l \
  --config train.yaml
```

Use `--backbone amt-s`, `--backbone amt-l`, or `--backbone amt-g` for the three
T2exture variants. The default training recipe first optimizes the
T2exture-specific adapters, then fine-tunes the rear AMT refinement modules with
layer-wise learning-rate decay.

Training writes:

```text
outputs/ours-l/
  best.pt
  last.pt
  config.json
```

## Notes

- Set `passive_context: 0` in `train.yaml` to remove passive-context
  conditioning and train the no-context variant.
- `pseudo_flow_dir` is resolved relative to `--data-root`.
- Large files should be distributed separately from this Git branch.
