# T2exture

T2exture interpolates texture frames in active/passive thermal sequences.

This branch keeps the code path for three trained models:

```text
T2exture-S   AMT-S backbone
T2exture-L   AMT-L backbone
T2exture-G   AMT-G backbone
```

The implementation follows the main figure: active texture anchors are first
mapped by the T2V adapter, the target time is encoded with FourierConv2d, and
passive frames are injected through a three-level Conv-P guidance pyramid in
the AMT decoder.

## quick start

```bash
git clone -b 0725_code_cjs https://github.com/dccc2025/T2exture.git
cd T2exture

conda env create -f environment.yaml
conda activate t2exture
git clone https://github.com/MCG-NKU/AMT third_party/AMT_official
```

Download the prepared data:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='chenjiashuo/T2exture_datasets', repo_type='dataset', local_dir='datasets')"
```

Run one synthetic test split:

```bash
python -B infer.py \
  --mode synthetic \
  --variant l \
  --data-root datasets \
  --split test \
  --output-dir outputs/infer/synthetic-l
```

Run one real sequence:

```bash
python -B infer.py \
  --mode real \
  --variant l \
  --data-root datasets/real \
  --sequences bag \
  --output-dir outputs/infer/real-l-bag
```

If `--checkpoint` is not given, `infer.py` downloads the selected checkpoint
from `chenjiashuo/T2exture_model`.

## data

Dataset repo:

```text
https://huggingface.co/datasets/chenjiashuo/T2exture_datasets
```

Expected layout after download:

```text
datasets/
  train.txt
  valid.txt
  test.txt
  dataset_manifest.json
  preview/
  sim/<scene>/
    texture/001.npy
    passive/001.npy
  flow/s10/<scene>/001_002_011.npz
  real/<sequence>/001.png
```

`train.yaml` uses the fixed synthetic split, active stride 10, four passive
context frames, 384 crops for training, and full ROI frames for validation,
testing, and inference.

Before training, check the local data once:

```bash
python -B scripts/preflight.py --data-root datasets --config train.yaml
```

## weights

Model repo:

```text
https://huggingface.co/chenjiashuo/T2exture_model
```

Files:

```text
t2exture-s.pt
t2exture-l.pt
t2exture-g.pt
```

These checkpoints contain the full T2exture model state, including the AMT
backbone weights used for inference. For training from scratch, place the
official AMT initialization files here:

```text
pretrained/amt-s.pth
pretrained/amt-l.pth
pretrained/amt-g.pth
```

## train

Train all three variants:

```bash
python -B scripts/train_ours.py --variants s l g
```

Train one variant manually:

```bash
python -B train.py \
  --data-root datasets \
  --pretrained pretrained/amt-l.pth \
  --backbone amt-l \
  --output-dir outputs/final/ours/t2exture-l \
  --config train.yaml
```

`train.py` writes `best.pt`, `last.pt`, and `config.json`.

## evaluate

```bash
python -B eval.py \
  --data-root datasets \
  --checkpoint pretrained/t2exture_model/t2exture-l.pt \
  --backbone amt-l \
  --split test \
  --output-dir outputs/eval/t2exture-l \
  --config train.yaml
```

Optional visual sheets and videos:

```bash
python -B vis.py \
  --eval-dir outputs/eval/t2exture-l \
  --output-dir outputs/eval/t2exture-l/vis \
  --method-label T2exture-L \
  --max-frames-per-scene 5
```

Synthetic test numbers for the hosted checkpoints:

| Model | PSNR | SSIM | Edge-FI@2px | IE | NIE |
| --- | ---: | ---: | ---: | ---: | ---: |
| T2exture-S | 28.9222 | 0.9576 | 0.6715 | 1.7670 | 0.006929 |
| T2exture-L | 30.5790 | 0.9634 | 0.7743 | 1.4302 | 0.005609 |
| T2exture-G | 32.8204 | 0.9705 | 0.8197 | 1.1034 | 0.004327 |

## code map

```text
model/conditioning.py       T2V adapter, FourierConv2d, Conv-P pyramid
model/t2texture_base.py     shared AMT wrapper
model/t2texture_amt_s.py    T2exture-S
model/t2texture_amt.py      T2exture-L
model/t2texture_amt_g.py    T2exture-G
data.py                     synthetic dataset loader
infer.py                    synthetic and real inference
train.py                    two-stage training
eval.py                     synthetic evaluation
vis.py                      qualitative visualization
data_preparation/           dataset conversion helper
flow_generation/            pseudo-flow generation helper
```

Large files stay out of Git: `datasets/`, `outputs/`, `pretrained/*.pt`,
`pretrained/*.pth`, and `third_party/`.
