# T2exture

T2exture interpolates texture frames in active/passive thermal sequences. The
code follows the main-figure architecture: a T2V adapter maps one-channel
texture anchors into AMT's RGB input space, FourierConv2d relative positional
encoding injects the target time, and a three-level Conv-P passive-guidance
pyramid is added to the AMT decoder before multi-field refinement.

This branch is the public reproduction path for T2exture-S, T2exture-L, and
T2exture-G.

## Quick Start

```bash
git clone -b 0725_code_cjs https://github.com/dccc2025/T2exture.git
cd T2exture

conda env create -f environment.yaml
conda activate t2exture
```

Install the official AMT source once:

```bash
git clone https://github.com/MCG-NKU/AMT third_party/AMT_official
```

Download the prepared dataset from Hugging Face:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='chenjiashuo/T2exture_datasets', repo_type='dataset', local_dir='datasets')"
```

Run one synthetic test-set inference pass. If `--checkpoint` is omitted,
`infer.py` downloads the selected weight from
`chenjiashuo/T2exture_model`.

```bash
python -B infer.py \
  --mode synthetic \
  --variant l \
  --data-root datasets \
  --output-dir outputs/infer/synthetic-l
```

Run real-sequence inference:

```bash
python -B infer.py \
  --mode real \
  --variant l \
  --data-root datasets/real \
  --output-dir outputs/infer/real-l
```

Expected outputs:

```text
outputs/infer/<run>/
  predictions.csv
  manifest.json
  pred/...
```

## Model Weights

The release model repo is:

```text
https://huggingface.co/chenjiashuo/T2exture_model
```

Files:

```text
t2exture-s.pt  T2exture-S, AMT-S backbone
t2exture-l.pt  T2exture-L, AMT-L backbone
t2exture-g.pt  T2exture-G, AMT-G backbone
```

Each checkpoint stores the full T2exture model state, including the AMT
backbone. Inference and evaluation therefore do not need separate AMT
checkpoint files. Training from scratch still uses official AMT weights as
initialization.

## Dataset Format

The prepared synthetic root is `datasets/`:

```text
datasets/
  train.txt
  valid.txt
  test.txt
  dataset_manifest.json
  sim/<scene>/
    texture/001.npy
    texture/002.npy
    passive/001.npy
    passive/002.npy
  flow/s10/<scene>/001_002_011.npz
  real/<sequence>/001.png
```

Synthetic frame files are single-channel NumPy arrays. Real frames are
grayscale PNG files. The default synthetic protocol uses 640x960 ROI frames,
20 train scenes, 4 validation scenes, 8 test scenes, active stride 10, and four
passive context frames around each target.

Pseudo-flow files store target-to-endpoint supervision:

```text
flow0: [2, H, W] target-to-left-anchor flow
flow1: [2, H, W] target-to-right-anchor flow
```

Verify the dataset and protocol before training:

```bash
python -B scripts/formal/preflight.py \
  --manifest configs/formal/experiment_manifest.yaml \
  --config train.yaml
```

## Architecture

The implementation mirrors the main figure:

```text
I0, I1 -> T2V-Adapter -> AMT pretrained model -> It
      t -> FourierConv2d Rel. P.E. -> T2V-Adapter / passive guidance
  S_off -> Conv-P -> G_t^1 -> Conv-P -> G_t^2 -> Conv-P -> G_t^3
```

Code map:

```text
model/conditioning.py      T2V adapter, FourierConv2d, Conv-P pyramid
model/t2texture_base.py    shared AMT wrapper and decoder injection
model/t2texture_amt_s.py   T2exture-S channel configuration
model/t2texture_amt.py     T2exture-L channel configuration
model/t2texture_amt_g.py   T2exture-G channel configuration
```

S/L/G differ only in AMT backbone width and decoder guidance channels.

## Evaluation

Evaluate a released checkpoint on the synthetic test split:

```bash
python -B eval.py \
  --data-root datasets \
  --checkpoint pretrained/t2exture_model/t2exture-l.pt \
  --backbone amt-l \
  --split test \
  --output-dir outputs/eval/t2exture-l \
  --config train.yaml
```

The evaluator writes:

```text
pred/
err/
frame.csv
scene.csv
metrics.json
manifest.json
```

Create qualitative PNG sheets and videos:

```bash
python -B vis.py \
  --eval-dir outputs/eval/t2exture-l \
  --output-dir outputs/eval/t2exture-l/vis \
  --method-label T2exture-L \
  --max-frames-per-scene 5
```

## Training

Place official AMT initialization checkpoints under:

```text
pretrained/amt-s.pth
pretrained/amt-l.pth
pretrained/amt-g.pth
```

Train and evaluate all release variants sequentially:

```bash
python -B scripts/formal/train_release_models.py --variants s l g
```

Train one variant manually:

```bash
python -B train.py \
  --data-root datasets \
  --pretrained pretrained/amt-l.pth \
  --backbone amt-l \
  --output-dir outputs/final/release/t2exture-l \
  --config train.yaml
```

Training writes `best.pt`, `last.pt`, and `config.json`. The schedule is fixed
for the formal protocol: 10k adapter iterations, 5k rear-refinement fine-tuning
iterations, crop size 384, batch size 4, Charbonnier/CSS/pseudo-flow losses,
and full-resolution validation/test evaluation.

## Tests

```bash
python -B -m unittest tests.test_smoke -v
```

The smoke tests cover config guards, passive-context indexing, pseudo-flow path
contracts, metrics, and summary-table contracts. They do not require a full
training run.

## Repository Layout

```text
model/                 T2exture modules
losses/                training losses
metrics/               synthetic and real metrics
data.py                synthetic dataset loader
infer.py               concise public inference entry point
train.py               two-stage training
eval.py                synthetic evaluation
vis.py                 qualitative visualization
data_preparation/      source-to-prepared dataset conversion
flow_generation/       LiteFlowNet pseudo-flow generation and indexing
configs/formal/        fixed protocol configs
scripts/formal/        preflight, release training, formal runners
scripts/real/          real benchmark evaluation
pretrained/README.txt  local checkpoint placement guide
```

Large assets stay out of Git: `datasets/`, `outputs/`, `pretrained/*.pt`,
`pretrained/*.pth`, and `third_party/`.
