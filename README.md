# T2exture

T2exture reconstructs dense source-conditioned thermal texture from sparse active LWIR keyframes and dense passive observations.

The repo uses the two-stage pipeline from the paper:

```text
Stage 1: estimate S_hat^off at active keyframes.
Stage 2: propagate texture anchors with centered passive context C_t.
```

In the paper notation, a texture frame is the nonnegative residual:

```text
X_k = [S_k^on - S_k^off]_+
```

For synthetic data, `sim/<scene>/texture/*.npy` stores the target residual `X`. A prepared root may also contain `sim/<scene>/source_on/*.npy`; when present, the loader constructs active anchors from the exact source-on frame. For real data, provide either precomputed `texture/` residual frames or `source_on/` plus `source_off/` frames so `infer.py` can construct the residual anchors.

## Setup

From a checked-out repository:

```bash
cd T2exture

conda env create -f environment.yaml
conda activate t2exture
```

Place the official AMT source tree at:

```text
third_party/AMT_official/
```

The code expects files such as `third_party/AMT_official/networks/AMT-L.py`.

Place the released T2exture checkpoints at:

```text
pretrained/t2exture_model/t2exture-s.pt
pretrained/t2exture_model/t2exture-l.pt
pretrained/t2exture_model/t2exture-g.pt
```

These full checkpoints are used by `infer.py` and `eval.py`. Stage 1 and training from scratch also require the official AMT initialization checkpoints:

```text
pretrained/amt-s.pth
pretrained/amt-l.pth
pretrained/amt-g.pth
```

Data layout:

```text
datasets/
  train.txt
  valid.txt
  test.txt
  dataset_manifest.json
  sim/<scene>/
    texture/001.npy     # source-conditioned texture X
    passive/001.npy     # source-off passive state S^off
    source_on/001.npy   # optional raw S^on, used for exact Eq. (8) anchors
  source_off/amt-s/<scene>/001.npy
  source_off/amt-l/<scene>/001.npy
  source_off/amt-g/<scene>/001.npy
  flow/s10/<scene>/001_002_011.npz
  real/<sequence>/001.png
```

Check the data and config before running experiments:

```bash
python -B scripts/preflight.py --data-root datasets --config train.yaml
python -B scripts/preflight.py --data-root datasets --config train.yaml --require-source-off
```

Formal S/L/G training, evaluation, and synthetic inference require the matching Stage 1 cache under `datasets/source_off/amt-s`, `datasets/source_off/amt-l`, or `datasets/source_off/amt-g`. If these caches are not included with the dataset, generate them with `stage1.py` before running those commands. The loader keeps a cache-free fallback only for small local interface checks.

## Method-aligned context

The default structural context is centered on the target time:

```text
C_t = [S_{t-2}^off, S_{t-1}^off, S_t^off, S_{t+1}^off, S_{t+2}^off]
```

So `train.yaml` uses:

```yaml
passive_context: 5
active_stride: 10
```

`passive_context: 1` means only the target-time passive observation. `passive_context: 0` disables passive guidance for ablations. Enabled context sizes must be odd.

## Stage 1

Stage 1 estimates the missing source-off passive state at active keyframes:

```text
S_hat_k^off = AMT(S_{k-1}^off, S_{k+1}^off, t=0.5)
```

Generate source-off caches for all three AMT scales:

```bash
python -B stage1.py \
  --mode synthetic \
  --data-root datasets \
  --backbone amt-s \
  --pretrained pretrained/amt-s.pth \
  --output-dir datasets/source_off/amt-s

python -B stage1.py \
  --mode synthetic \
  --data-root datasets \
  --backbone amt-l \
  --pretrained pretrained/amt-l.pth \
  --output-dir datasets/source_off/amt-l

python -B stage1.py \
  --mode synthetic \
  --data-root datasets \
  --backbone amt-g \
  --pretrained pretrained/amt-g.pth \
  --output-dir datasets/source_off/amt-g
```

Stage 2 reads these caches when it builds texture anchors and the centered source-off context. Paper-aligned training and evaluation should always pass the matching cache with `--source-off-root`. The synthetic loader can run without this cache for quick code checks, but that path is only a convenience mode and should not be used for reporting the main method.

For real sequences stored in `datasets/real`, generate a matching cache before real inference:

```bash
python -B stage1.py \
  --mode real \
  --data-root datasets/real \
  --real-config configs/real.yaml \
  --sequences bag \
  --backbone amt-l \
  --pretrained pretrained/amt-l.pth \
  --output-dir datasets/real/source_off/amt-l
```

## Train

Use `train.py` for reproduction. It is the top-level runner that trains, evaluates, and visualizes T2exture-S/L/G with matched settings:

```bash
python -B train.py \
  --variants s l g \
  --data-root datasets \
  --source-off-root datasets/source_off \
  --output-root outputs/final/ours
```

Train one variant:

```bash
python -B train.py \
  --variants l \
  --data-root datasets \
  --source-off-root datasets/source_off \
  --output-root outputs/final/ours \
  --config train.yaml
```

Each run writes `best.pt`, `last.pt`, and `config.json`.

`stage2.py` is the single-backbone Stage 2 trainer. `train.py` calls it once per selected variant. Most users should run `train.py`; call `stage2.py` directly only for a custom single-backbone run.

## Inference

Run synthetic smoke inference with a local checkpoint:

```bash
python -B infer.py \
  --mode synthetic \
  --variant l \
  --checkpoint pretrained/t2exture_model/t2exture-l.pt \
  --data-root datasets \
  --source-off-root datasets/source_off/amt-l \
  --split test \
  --max-samples 2 \
  --output-dir outputs/infer/synthetic-l
```

Remove `--max-samples` to run the full split.

For method-aligned synthetic inference or evaluation, also provide the Stage 1 source-off cache for the selected backbone:

```bash
python -B infer.py \
  --mode synthetic \
  --variant l \
  --checkpoint pretrained/t2exture_model/t2exture-l.pt \
  --data-root datasets \
  --source-off-root datasets/source_off/amt-l \
  --split test \
  --max-samples 2 \
  --output-dir outputs/infer/synthetic-l-sourceoff
```

Real data can use one of these input layouts:

```text
real/<sequence>/texture/001.png       # precomputed residual X
real/<sequence>/passive/001.png       # source-off/passive S^off

real/<sequence>/source_on/001.png     # source-on active frame
real/<sequence>/source_off/001.png    # source-off/passive frame in the sequence folder
```

When `source_on/` and `source_off/` are supplied, the implementation computes `[S^on - S^off]_+` in their shared input scale and only then normalizes the residual. PNG inputs use the same `[0, 1]` scale as Stage 1 caches; NumPy inputs must already use a matching physical scale.

Then run:

```bash
python -B infer.py \
  --mode real \
  --variant l \
  --checkpoint pretrained/t2exture_model/t2exture-l.pt \
  --data-root datasets/real \
  --source-off-root datasets/real/source_off/amt-l \
  --sequences bag \
  --max-samples 2 \
  --output-dir outputs/infer/real-l-bag
```

If Stage 1 generated `datasets/real/source_off/amt-l/<sequence>/*.npy`, keep that cache outside the sequence folder and pass it through `--source-off-root` as shown above.

## Evaluate

```bash
python -B eval.py \
  --data-root datasets \
  --checkpoint pretrained/t2exture_model/t2exture-l.pt \
  --backbone amt-l \
  --source-off-root datasets/source_off/amt-l \
  --split test \
  --output-dir outputs/eval/t2exture-l \
  --config train.yaml
```

Qualitative sheets and videos:

```bash
python -B vis.py \
  --eval-dir outputs/eval/t2exture-l \
  --output-dir outputs/eval/t2exture-l/vis \
  --method-label T2exture-L \
  --max-frames-per-scene 5
```

`vis.py` writes comparison PNGs and MP4 videos with an embedded caption and footnote. The caption records the model, scene, target frame, active anchor IDs, and normalized time; the footnote explains the panel order and error display.

## Code map

```text
config.py                    shared runtime defaults and context indexing
data.py                      synthetic texture/context/flow dataset
stage1.py                    Stage 1 source-off cache generation
stage2.py                    train one Stage 2 backbone
train.py                     reproduce S/L/G by calling stage2.py, eval.py, and vis.py
eval.py                      full-resolution synthetic evaluation
infer.py                     synthetic and real inference
model/conditioning.py        T2V adapter, Fourier features, Conv-P pyramid
model/t2texture_base.py      shared AMT wrapper for S/L/G
model/t2texture_amt_s.py     T2exture-S wrapper
model/t2texture_amt_l.py     T2exture-L wrapper
model/t2texture_amt_g.py     T2exture-G wrapper
scripts/                     dataset checks
flow_generation/             pseudo-flow supervision utilities
data_preparation/            dataset conversion utilities
```

Large artifacts stay out of Git: `datasets/`, `outputs/`, `pretrained/*.pt`, `pretrained/*.pth`, and `third_party/`.
