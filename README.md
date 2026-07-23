# T2exture

AMT-L fine-tuning for texture-frame interpolation using four observed passive neighbours: two before and two after the target timestamp.

## Setup

Install PyTorch, NumPy, and PyYAML. Download `amt-l.pth` from the link in `pretrained_models/README.txt`.

Expected dataset layout:

```text
DATA_ROOT/
├── train.txt
├── valid.txt
├── test.txt
└── sim/SCENE/{texture,passive,flow}/001.npy
```

Each split contains one scene name per line. Texture/passive arrays are grayscale `H x W`; every target flow label is `4 x H x W`.

## Train

```bash
bash scripts/train.sh DATA_ROOT pretrained_models/amt-l.pth runs/experiment_01
```

`train.yaml` fixes the loss to `1.0 * Charbonnier + 0.1 * CSS (ternary census) + 0.001 * MultipleFlow`, trains adapters for 10,000 iterations, then fine-tunes AMT-L for 5,000 iterations with layer-wise LR decay.

## Test

```bash
python -m unittest tests.test_smoke -v
```
