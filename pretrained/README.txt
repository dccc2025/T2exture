# Pretrained And Release Checkpoints

This directory is a local cache for model checkpoints. Checkpoint files are not
tracked in Git. Download the required weights from the official project pages
or from the release links provided with the paper, then place them at the paths
below.

## Released T2exture Weights

The public T2exture-S/L/G checkpoints are hosted at:

https://huggingface.co/chenjiashuo/T2exture_model

Recommended local cache paths:

```text
pretrained/t2exture_model/t2exture-s.pt
pretrained/t2exture_model/t2exture-l.pt
pretrained/t2exture_model/t2exture-g.pt
```

These checkpoints contain the full T2exture model state. They can be used by
`infer.py` and `eval.py` without separate AMT initialization files.

## AMT Initialization Weights

AMT checkpoints are used to initialize the T2exture AMT-S, AMT-L, and AMT-G
models during training.

```text
pretrained/amt-s.pth
pretrained/amt-l.pth
pretrained/amt-g.pth
```

Official AMT repository:
https://github.com/MCG-NKU/AMT

## Pseudo-Flow Teacher

LiteFlowNet is used only for generating cached pseudo optical flow supervision.

```text
pretrained/LiteFlowNet.pth
```

Reference implementation:
https://github.com/sniklaus/pytorch-liteflownet

## Optional Baseline Checkpoints

These checkpoints are needed only when reproducing baseline comparisons.

```text
pretrained/IFRNet/IFRNet_Vimeo90K.pth
pretrained/IFRNet/IFRNet_GoPro.pth
pretrained/GIMM-VFI-F/flowformer_sintel.pth
pretrained/GIMM-VFI-F/gimm.pt
pretrained/GIMM-VFI-F/gimmvfi_f_arb.pt
pretrained/sgm-vfi/
pretrained/bim-vfi/
```

Baseline project pages:
- IFRNet: https://github.com/ltkong218/IFRNet
- GIMM-VFI: https://github.com/GSeanCDAT/GIMM-VFI
- SGM-VFI: https://github.com/MCG-NJU/SGM-VFI
- BiM-VFI: https://github.com/KAIST-VICLab/BiM-VFI

## Notes

- Keep all checkpoint files out of Git.
- Keep third-party source code under `third_party/`, not inside this directory.
- If a checkpoint path is changed, update the corresponding command examples.
