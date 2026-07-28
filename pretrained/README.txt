# Local checkpoint cache

Checkpoint files are not tracked in Git.

Hosted T2exture checkpoints:

https://huggingface.co/chenjiashuo/T2exture_model

Recommended local paths:

```text
pretrained/t2exture_model/t2exture-s.pt
pretrained/t2exture_model/t2exture-l.pt
pretrained/t2exture_model/t2exture-g.pt
```

These files contain the full T2exture model state and can be used directly by
`infer.py` and `eval.py`.

Training from scratch uses official AMT initialization checkpoints:

```text
pretrained/amt-s.pth
pretrained/amt-l.pth
pretrained/amt-g.pth
```

AMT source:

https://github.com/MCG-NKU/AMT

LiteFlowNet is only needed if you regenerate pseudo-flow supervision:

```text
pretrained/LiteFlowNet.pth
```

Reference:

https://github.com/sniklaus/pytorch-liteflownet
