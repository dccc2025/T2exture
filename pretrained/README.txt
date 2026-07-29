# Local checkpoint cache

Checkpoint files are not tracked in Git.

T2exture checkpoints:

```text
pretrained/t2exture_model/t2exture-s.pt
pretrained/t2exture_model/t2exture-l.pt
pretrained/t2exture_model/t2exture-g.pt
```

These full checkpoints are used directly by `infer.py` and `eval.py`.

Training from scratch uses official AMT initialization checkpoints:

```text
pretrained/amt-s.pth
pretrained/amt-l.pth
pretrained/amt-g.pth
```

AMT source should be placed at:

```text
third_party/AMT_official/
```

LiteFlowNet is only needed if you regenerate pseudo-flow supervision:

```text
pretrained/LiteFlowNet.pth
```
