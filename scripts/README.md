# scripts

There are two public entry points here.

Check that the downloaded dataset matches `train.yaml`:

```bash
python -B scripts/preflight.py --data-root datasets --config train.yaml
```

Train, evaluate, and visualize T2exture-S/L/G:

```bash
python -B scripts/train_ours.py --variants s l g
```

Other helpers live at the project root:

```text
infer.py   run synthetic or real inference
eval.py    compute synthetic test metrics
vis.py     make qualitative PNG sheets and videos
```
