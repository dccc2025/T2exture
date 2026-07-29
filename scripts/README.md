# Scripts

This directory contains auxiliary checks for the main T2exture pipeline.

```text
preflight.py    validate split files, frame layout, context windows, and flow files
```

Run before training or evaluation:

```bash
python -B scripts/preflight.py --data-root datasets --config train.yaml
```

For the full paper protocol, also verify the Stage 1 source-off caches:

```bash
python -B scripts/preflight.py --data-root datasets --config train.yaml --require-source-off
```
