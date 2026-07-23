#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT="$1"
PRETRAINED="$2"
OUTPUT_DIR="$3"
python train.py --data-root "$DATA_ROOT" --pretrained "$PRETRAINED" --output-dir "$OUTPUT_DIR"
