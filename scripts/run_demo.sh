#!/usr/bin/env bash
set -euo pipefail

python -m fedgagan make-demo-data --output data/raw/demo.csv --rows 2400
python -m fedgagan run --config configs/demo.yaml \
  --stages preprocess,train,generate,evaluate

