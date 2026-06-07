#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p runs

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
LATENT_STEPS="${LATENT_STEPS:-2}"
COCONUT_DIR="${COCONUT_DIR:-runs/gsm8k_coconut_lora}"
COT_DIR="${COT_DIR:-runs/gsm8k_cot_lora}"

echo "[eval] Starting Coconut evaluation"
python scripts/evaluate.py \
  --mode coconut \
  --model "$MODEL" \
  --adapter "$COCONUT_DIR/final" \
  --backend hf \
  --latent-steps "$LATENT_STEPS" \
  --output runs/coconut_lora_predictions.csv

echo "[eval] Starting CoT evaluation"
python scripts/evaluate.py \
  --mode cot \
  --model "$MODEL" \
  --adapter "$COT_DIR/final" \
  --backend hf \
  --output runs/cot_lora_predictions.csv

echo "=== Coconut summary ==="
cat runs/coconut_lora_predictions.json


echo "=== CoT summary ==="
cat runs/cot_lora_predictions.json
