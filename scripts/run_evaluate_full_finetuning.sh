#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p runs

LATENT_STEPS="${LATENT_STEPS:-2}"
COCONUT_DIR="${COCONUT_DIR:-runs/gsm8k_coconut_fullft}"
COT_DIR="${COT_DIR:-runs/gsm8k_cot_fullft}"

echo "[eval] Full fine-tuned Coconut checkpoint"
python scripts/evaluate.py \
  --mode coconut \
  --model "$COCONUT_DIR/final" \
  --backend hf \
  --latent-steps "$LATENT_STEPS" \
  --output runs/coconut_full_finetuning_predictions.csv

echo "[eval] Full fine-tuned CoT checkpoint"
python scripts/evaluate.py \
  --mode cot \
  --model "$COT_DIR/final" \
  --backend vllm \
  --output runs/cot_full_finetuning_predictions.csv


echo "=== Full fine-tuned Coconut summary ==="
cat runs/coconut_full_finetuning_predictions.json

echo
echo "=== Full fine-tuned CoT summary ==="
cat runs/cot_full_finetuning_predictions.json
