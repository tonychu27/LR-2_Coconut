#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p runs

ORIGINAL_MODEL="${ORIGINAL_MODEL:-Qwen/Qwen3-0.6B-Base}"
LATENT_STEPS="${LATENT_STEPS:-2}"

echo "[eval] Original unmodified $ORIGINAL_MODEL direct-answer setting"
python scripts/evaluate.py \
  --mode direct \
  --model "$ORIGINAL_MODEL" \
  --backend vllm \
  --max-new-tokens 64 \
  --output runs/original_direct_predictions.csv

echo "[eval] Original unmodified $ORIGINAL_MODEL CoT framework with Qwen-report GSM8K 4-shot CoT setting"
python scripts/evaluate.py \
  --mode cot \
  --model "$ORIGINAL_MODEL" \
  --backend vllm \
  --eval-style qwen_report \
  --num-fewshot 4 \
  --max-new-tokens 512 \
  --output runs/original_cot_predictions.csv

echo "[eval] Original unmodified $ORIGINAL_MODEL Coconut wrapper with Qwen-report GSM8K 4-shot CoT context"
python scripts/evaluate.py \
  --mode coconut \
  --model "$ORIGINAL_MODEL" \
  --backend hf \
  --latent-steps "$LATENT_STEPS" \
  --eval-style qwen_report \
  --num-fewshot 4 \
  --max-new-tokens 512 \
  --output runs/original_coconut_predictions.csv

echo "=== Original $ORIGINAL_MODEL direct-answer summary ==="
cat runs/original_direct_predictions.json

echo "=== Original $ORIGINAL_MODEL CoT Qwen-report-style summary ==="
cat runs/original_cot_predictions.json

echo "=== Original $ORIGINAL_MODEL Coconut Qwen-report-style summary ==="
cat runs/original_coconut_predictions.json
