#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p runs

ORIGINAL_MODEL="${ORIGINAL_MODEL:-Qwen/Qwen3-0.6B}"
LATENT_STEPS="${LATENT_STEPS:-2}"

echo "[eval] Original unmodified Qwen3-0.6B with Coconut latent prompt"
python scripts/evaluate.py \
  --mode coconut \
  --model "$ORIGINAL_MODEL" \
  --backend hf \
  --latent-steps "$LATENT_STEPS" \
  --output runs/original_coconut_predictions.csv

echo "[eval] Original unmodified Qwen3-0.6B with CoT prompt"
python scripts/evaluate.py \
  --mode cot \
  --model "$ORIGINAL_MODEL" \
  --backend vllm \
  --output runs/original_cot_predictions.csv

echo "[eval] Original unmodified Qwen3-0.6B with Direct prompt"
python scripts/evaluate.py \
  --mode direct \
  --model "$ORIGINAL_MODEL" \
  --backend vllm \
  --output runs/original_direct_predictions.csv

echo "=== Original Qwen3-0.6B Coconut summary ==="
cat runs/original_coconut_predictions.json

echo "=== Original Qwen3-0.6B CoT summary ==="
cat runs/original_cot_predictions.json

echo "=== Original Qwen3-0.6B Direct summary ==="
cat runs/original_direct_predictions.json
