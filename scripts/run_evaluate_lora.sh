#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p runs

MODEL="${MODEL:-Qwen/Qwen3-0.6B-Base}"
LATENT_STEPS="${LATENT_STEPS:-2}"
COCONUT_DIR="${COCONUT_DIR:-runs/gsm8k_coconut_lora}"
COT_DIR="${COT_DIR:-runs/gsm8k_cot_lora}"
COT_OUTPUT="${COT_OUTPUT:-runs/cot_lora_predictions.csv}"
COCONUT_OUTPUT="${COCONUT_OUTPUT:-runs/coconut_lora_predictions.csv}"

echo "[eval] CoT LoRA checkpoint with Qwen-report GSM8K 4-shot CoT setting"
python scripts/evaluate.py \
  --mode cot \
  --model "$MODEL" \
  --adapter "$COT_DIR/final" \
  --backend vllm \
  --eval-style qwen_report \
  --num-fewshot 4 \
  --max-new-tokens 512 \
  --output "$COT_OUTPUT"


echo "[eval] Coconut LoRA checkpoint with Qwen-report GSM8K 4-shot CoT context"
python scripts/evaluate.py \
  --mode coconut \
  --model "$MODEL" \
  --adapter "$COCONUT_DIR/stage1" \
  --backend hf \
  --latent-steps "$LATENT_STEPS" \
  --eval-style qwen_report \
  --num-fewshot 4 \
  --max-new-tokens 512 \
  --output "$COCONUT_OUTPUT"

echo "=== Coconut LoRA Qwen-report-style summary ==="
cat "${COCONUT_OUTPUT%.csv}.json"

echo "=== CoT LoRA Qwen-report-style summary ==="
cat "${COT_OUTPUT%.csv}.json"
