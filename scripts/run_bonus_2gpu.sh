#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p runs

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
LATENT_STEPS="${LATENT_STEPS:-2}"
COCONUT_DIR="${COCONUT_DIR:-runs/gsm8k_coconut_bonus_full}"
COT_DIR="${COT_DIR:-runs/gsm8k_cot_baseline_full}"

echo "[setup] Installing package and dependencies"
python -m pip install -e ".[dev]"

echo "[train] Starting Coconut on GPU 0"
CUDA_VISIBLE_DEVICES=0 python scripts/train_gsm8k.py \
  --model "$MODEL" \
  --output-dir "$COCONUT_DIR" \
  --latent-steps "$LATENT_STEPS" \
  --epochs-per-stage 1 \
  --max-stages 2 \
  --lr 2e-4 \
  --lora > runs/coconut_train.log 2>&1 &
COCONUT_PID=$!

echo "[train] Starting CoT baseline on GPU 1"
CUDA_VISIBLE_DEVICES=1 python scripts/train_cot_baseline.py \
  --model "$MODEL" \
  --output-dir "$COT_DIR" \
  --epochs 1 \
  --lr 2e-4 \
  --lora > runs/cot_train.log 2>&1 &
COT_PID=$!

echo "[train] Logs:"
echo "  tail -f runs/coconut_train.log"
echo "  tail -f runs/cot_train.log"

wait "$COCONUT_PID"
echo "[train] Coconut finished"

wait "$COT_PID"
echo "[train] CoT baseline finished"

echo "[eval] Starting Coconut evaluation on GPU 0"
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_gsm8k.py \
  --model "$MODEL" \
  --adapter "$COCONUT_DIR/final" \
  --latent-steps "$LATENT_STEPS" \
  --output "$COCONUT_DIR/test_predictions.csv" > runs/coconut_eval.log 2>&1 &
COCONUT_EVAL_PID=$!

echo "[eval] Starting CoT baseline evaluation on GPU 1"
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_cot_baseline.py \
  --model "$MODEL" \
  --adapter "$COT_DIR/final" \
  --output "$COT_DIR/test_predictions.csv" > runs/cot_eval.log 2>&1 &
COT_EVAL_PID=$!

echo "[eval] Logs:"
echo "  tail -f runs/coconut_eval.log"
echo "  tail -f runs/cot_eval.log"

wait "$COCONUT_EVAL_PID"
echo "[eval] Coconut evaluation finished"

wait "$COT_EVAL_PID"
echo "[eval] CoT baseline evaluation finished"

echo
echo "=== Coconut summary ==="
cat "$COCONUT_DIR/test_predictions.json"

echo
echo "=== CoT baseline summary ==="
cat "$COT_DIR/test_predictions.json"
