#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p runs

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
LATENT_STEPS="${LATENT_STEPS:-2}"
COCONUT_DIR="${COCONUT_DIR:-runs/gsm8k_coconut_lora}"
COT_DIR="${COT_DIR:-runs/gsm8k_cot_lora}"

echo "[setup] Installing package and dependencies"
python -m pip install -e ".[dev]"

echo "[train] Starting Coconut on"
python scripts/train_coconut.py \
  --model "$MODEL" \
  --output-dir "$COCONUT_DIR" \
  --latent-steps "$LATENT_STEPS" \
  --epochs-per-stage 1 \
  --max-stages 2 \
  --lr 2e-4 \
  --lora > runs/coconut_train.log 2>&1

echo "[train] Starting CoT baseline"
python scripts/train_cot.py \
  --model "$MODEL" \
  --output-dir "$COT_DIR" \
  --epochs 1 \
  --lr 2e-4 \
  --lora > runs/cot_train.log 2>&1

echo "[train] Logs:"
echo "  tail -f runs/coconut_train.log"
echo "  tail -f runs/cot_train.log"
