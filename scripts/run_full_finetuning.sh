#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p runs

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
LATENT_STEPS="${LATENT_STEPS:-2}"
COCONUT_DIR="${COCONUT_DIR:-runs/gsm8k_coconut_fullft}"
COT_DIR="${COT_DIR:-runs/gsm8k_cot_fullft}"

COCONUT_EPOCHS_PER_STAGE="${COCONUT_EPOCHS_PER_STAGE:-2}"
COCONUT_MAX_STAGES="${COCONUT_MAX_STAGES:-4}"
COT_EPOCHS="${COT_EPOCHS:-8}"
LR="${LR:-5e-5}"
ANSWER_ONLY_LOSS="${ANSWER_ONLY_LOSS:-0}"

COCONUT_EXTRA_ARGS=()
if [[ "$ANSWER_ONLY_LOSS" == "1" ]]; then
  COCONUT_EXTRA_ARGS+=(--answer-only-loss)
fi

echo "[train] Starting full fine-tune Coconut"
python scripts/train_coconut.py \
  --model "$MODEL" \
  --output-dir "$COCONUT_DIR" \
  --latent-steps "$LATENT_STEPS" \
  --epochs-per-stage "$COCONUT_EPOCHS_PER_STAGE" \
  --max-stages "$COCONUT_MAX_STAGES" \
  --lr "$LR" \
  "${COCONUT_EXTRA_ARGS[@]}" > runs/coconut_full_finetuning_train.log 2>&1

echo "[train] Starting full fine-tune CoT"
python scripts/train_cot.py \
  --model "$MODEL" \
  --output-dir "$COT_DIR" \
  --epochs "$COT_EPOCHS" \
  --lr "$LR" > runs/cot_full_finetuning_train.log 2>&1

echo "[train] Logs:"
echo "  tail -f runs/coconut_full_finetuning_train.log"
echo "  tail -f runs/cot_full_finetuning_train.log"
