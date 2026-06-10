#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p runs

MODEL="${MODEL:-Qwen/Qwen3-0.6B-Base}"
LATENT_STEPS="${LATENT_STEPS:-2}"
COCONUT_DIR="${COCONUT_DIR:-runs/gsm8k_coconut_lora}"
COT_DIR="${COT_DIR:-runs/gsm8k_cot_lora}"
COCONUT_GPU="${COCONUT_GPU:-0}"
COT_GPU="${COT_GPU:-1}"
COCONUT_BATCH_SIZE="${COCONUT_BATCH_SIZE:-1}"
COCONUT_GRAD_ACCUM_STEPS="${COCONUT_GRAD_ACCUM_STEPS:-32}"
COT_BATCH_SIZE="${COT_BATCH_SIZE:-8}"
COT_GRAD_ACCUM_STEPS="${COT_GRAD_ACCUM_STEPS:-4}"
TRAIN_STYLE="${TRAIN_STYLE:-qwen_report}"
NUM_FEWSHOT="${NUM_FEWSHOT:-4}"
LR="${LR:-5e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
COCONUT_EPOCHS_PER_STAGE="${COCONUT_EPOCHS_PER_STAGE:-1}"
COCONUT_MAX_STAGES="${COCONUT_MAX_STAGES:-3}"
COT_EPOCHS="${COT_EPOCHS:-3}"
COCONUT_LOG="${COCONUT_LOG:-runs/coconut_train.log}"
COT_LOG="${COT_LOG:-runs/cot_train.log}"

echo "[train] Starting Coconut LoRA on GPU $COCONUT_GPU"
CUDA_VISIBLE_DEVICES="$COCONUT_GPU" python scripts/train_coconut.py \
  --model "$MODEL" \
  --output-dir "$COCONUT_DIR" \
  --latent-steps "$LATENT_STEPS" \
  --epochs-per-stage "$COCONUT_EPOCHS_PER_STAGE" \
  --max-stages "$COCONUT_MAX_STAGES" \
  --batch-size "$COCONUT_BATCH_SIZE" \
  --grad-accum-steps "$COCONUT_GRAD_ACCUM_STEPS" \
  --train-style "$TRAIN_STYLE" \
  --num-fewshot "$NUM_FEWSHOT" \
  --lr "$LR" \
  --warmup-ratio "$WARMUP_RATIO" \
  --min-lr-ratio "$MIN_LR_RATIO" > "$COCONUT_LOG" 2>&1 &
coconut_pid=$!

echo "[train] Starting CoT LoRA on GPU $COT_GPU with batch size $COT_BATCH_SIZE"
CUDA_VISIBLE_DEVICES="$COT_GPU" python scripts/train_cot.py \
  --model "$MODEL" \
  --output-dir "$COT_DIR" \
  --epochs "$COT_EPOCHS" \
  --batch-size "$COT_BATCH_SIZE" \
  --grad-accum-steps "$COT_GRAD_ACCUM_STEPS" \
  --train-style "$TRAIN_STYLE" \
  --num-fewshot "$NUM_FEWSHOT" \
  --lr "$LR" \
  --warmup-ratio "$WARMUP_RATIO" \
  --min-lr-ratio "$MIN_LR_RATIO" > "$COT_LOG" 2>&1 &
cot_pid=$!

echo "[train] Logs:"
echo "  Coconut: tail -f $COCONUT_LOG"
echo "  CoT:     tail -f $COT_LOG"

status=0
if ! wait "$coconut_pid"; then
  echo "[train] Coconut failed; see $COCONUT_LOG" >&2
  status=1
fi
if ! wait "$cot_pid"; then
  echo "[train] CoT failed; see $COT_LOG" >&2
  status=1
fi

if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

echo "[train] Finished base-model LoRA training jobs"