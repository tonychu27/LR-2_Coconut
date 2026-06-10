# LR-2 Coconut with Qwen3

Setup and run instructions for training and evaluating GSM8K models with
`Qwen/Qwen3-0.6B-Base`.

## Setup

Create an environment and install the project:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The environment should already have a CUDA-enabled PyTorch build. If vLLM or
Triton needs to compile kernels, install build tools:

```bash
apt-get update && apt-get install -y build-essential
```

## Scripts

- `scripts/train_cot.py`: train a CoT LoRA adapter.
- `scripts/train_coconut.py`: train a Coconut LoRA adapter.
- `scripts/evaluate.py`: evaluate original models or LoRA adapters.
- `scripts/run_lora.sh`: launch CoT and Coconut LoRA training.
- `scripts/run_evaluate_original.sh`: evaluate original Qwen.
- `scripts/run_evaluate_lora.sh`: evaluate one LoRA checkpoint.

## Train

Train both LoRA adapters:

```bash
bash scripts/run_lora.sh
```

Default outputs:

- Coconut LoRA: `runs/gsm8k_coconut_lora/final`
- CoT LoRA: `runs/gsm8k_cot_lora/final`

Useful overrides:

```bash
MODEL=Qwen/Qwen3-0.6B-Base \
LATENT_STEPS=2 \
COCONUT_GPU=0 \
COT_GPU=1 \
LR=5e-5 \
bash scripts/run_lora.sh
```

## Evaluate Original Model

Run all original-model evaluations:

```bash
bash scripts/run_evaluate_original.sh
```

This writes:

- `runs/original_direct_predictions.csv/json`
- `runs/original_cot_predictions.csv/json`
- `runs/original_coconut_predictions.csv/json`

## Evaluate LoRA Checkpoints

Run the LoRA evaluation helper:

```bash
bash scripts/run_evaluate_lora.sh
```

By default this evaluates:

- CoT LoRA: `$COT_DIR/final`
- Coconut LoRA: `$COCONUT_DIR/stage1`

Default outputs:

- `runs/cot_lora_predictions.csv/json`
- `runs/coconut_lora_predictions.csv/json`

Useful overrides:

```bash
MODEL=Qwen/Qwen3-0.6B-Base \
LATENT_STEPS=2 \
COCONUT_DIR=runs/gsm8k_coconut_lora \
COT_DIR=runs/gsm8k_cot_lora \
COCONUT_OUTPUT=runs/coconut_lora_predictions.csv \
COT_OUTPUT=runs/cot_lora_predictions.csv \
bash scripts/run_evaluate_lora.sh
```

To evaluate a different Coconut checkpoint manually:

```bash
python scripts/evaluate.py \
  --mode coconut \
  --model Qwen/Qwen3-0.6B-Base \
  --adapter runs/gsm8k_coconut_lora/stage_0 \
  --backend hf \
  --latent-steps 2 \
  --eval-style qwen_report \
  --num-fewshot 4 \
  --max-new-tokens 512 \
  --output runs/coconut_lora_predictions_stage_0.csv
```

CoT LoRA can be evaluated manually with:

```bash
python scripts/evaluate.py \
  --mode cot \
  --model Qwen/Qwen3-0.6B-Base \
  --adapter runs/gsm8k_cot_lora/final \
  --backend vllm \
  --eval-style qwen_report \
  --num-fewshot 4 \
  --max-new-tokens 512 \
  --output runs/cot_lora_predictions.csv
```

## Outputs

Evaluation writes a CSV file plus a JSON summary.

CSV columns:

- `idx`
- `gold`
- `pred`
- `correct`
- `text`

The JSON summary includes the mode, model, adapter, backend, number correct,
total examples, and accuracy.

## Single-Example Overfit

Run the Coconut one-example overfit:

```bash
python scripts/overfit_one.py \
  --model Qwen/Qwen3-0.6B-Base \
  --output-dir runs/one_example \
  --latent-steps 2 \
  --max-steps 300 \
  --lr 2e-4 \
  --lora
```

This writes:

- `runs/one_example/loss_curve.csv`
- `runs/one_example/loss_curve.png`
- `runs/one_example/prediction.txt`

## Notes
The default model is `Qwen/Qwen3-0.6B`. To use a different Qwen model, pass `--model` or set `MODEL=...` in the helper scripts.
