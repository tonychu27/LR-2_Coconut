# LR-2 Coconut with Qwen3

This repo trains and evaluates GSM8K models with `Qwen/Qwen3-0.6B`.
It supports three evaluation styles:

- `direct`: ask for only the final answer.
- `cot`: chain-of-thought text reasoning.
- `coconut`: latent reasoning with continuous hidden states after `<bot>`.

It supports three model types:

- original unmodified `Qwen/Qwen3-0.6B`
- LoRA adapters
- full fine-tuned checkpoints

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The environment must already have a CUDA-enabled PyTorch build. vLLM is used for fast evaluation of original/full-fine-tuned text models, so it may also need a system compiler for Triton:

```bash
apt-get update && apt-get install -y build-essential
```

## Main Scripts

- `scripts/train_cot.py`: train the CoT baseline.
- `scripts/train_coconut.py`: train Coconut latent reasoning.
- `scripts/evaluate.py`: unified evaluator for original, LoRA, and full fine-tuned models.
- `scripts/run_lora.sh`: train LoRA Coconut and LoRA CoT.
- `scripts/run_full_finetuning.sh`: train full fine-tuned Coconut and CoT.
- `scripts/run_evaluate_lora.sh`: evaluate LoRA models.
- `scripts/run_evaluate_full_finetuning.sh`: evaluate full fine-tuned models.
- `scripts/run_evaluate_original.sh`: evaluate the original model.

## Evaluation Backends

Backend policy:

- LoRA evaluation uses Hugging Face (`--backend hf`) for correctness.
- Coconut evaluation uses Hugging Face because it needs hidden-state feedback.
- Original and full fine-tuned `direct`/`cot` evaluation can use vLLM (`--backend vllm`).

`scripts/evaluate.py --backend auto` follows that policy automatically.

## Train LoRA Models

```bash
bash scripts/run_lora.sh
```

Defaults:

- Coconut LoRA output: `runs/gsm8k_coconut_lora/final`
- CoT LoRA output: `runs/gsm8k_cot_lora/final`

You can override defaults:

```bash
MODEL=Qwen/Qwen3-0.6B LATENT_STEPS=2 bash scripts/run_lora.sh
```

## Train Full Fine-Tuned Models

```bash
bash scripts/run_full_finetuning.sh
```

Defaults:

- Coconut full fine-tune output: `runs/gsm8k_coconut_fullft/final`
- CoT full fine-tune output: `runs/gsm8k_cot_fullft/final`

Useful overrides:

```bash
COCONUT_EPOCHS_PER_STAGE=2 \
COCONUT_MAX_STAGES=4 \
COT_EPOCHS=8 \
LR=5e-5 \
bash scripts/run_full_finetuning.sh
```

## Evaluate Original Model

Original model means unmodified `Qwen/Qwen3-0.6B`.

Direct answer with vLLM:

```bash
python scripts/evaluate.py \
  --mode direct \
  --model Qwen/Qwen3-0.6B \
  --backend vllm \
  --output runs/original_direct_predictions.csv
```

CoT with vLLM:

```bash
python scripts/evaluate.py \
  --mode cot \
  --model Qwen/Qwen3-0.6B \
  --backend vllm \
  --output runs/original_cot_predictions.csv
```

Coconut-style original-model evaluation with Hugging Face:

```bash
python scripts/evaluate.py \
  --mode coconut \
  --model Qwen/Qwen3-0.6B \
  --backend hf \
  --latent-steps 2 \
  --output runs/original_coconut_predictions.csv
```

The helper script currently runs the original Coconut evaluation:

```bash
bash scripts/run_evaluate_original.sh
```

## Evaluate LoRA Models

CoT LoRA with Hugging Face:

```bash
python scripts/evaluate.py \
  --mode cot \
  --model Qwen/Qwen3-0.6B \
  --adapter runs/gsm8k_cot_lora/final \
  --backend hf \
  --output runs/cot_lora_predictions.csv
```

Coconut LoRA with Hugging Face:

```bash
python scripts/evaluate.py \
  --mode coconut \
  --model Qwen/Qwen3-0.6B \
  --adapter runs/gsm8k_coconut_lora/final \
  --backend hf \
  --latent-steps 2 \
  --output runs/coconut_lora_predictions.csv
```

Or run the LoRA evaluation helper:

```bash
bash scripts/run_evaluate_lora.sh
```

## Evaluate Full Fine-Tuned Models

CoT full fine-tuned checkpoint with vLLM:

```bash
python scripts/evaluate.py \
  --mode cot \
  --model runs/gsm8k_cot_fullft/final \
  --backend vllm \
  --output runs/cot_full_finetuning_predictions.csv
```

Coconut full fine-tuned checkpoint with Hugging Face:

```bash
python scripts/evaluate.py \
  --mode coconut \
  --model runs/gsm8k_coconut_fullft/final \
  --backend hf \
  --latent-steps 2 \
  --output runs/coconut_full_finetuning_predictions.csv
```

Or run:

```bash
bash scripts/run_evaluate_full_finetuning.sh
```

## Outputs

Prediction results are written as CSV plus JSON summary files. Examples:

- `runs/cot_lora_predictions.csv`
- `runs/cot_lora_predictions.json`
- `runs/cot_full_finetuning_predictions.csv`
- `runs/original_direct_predictions.json`

CSV columns:

- `idx`
- `gold`
- `pred`
- `correct`
- `text`

The JSON file includes accuracy, total examples, model path, adapter path, and backend.

## Single-Example Overfit

For a quick Coconut proof target:

```bash
python scripts/overfit_one.py \
  --model Qwen/Qwen3-0.6B \
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

`scripts/evalatuon.py` is only a compatibility wrapper for the misspelled old filename. Prefer `scripts/evaluate.py`.

The default model is `Qwen/Qwen3-0.6B`. To use a different Qwen model, pass `--model` or set `MODEL=...` in the helper scripts.
