# LR-2 Coconut: latent reasoning with Qwen3

This repo is a compact reproduction of Coconut, or Chain of Continuous Thought, for GSM8K using `Qwen/Qwen3-0.6B` by default. The important part is the mechanism: after a `<bot>` marker, the model feeds the last-layer hidden state back as the next input embedding for `k` latent reasoning steps, then emits `<eot>` and decodes the final answer.

The code is intentionally small and auditable. The training path recomputes each latent prefix sequentially so gradients flow through the fed-back hidden states. That is slower than a cache-heavy implementation, but it makes the single-example overfit test straightforward.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Single-example overfit

This is the required proof target for the assignment.

```bash
python scripts/overfit_one.py \
  --model Qwen/Qwen3-0.6B \
  --output-dir runs/one_example \
  --latent-steps 2 \
  --max-steps 300 \
  --lr 2e-4 \
  --lora
```

The script writes:

- `runs/one_example/loss_curve.csv`
- `runs/one_example/loss_curve.png`
- `runs/one_example/prediction.txt`
- adapter/model artifacts when `--save-model` is passed

For a very small smoke run without committing to an overfit:

```bash
python scripts/overfit_one.py --max-steps 3 --lora --output-dir runs/smoke
```

## Full GSM8K training

The curriculum progressively moves reasoning text from language space into latent space.

```bash
python scripts/train_gsm8k.py \
  --model Qwen/Qwen3-0.6B \
  --output-dir runs/gsm8k_coconut \
  --latent-steps 2 \
  --epochs-per-stage 1 \
  --max-stages 4 \
  --lora
```

Evaluation:

```bash
python scripts/evaluate_gsm8k.py \
  --model Qwen/Qwen3-0.6B \
  --adapter runs/gsm8k_coconut/final \
  --latent-steps 2 \
  --limit 200
```

## Files

- `src/coconut_qwen/modeling.py` implements hidden-state-as-next-embedding feedback.
- `src/coconut_qwen/data.py` formats GSM8K and curriculum stages.
- `scripts/overfit_one.py` trains on one GSM8K example and saves evidence.
- `scripts/train_gsm8k.py` runs the staged curriculum.
- `scripts/evaluate_gsm8k.py` generates answers and computes exact numeric accuracy.
- `report.html` is the self-contained report.

## Notes

The default base model is `Qwen/Qwen3-0.6B`; using `Qwen/Qwen3-1.7B` only requires changing `--model`. The implementation adds `<bot>`, `<eot>`, and `<latent>` special tokens and resizes the embedding table.
