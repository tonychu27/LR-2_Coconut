#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coconut_qwen.data import load_gsm8k_examples, prompt_for_question
from coconut_qwen.modeling import add_coconut_tokens, load_tokenizer_and_model
from coconut_qwen.train_utils import maybe_enable_lora


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--output-dir", default="runs/gsm8k_cot_baseline")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora", action="store_true")
    return p.parse_args()


def encode_baseline(ex, tokenizer):
    prompt = prompt_for_question(ex.question)
    target = "\n".join(ex.reasoning_steps) + f"\n#### {ex.final_answer}"
    prefix_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    target_ids = tokenizer(target, add_special_tokens=False).input_ids
    input_ids = torch.tensor([prefix_ids + target_ids], dtype=torch.long)
    labels = torch.tensor([[-100] * len(prefix_ids) + target_ids], dtype=torch.long)
    return input_ids, labels


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_tokenizer_and_model(args.model)
    add_coconut_tokens(tokenizer, model)
    model = maybe_enable_lora(model, args.lora)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    examples = load_gsm8k_examples("train", limit=args.limit)

    log_path = out_dir / "train_log.csv"
    log_path.write_text("global_step,loss\n", encoding="utf-8")
    step = 0
    for _ in range(args.epochs):
        for ex in tqdm(examples, desc="cot", mininterval=30):
            input_ids, labels = encode_baseline(ex, tokenizer)
            input_ids = input_ids.to(model.device)
            labels = labels.to(model.device)
            optim.zero_grad(set_to_none=True)
            out = model(input_ids=input_ids, labels=labels, use_cache=False)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            step += 1
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"{step},{float(loss.detach().cpu()):.6f}\n")

    final_dir = out_dir / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"saved {final_dir}")


if __name__ == "__main__":
    main()
