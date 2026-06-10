#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coconut_qwen.data import (
    load_gsm8k_examples,
    prompt_for_question,
    strip_gsm8k_calculator_annotations,
)
from coconut_qwen.modeling import add_coconut_tokens, load_tokenizer_and_model
from coconut_qwen.train_utils import maybe_enable_lora

import evaluate as eval_base


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B-base")
    p.add_argument("--output-dir", default="runs/gsm8k_cot_baseline")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--train-style",
        choices=["project", "qwen_report"],
        default="project",
        help="project trains on the repo's #### format; qwen_report matches the 4-shot Qwen GSM8K eval prompt.",
    )
    p.add_argument("--num-fewshot", type=int, default=4)
    return p.parse_args()


def encode_baseline(ex, tokenizer, *, train_style: str, num_fewshot: int):
    if train_style == "qwen_report":
        prompt = eval_base.qwen_report_prompt_for_question(ex.question, num_fewshot=num_fewshot)
        rationale = strip_gsm8k_calculator_annotations("\n".join(ex.reasoning_steps))
        if rationale:
            target = f" {rationale}\nThe answer is {ex.final_answer}."
        else:
            target = f" The answer is {ex.final_answer}."
    else:
        prompt = prompt_for_question(ex.question)
        target = "\n".join(ex.reasoning_steps) + f"\n#### {ex.final_answer}"
    if tokenizer.eos_token:
        target += tokenizer.eos_token
    prefix_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    target_ids = tokenizer(target, add_special_tokens=False).input_ids
    input_ids = torch.tensor([prefix_ids + target_ids], dtype=torch.long)
    labels = torch.tensor([[-100] * len(prefix_ids) + target_ids], dtype=torch.long)
    return input_ids, labels


def collate_baseline(batch, tokenizer, *, train_style: str, num_fewshot: int):
    input_rows = []
    label_rows = []
    for ex in batch:
        input_ids, labels = encode_baseline(
            ex,
            tokenizer,
            train_style=train_style,
            num_fewshot=num_fewshot,
        )
        input_rows.append(input_ids.squeeze(0))
        label_rows.append(labels.squeeze(0))
    input_ids = pad_sequence(
        input_rows,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    labels = pad_sequence(label_rows, batch_first=True, padding_value=-100)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
    return input_ids, attention_mask, labels


def make_warmup_decay_scheduler(
    optim: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> LambdaLR:
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("--min-lr-ratio must be between 0 and 1")

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, float(step - warmup_steps + 1) / float(decay_steps))
        return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))

    return LambdaLR(optim, lr_lambda)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_tokenizer_and_model(args.model)
    add_coconut_tokens(tokenizer, model)
    model = maybe_enable_lora(
        model,
        True,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    examples = load_gsm8k_examples("train", limit=args.limit)
    loader = DataLoader(
        examples,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_baseline(
            batch,
            tokenizer,
            train_style=args.train_style,
            num_fewshot=args.num_fewshot,
        ),
    )
    total_micro_steps = len(loader) * args.epochs
    total_optim_steps = (total_micro_steps + args.grad_accum_steps - 1) // args.grad_accum_steps
    warmup_steps = args.warmup_steps or int(total_optim_steps * args.warmup_ratio)
    scheduler = make_warmup_decay_scheduler(
        optim,
        warmup_steps=warmup_steps,
        total_steps=total_optim_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    log_path = out_dir / "train_log.csv"
    log_path.write_text("micro_step,optimizer_step,loss,lr\n", encoding="utf-8")
    micro_step = 0
    optim_step = 0
    optim.zero_grad(set_to_none=True)
    for _ in range(args.epochs):
        for input_ids, attention_mask, labels in tqdm(loader, desc="cot", mininterval=30):
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)
            labels = labels.to(model.device)
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
            loss = out.loss
            (loss / args.grad_accum_steps).backward()
            micro_step += 1
            if micro_step % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)
                optim_step += 1
            with log_path.open("a", encoding="utf-8") as f:
                lr = scheduler.get_last_lr()[0]
                f.write(f"{micro_step},{optim_step},{float(loss.detach().cpu()):.6f},{lr:.8g}\n")

    if micro_step % args.grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        scheduler.step()
        optim.zero_grad(set_to_none=True)

    final_dir = out_dir / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"saved {final_dir}")


if __name__ == "__main__":
    main()
