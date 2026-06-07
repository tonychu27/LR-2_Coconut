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
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--warmup-ratio", type=float, default=0.0)
    p.add_argument("--lora", action="store_true")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    return p.parse_args()


def encode_baseline(ex, tokenizer):
    prompt = prompt_for_question(ex.question)
    target = "\n".join(ex.reasoning_steps) + f"\n#### {ex.final_answer}"
    prefix_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    target_ids = tokenizer(target, add_special_tokens=False).input_ids
    input_ids = torch.tensor([prefix_ids + target_ids], dtype=torch.long)
    labels = torch.tensor([[-100] * len(prefix_ids) + target_ids], dtype=torch.long)
    return input_ids, labels


def collate_baseline(batch, tokenizer):
    input_rows = []
    label_rows = []
    for ex in batch:
        input_ids, labels = encode_baseline(ex, tokenizer)
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


def make_warmup_scheduler(optim: torch.optim.Optimizer, warmup_steps: int) -> LambdaLR:
    if warmup_steps <= 0:
        return LambdaLR(optim, lambda _: 1.0)

    def lr_lambda(step: int) -> float:
        return min(1.0, float(step + 1) / float(warmup_steps))

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
        args.lora,
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
        shuffle=False,
        collate_fn=lambda batch: collate_baseline(batch, tokenizer),
    )
    total_micro_steps = len(loader) * args.epochs
    total_optim_steps = (total_micro_steps + args.grad_accum_steps - 1) // args.grad_accum_steps
    warmup_steps = args.warmup_steps or int(total_optim_steps * args.warmup_ratio)
    scheduler = make_warmup_scheduler(optim, warmup_steps)

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
