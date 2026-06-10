#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coconut_qwen.data import (
    EncodedCoconutExample,
    encode_for_stage,
    load_gsm8k_examples,
    strip_gsm8k_calculator_annotations,
)
from coconut_qwen.modeling import (
    BOT_TOKEN,
    CoconutConfig,
    CoconutForCausalLM,
    EOT_TOKEN,
    load_tokenizer_and_model,
    make_labels,
)
from coconut_qwen.train_utils import maybe_enable_lora, to_tensor

import evaluate as eval_base


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B-base")
    p.add_argument("--output-dir", default="runs/gsm8k_coconut")
    p.add_argument("--latent-steps", type=int, default=2)
    p.add_argument("--epochs-per-stage", type=int, default=1)
    p.add_argument("--max-stages", type=int, default=4)
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
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--answer-only-loss", action="store_true")
    return p.parse_args()


def encode_coconut(
    ex,
    tokenizer,
    *,
    latent_steps: int,
    stage: int,
    supervise_reasoning: bool,
    train_style: str,
    num_fewshot: int,
) -> EncodedCoconutExample:
    if train_style == "project":
        return encode_for_stage(
            ex,
            tokenizer,
            latent_steps=latent_steps,
            stage=stage,
            supervise_reasoning=supervise_reasoning,
        )

    hidden_step_count = min(stage, len(ex.reasoning_steps))
    visible_steps = ex.reasoning_steps[hidden_step_count:]
    prompt = eval_base.qwen_report_prompt_for_question(ex.question, num_fewshot=num_fewshot)
    prefix_text = prompt + "\n" + BOT_TOKEN
    suffix_text = EOT_TOKEN
    if visible_steps:
        suffix_text += "\n" + strip_gsm8k_calculator_annotations("\n".join(visible_steps))
    suffix_text += f"\nThe answer is {ex.final_answer}."
    if tokenizer.eos_token:
        suffix_text += tokenizer.eos_token

    prefix_ids = tokenizer(prefix_text, add_special_tokens=True).input_ids
    suffix_ids = tokenizer(suffix_text, add_special_tokens=False).input_ids
    if supervise_reasoning:
        supervised_mask = [True] * len(suffix_ids)
    else:
        answer_text = f"\nThe answer is {ex.final_answer}."
        if tokenizer.eos_token:
            answer_text += tokenizer.eos_token
        answer_ids = tokenizer(answer_text, add_special_tokens=False).input_ids
        supervised_mask = [False] * len(suffix_ids)
        if len(answer_ids) <= len(suffix_ids):
            start = len(suffix_ids) - len(answer_ids)
            supervised_mask[start:] = [True] * len(answer_ids)

    labels = make_labels(len(prefix_ids), latent_steps, suffix_ids, supervised_mask).squeeze(0).tolist()
    return EncodedCoconutExample(prefix_ids, suffix_ids, labels, prefix_text, suffix_text)


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
    if args.batch_size != 1:
        raise ValueError("Coconut latent training currently supports only --batch-size 1; use --grad-accum-steps for an effective batch size.")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, base_model = load_tokenizer_and_model(args.model)
    base_model = maybe_enable_lora(
        base_model,
        True,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    cfg = CoconutConfig(
        latent_steps=args.latent_steps,
        bot_token_id=tokenizer.convert_tokens_to_ids("<bot>"),
        eot_token_id=tokenizer.convert_tokens_to_ids(EOT_TOKEN),
        latent_token_id=tokenizer.convert_tokens_to_ids("<latent>"),
    )
    model = CoconutForCausalLM(base_model, cfg)
    examples = load_gsm8k_examples("train", limit=args.limit)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_micro_steps = len(examples) * args.epochs_per_stage * args.max_stages
    total_optim_steps = (total_micro_steps + args.grad_accum_steps - 1) // args.grad_accum_steps
    warmup_steps = args.warmup_steps or int(total_optim_steps * args.warmup_ratio)
    scheduler = make_warmup_decay_scheduler(
        optim,
        warmup_steps=warmup_steps,
        total_steps=total_optim_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    base_model.train()
    rng = random.Random(args.seed)
    micro_step = 0
    optim_step = 0
    log_path = out_dir / "train_log.csv"
    log_path.write_text("micro_step,optimizer_step,stage,loss,lr\n", encoding="utf-8")
    optim.zero_grad(set_to_none=True)

    for stage in range(args.max_stages):
        for _ in range(args.epochs_per_stage):
            epoch_examples = examples.copy()
            rng.shuffle(epoch_examples)
            for ex in tqdm(epoch_examples, desc=f"stage {stage}", mininterval=30):
                enc = encode_coconut(
                    ex,
                    tokenizer,
                    latent_steps=args.latent_steps,
                    stage=stage,
                    supervise_reasoning=not args.answer_only_loss,
                    train_style=args.train_style,
                    num_fewshot=args.num_fewshot,
                )
                prefix = to_tensor(enc.prefix_ids, model.device)
                suffix = to_tensor(enc.suffix_ids, model.device)
                labels = to_tensor(enc.labels, model.device)
                loss = model.forward_cached(prefix, suffix, labels)["loss"]
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
                    f.write(
                        f"{micro_step},{optim_step},{stage},{float(loss.detach().cpu()):.6f},{lr:.8g}\n"
                    )

        if micro_step % args.grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            scheduler.step()
            optim.zero_grad(set_to_none=True)
            optim_step += 1

        stage_dir = out_dir / f"stage_{stage}"
        base_model.save_pretrained(stage_dir)
        tokenizer.save_pretrained(stage_dir)

    final_dir = out_dir / "final"
    base_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"saved {final_dir}")


if __name__ == "__main__":
    main()
