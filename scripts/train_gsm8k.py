#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coconut_qwen.data import encode_for_stage, load_gsm8k_examples
from coconut_qwen.modeling import CoconutConfig, CoconutForCausalLM, EOT_TOKEN, load_tokenizer_and_model
from coconut_qwen.train_utils import maybe_enable_lora, to_tensor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--output-dir", default="runs/gsm8k_coconut")
    p.add_argument("--latent-steps", type=int, default=2)
    p.add_argument("--epochs-per-stage", type=int, default=1)
    p.add_argument("--max-stages", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora", action="store_true")
    p.add_argument("--answer-only-loss", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, base_model = load_tokenizer_and_model(args.model)
    base_model = maybe_enable_lora(base_model, args.lora)
    cfg = CoconutConfig(
        latent_steps=args.latent_steps,
        bot_token_id=tokenizer.convert_tokens_to_ids("<bot>"),
        eot_token_id=tokenizer.convert_tokens_to_ids(EOT_TOKEN),
        latent_token_id=tokenizer.convert_tokens_to_ids("<latent>"),
    )
    model = CoconutForCausalLM(base_model, cfg)
    examples = load_gsm8k_examples("train", limit=args.limit)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    base_model.train()
    global_step = 0
    log_path = out_dir / "train_log.csv"
    log_path.write_text("global_step,stage,loss\n", encoding="utf-8")

    for stage in range(args.max_stages):
        for _ in range(args.epochs_per_stage):
            for ex in tqdm(examples, desc=f"stage {stage}", mininterval=30):
                enc = encode_for_stage(
                    ex,
                    tokenizer,
                    latent_steps=args.latent_steps,
                    stage=stage,
                    supervise_reasoning=not args.answer_only_loss,
                )
                prefix = to_tensor(enc.prefix_ids, model.device)
                suffix = to_tensor(enc.suffix_ids, model.device)
                labels = to_tensor(enc.labels, model.device)
                optim.zero_grad(set_to_none=True)
                loss = model.forward_cached(prefix, suffix, labels)["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                global_step += 1
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(f"{global_step},{stage},{float(loss.detach().cpu()):.6f}\n")

        stage_dir = out_dir / f"stage_{stage}"
        base_model.save_pretrained(stage_dir)
        tokenizer.save_pretrained(stage_dir)

    final_dir = out_dir / "final"
    base_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"saved {final_dir}")


if __name__ == "__main__":
    main()
