#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from tqdm import trange

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coconut_qwen.data import encode_for_stage, load_gsm8k_examples
from coconut_qwen.modeling import CoconutConfig, CoconutForCausalLM, EOT_TOKEN, load_tokenizer_and_model
from coconut_qwen.train_utils import maybe_enable_lora, to_tensor, write_loss_artifacts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--output-dir", default="runs/one_example")
    p.add_argument("--latent-steps", type=int, default=2)
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora", action="store_true")
    p.add_argument("--save-model", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, base_model = load_tokenizer_and_model(args.model)
    base_model = maybe_enable_lora(base_model, args.lora)
    base_model.train()

    cfg = CoconutConfig(
        latent_steps=args.latent_steps,
        bot_token_id=tokenizer.convert_tokens_to_ids("<bot>"),
        eot_token_id=tokenizer.convert_tokens_to_ids(EOT_TOKEN),
        latent_token_id=tokenizer.convert_tokens_to_ids("<latent>"),
    )
    model = CoconutForCausalLM(base_model, cfg)
    device = model.device

    example = load_gsm8k_examples("train", limit=1)[0]
    encoded = encode_for_stage(example, tokenizer, latent_steps=args.latent_steps, stage=args.stage)
    prefix = to_tensor(encoded.prefix_ids, device)
    suffix = to_tensor(encoded.suffix_ids, device)
    labels = to_tensor(encoded.labels, device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    losses: list[float] = []
    for _ in trange(args.max_steps, desc="overfit"):
        optim.zero_grad(set_to_none=True)
        loss = model.forward_cached(prefix, suffix, labels)["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        losses.append(float(loss.detach().cpu()))
        if losses[-1] < 0.02:
            break

    model.eval()
    generated_ids = model.generate_cached(
        prefix,
        max_new_tokens=160,
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(generated_ids[0], skip_special_tokens=False)

    write_loss_artifacts(losses, out_dir)
    (out_dir / "prediction.txt").write_text(
        "QUESTION\n"
        f"{example.question}\n\n"
        "TARGET\n"
        f"{encoded.target_text}\n\n"
        "GENERATED_AFTER_LATENTS\n"
        f"{EOT_TOKEN}{generated}\n\n"
        f"FINAL_LOSS\n{losses[-1]:.6f}\n",
        encoding="utf-8",
    )

    if args.save_model:
        save_dir = out_dir / "final"
        base_model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

    print(f"final_loss={losses[-1]:.6f}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
