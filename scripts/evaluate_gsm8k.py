#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from peft import PeftModel
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coconut_qwen.data import encode_for_stage, extract_numeric_answer, load_gsm8k_examples, normalize_number
from coconut_qwen.modeling import CoconutConfig, CoconutForCausalLM, EOT_TOKEN, load_tokenizer_and_model
from coconut_qwen.train_utils import to_tensor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--adapter", default=None)
    p.add_argument("--latent-steps", type=int, default=2)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output", default="runs/eval_predictions.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer, base_model = load_tokenizer_and_model(args.model)
    if args.adapter:
        base_model = PeftModel.from_pretrained(base_model, args.adapter)
    base_model.eval()

    cfg = CoconutConfig(
        latent_steps=args.latent_steps,
        bot_token_id=tokenizer.convert_tokens_to_ids("<bot>"),
        eot_token_id=tokenizer.convert_tokens_to_ids(EOT_TOKEN),
        latent_token_id=tokenizer.convert_tokens_to_ids("<latent>"),
    )
    model = CoconutForCausalLM(base_model, cfg)

    rows = ["idx,gold,pred,correct,text\n"]
    correct = 0
    examples = load_gsm8k_examples("test", limit=args.limit)
    for idx, ex in enumerate(tqdm(examples, desc="eval", mininterval=30)):
        enc = encode_for_stage(ex, tokenizer, latent_steps=args.latent_steps, stage=999)
        prefix = to_tensor(enc.prefix_ids, model.device)
        out_ids = model.generate_cached(prefix, max_new_tokens=128, eos_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        pred = extract_numeric_answer(text)
        gold = normalize_number(ex.final_answer)
        ok = pred == gold
        correct += int(ok)
        escaped = text.replace('"', '""').replace("\n", "\\n")
        rows.append(f'{idx},{gold},{pred},{int(ok)},"{escaped}"\n')

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(rows), encoding="utf-8")
    acc = correct / max(1, len(examples))
    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "latent_steps": args.latent_steps,
        "limit": args.limit,
        "correct": correct,
        "total": len(examples),
        "accuracy": acc,
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"accuracy={acc:.4f} ({correct}/{len(examples)})")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
