#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from peft import PeftModel
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coconut_qwen.data import extract_numeric_answer, load_gsm8k_examples, normalize_number, prompt_for_question
from coconut_qwen.modeling import load_tokenizer_and_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--adapter", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output", default="runs/eval_cot_predictions.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer, model = load_tokenizer_and_model(args.model)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = ["idx,gold,pred,correct,text\n"]
    correct = 0
    examples = load_gsm8k_examples("test", limit=args.limit)
    for idx, ex in enumerate(tqdm(examples, desc="cot eval", mininterval=30)):
        prompt = prompt_for_question(ex.question)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            out = model.generate(input_ids=input_ids, max_new_tokens=160, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
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
