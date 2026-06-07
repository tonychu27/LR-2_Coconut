#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified GSM8K evaluator for original, LoRA, and full fine-tuned models."
    )
    p.add_argument("--mode", choices=["direct", "cot", "coconut"], required=True)
    p.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="Original model name or full fine-tuned checkpoint. For LoRA, use the base model here.",
    )
    p.add_argument("--adapter", default=None, help="Optional LoRA/PEFT adapter directory.")
    p.add_argument("--latent-steps", type=int, default=2)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument(
        "--backend",
        choices=["auto", "hf", "vllm"],
        default="auto",
        help="auto uses HF for LoRA/Coconut and vLLM for original/full-ft direct/cot.",
    )
    p.add_argument("--output", default=None)
    return p.parse_args()


def default_output_path(args: argparse.Namespace) -> Path:
    if args.adapter:
        adapter_path = Path(args.adapter)
        base = adapter_path.parent if adapter_path.name == "final" else adapter_path
        return Path("runs") / f"{base.name.removeprefix('gsm8k_')}_predictions.csv"

    model_path = Path(args.model)
    if model_path.exists():
        base = model_path.parent if model_path.name == "final" else model_path
        return Path("runs") / f"{base.name.removeprefix('gsm8k_')}_predictions.csv"

    return Path("runs") / f"original_{args.mode}_predictions.csv"


def direct_prompt_for_question(question: str) -> str:
    return (
        "Solve the grade-school math problem. Finish with '#### <answer>'.\n\n"
        f"Problem: {question}\nAnswer:\n"
    )


def load_model_and_tokenizer(model_name: str, adapter: str | None):
    from peft import PeftModel

    from coconut_qwen.modeling import load_tokenizer_and_model

    tokenizer, model = load_tokenizer_and_model(model_name)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tokenizer, model


def write_outputs(
    output: Path,
    *,
    rows: list[dict[str, str | int]],
    summary: dict[str, str | int | float | None],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "gold", "pred", "correct", "text"])
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def evaluate_text_hf(args: argparse.Namespace) -> tuple[list[dict[str, str | int]], int, int]:
    import torch
    from tqdm import tqdm

    from coconut_qwen.data import (
        extract_numeric_answer,
        load_gsm8k_examples,
        normalize_number,
        prompt_for_question,
    )

    tokenizer, model = load_model_and_tokenizer(args.model, args.adapter)
    max_new_tokens = args.max_new_tokens or (64 if args.mode == "direct" else 160)

    rows: list[dict[str, str | int]] = []
    correct = 0
    examples = load_gsm8k_examples("test", limit=args.limit)
    for idx, ex in enumerate(tqdm(examples, desc=f"{args.mode} hf eval", mininterval=30)):
        prompt = direct_prompt_for_question(ex.question) if args.mode == "direct" else prompt_for_question(ex.question)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0, input_ids.shape[1] :], skip_special_tokens=True)
        pred = extract_numeric_answer(text)
        gold = normalize_number(ex.final_answer)
        ok = pred == gold
        correct += int(ok)
        rows.append({"idx": idx, "gold": gold, "pred": pred, "correct": int(ok), "text": text})
    return rows, correct, len(examples)


def evaluate_text_vllm(args: argparse.Namespace) -> tuple[list[dict[str, str | int]], int, int]:
    from tqdm import tqdm

    from coconut_qwen.data import (
        extract_numeric_answer,
        load_gsm8k_examples,
        normalize_number,
        prompt_for_question,
    )
    from vllm import LLM, SamplingParams

    examples = load_gsm8k_examples("test", limit=args.limit)
    prompts = [
        direct_prompt_for_question(ex.question) if args.mode == "direct" else prompt_for_question(ex.question)
        for ex in examples
    ]

    max_new_tokens = args.max_new_tokens or (64 if args.mode == "direct" else 160)
    llm = LLM(model=args.model, trust_remote_code=True)
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=max_new_tokens))

    rows: list[dict[str, str | int]] = []
    correct = 0
    for idx, (ex, out) in enumerate(tqdm(zip(examples, outputs), total=len(examples), desc=f"{args.mode} score", mininterval=30)):
        text = out.outputs[0].text
        pred = extract_numeric_answer(text)
        gold = normalize_number(ex.final_answer)
        ok = pred == gold
        correct += int(ok)
        rows.append({"idx": idx, "gold": gold, "pred": pred, "correct": int(ok), "text": text})
    return rows, correct, len(examples)


def evaluate_coconut(args: argparse.Namespace) -> tuple[list[dict[str, str | int]], int, int]:
    from tqdm import tqdm

    from coconut_qwen.data import (
        encode_for_stage,
        extract_numeric_answer,
        load_gsm8k_examples,
        normalize_number,
    )
    from coconut_qwen.modeling import CoconutConfig, CoconutForCausalLM, EOT_TOKEN
    from coconut_qwen.train_utils import to_tensor

    tokenizer, base_model = load_model_and_tokenizer(args.model, args.adapter)
    model = CoconutForCausalLM(
        base_model,
        CoconutConfig(
            latent_steps=args.latent_steps,
            bot_token_id=tokenizer.convert_tokens_to_ids("<bot>"),
            eot_token_id=tokenizer.convert_tokens_to_ids(EOT_TOKEN),
            latent_token_id=tokenizer.convert_tokens_to_ids("<latent>"),
        ),
    )
    max_new_tokens = args.max_new_tokens or 128

    rows: list[dict[str, str | int]] = []
    correct = 0
    examples = load_gsm8k_examples("test", limit=args.limit)
    for idx, ex in enumerate(tqdm(examples, desc="coconut hf eval", mininterval=30)):
        enc = encode_for_stage(ex, tokenizer, latent_steps=args.latent_steps, stage=999)
        prefix = to_tensor(enc.prefix_ids, model.device)
        out_ids = model.generate_cached(
            prefix,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        pred = extract_numeric_answer(text)
        gold = normalize_number(ex.final_answer)
        ok = pred == gold
        correct += int(ok)
        rows.append({"idx": idx, "gold": gold, "pred": pred, "correct": int(ok), "text": text})
    return rows, correct, len(examples)


def resolve_backend(args: argparse.Namespace) -> str:
    if args.backend != "auto":
        return args.backend
    if args.adapter or args.mode == "coconut":
        return "hf"
    return "vllm"


def main() -> None:
    args = parse_args()
    backend = resolve_backend(args)
    if args.adapter and backend == "vllm":
        raise ValueError("LoRA adapter evaluation uses Hugging Face. Pass --backend hf or --backend auto.")
    if args.mode == "coconut" and backend == "vllm":
        raise ValueError("Coconut evaluation uses Hugging Face because it needs latent hidden-state feedback.")

    output = Path(args.output) if args.output else default_output_path(args)
    if args.mode == "coconut":
        rows, correct, total = evaluate_coconut(args)
    elif backend == "vllm":
        rows, correct, total = evaluate_text_vllm(args)
    else:
        rows, correct, total = evaluate_text_hf(args)

    acc = correct / max(1, total)
    summary = {
        "mode": args.mode,
        "model": args.model,
        "adapter": args.adapter,
        "latent_steps": args.latent_steps if args.mode == "coconut" else None,
        "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "backend": backend,
        "correct": correct,
        "total": total,
        "accuracy": acc,
    }
    write_outputs(output, rows=rows, summary=summary)
    print(f"accuracy={acc:.4f} ({correct}/{total})")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
