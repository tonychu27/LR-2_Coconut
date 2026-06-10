#!/usr/bin/env python
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import gc
import json
from pathlib import Path
import random
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified GSM8K evaluator for original models and LoRA adapters."
    )
    p.add_argument("--mode", choices=["direct", "cot", "coconut"], required=True)
    p.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="Original model name. For LoRA, use the base model here.",
    )
    p.add_argument("--adapter", default=None, help="Optional LoRA/PEFT adapter directory.")
    p.add_argument("--latent-steps", type=int, default=2)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument(
        "--eval-style",
        choices=["project", "lm_eval", "qwen_report"],
        default="project",
        help=(
            "project uses this repo's prompts; lm_eval uses the EleutherAI direct GSM8K task-style "
            "prompt; qwen_report uses the Qwen3 report's GSM8K 4-shot CoT setting."
        ),
    )
    p.add_argument("--num-fewshot", type=int, default=None)
    p.add_argument("--fewshot-seed", type=int, default=1234)
    p.add_argument(
        "--backend",
        choices=["auto", "hf", "vllm"],
        default="auto",
        help="auto uses vLLM for direct/cot and HF for Coconut.",
    )
    p.add_argument("--output", default=None)
    p.add_argument(
        "--merged-model-dir",
        default=None,
        help="Optional directory for a merged LoRA checkpoint used by --backend vllm. Reused if it already exists.",
    )
    p.add_argument(
        "--keep-merged-model",
        action="store_true",
        help="Keep the temporary merged LoRA checkpoint. Implied when --merged-model-dir is set.",
    )
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--gpu-memory-utilization", type=float, default=None)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--dtype", default="auto")
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

    style = f"_{args.eval_style}" if args.eval_style != "project" else ""
    return Path("runs") / f"original_{args.mode}{style}_predictions.csv"


def direct_prompt_for_question(question: str) -> str:
    return (
        "Solve the grade-school math problem. Finish with '#### <answer>'.\n\n"
        f"Problem: {question}\nAnswer:\n"
    )


GSM8K_STOP_STRINGS = ["Question:", "</s>", "<|im_end|>"]
GSM8K_COT_STOP_STRINGS = ["\nA:", "Q:", "</s>", "<|im_end|>"]

QWEN_REPORT_COT_FEWSHOTS: list[dict[str, str]] = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
        "After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted.\n"
        "So there must have been 21 - 15 = 6. The answer is 6.",
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. "
        "After eating 35, they had 74 - 35 = 39.\nThe answer is 39.",
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. "
        "How many lollipops did Jason give to Denny?",
        "answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. "
        "So he gave Denny 20 - 12 = 8. The answer is 8.",
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. "
        "How many toys does he have now?",
        "answer": "Shawn started with 5 toys.\nIf he got 2 toys each from his mom and dad, then that is 4 more toys. "
        "5 + 4 = 9. The answer is 9.",
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, "
        "from monday to thursday. How many computers are now in the server room?",
        "answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. "
        "So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.",
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?",
        "answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. "
        "After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.",
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
        "So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.",
    },
]


def benchmark_prompt_for_question(
    question: str,
    *,
    fewshot_rows: list[dict[str, str]],
) -> str:
    prompt = ""
    for row in fewshot_rows:
        prompt += f"Question: {row['question']}\nAnswer: {row['answer']}\n\n"
    prompt += f"Question: {question}\nAnswer:"
    return prompt


def qwen_report_prompt_for_question(question: str, *, num_fewshot: int) -> str:
    prompt = ""
    for row in QWEN_REPORT_COT_FEWSHOTS[:num_fewshot]:
        prompt += f"Q: {row['question']}\nA: {row['answer']}\n\n"
    prompt += f"Q: {question}\nA:"
    return prompt


def sample_lm_eval_fewshots(
    train_rows: list[dict[str, str]],
    *,
    num_fewshot: int,
    rng: random.Random,
) -> list[dict[str, str]]:
    return rng.sample(train_rows, num_fewshot)


def trim_at_stop_strings(text: str) -> str:
    stop_positions = [pos for stop in [*GSM8K_STOP_STRINGS, *GSM8K_COT_STOP_STRINGS] if (pos := text.find(stop)) >= 0]
    return text[: min(stop_positions)] if stop_positions else text


def stop_token_sequences_for_strings(tokenizer, stop_strings: list[str]) -> list[list[int]]:
    return [
        tokenizer(stop, add_special_tokens=False).input_ids
        for stop in stop_strings
        if tokenizer(stop, add_special_tokens=False).input_ids
    ]


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


def merged_model_ready(path: Path) -> bool:
    return (path / "config.json").exists() and (
        (path / "model.safetensors").exists()
        or (path / "model.safetensors.index.json").exists()
        or (path / "pytorch_model.bin").exists()
        or (path / "pytorch_model.bin.index.json").exists()
    )


def merge_lora_adapter(base_model: str, adapter: str, output_dir: Path) -> None:
    from peft import PeftModel

    from coconut_qwen.modeling import load_tokenizer_and_model

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_tokenizer_and_model(base_model)
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    del model
    del tokenizer
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


@contextmanager
def model_path_for_vllm(args: argparse.Namespace):
    if not args.adapter:
        yield args.model, None
        return

    if args.merged_model_dir:
        merged_dir = Path(args.merged_model_dir)
        if not merged_model_ready(merged_dir):
            print(f"[merge] Writing merged LoRA checkpoint to {merged_dir}")
            merge_lora_adapter(args.model, args.adapter, merged_dir)
        else:
            print(f"[merge] Reusing merged LoRA checkpoint at {merged_dir}")
        yield str(merged_dir), str(merged_dir)
        return

    tmp = tempfile.mkdtemp(prefix="merged_lora_vllm_")
    merged_dir = Path(tmp)
    try:
        print(f"[merge] Writing temporary merged LoRA checkpoint to {merged_dir}")
        merge_lora_adapter(args.model, args.adapter, merged_dir)
        yield str(merged_dir), (str(merged_dir) if args.keep_merged_model else None)
    finally:
        if args.keep_merged_model:
            print(f"[merge] Kept merged LoRA checkpoint at {merged_dir}")
        else:
            shutil.rmtree(merged_dir, ignore_errors=True)


def build_examples_and_prompts(args: argparse.Namespace):
    from coconut_qwen.data import load_gsm8k_examples, prompt_for_question
    from datasets import load_dataset

    if args.eval_style != "project":
        train_ds = load_dataset("openai/gsm8k", "main", split="train")
        test_ds = load_dataset("openai/gsm8k", "main", split="test")
        if args.limit:
            test_ds = test_ds.select(range(min(args.limit, len(test_ds))))
        train_rows = [dict(row) for row in train_ds]
        examples = list(test_ds)
        fewshot_rng = random.Random(args.fewshot_seed)
        if args.eval_style == "qwen_report":
            prompts = [
                qwen_report_prompt_for_question(ex["question"], num_fewshot=args.num_fewshot)
                for ex in examples
            ]
        else:
            prompts = [
                benchmark_prompt_for_question(
                    ex["question"],
                    fewshot_rows=sample_lm_eval_fewshots(
                        train_rows,
                        num_fewshot=args.num_fewshot,
                        rng=fewshot_rng,
                    ),
                )
                for ex in examples
            ]
        return examples, prompts

    examples = load_gsm8k_examples("test", limit=args.limit)
    prompts = [
        direct_prompt_for_question(ex.question)
        if args.mode == "direct"
        else prompt_for_question(ex.question)
        for ex in examples
    ]
    return examples, prompts


def evaluate_text_hf(args: argparse.Namespace) -> tuple[list[dict[str, str | int]], int, int]:
    import torch
    from tqdm import tqdm

    from coconut_qwen.data import (
        extract_numeric_answer,
        load_gsm8k_examples,
        normalize_number,
        prompt_for_question,
        split_gsm8k_answer,
    )
    from datasets import load_dataset

    tokenizer, model = load_model_and_tokenizer(args.model, args.adapter)
    max_new_tokens = args.max_new_tokens or (512 if args.eval_style != "project" else (64 if args.mode == "direct" else 160))

    rows: list[dict[str, str | int]] = []
    correct = 0
    if args.eval_style != "project":
        train_ds = load_dataset("openai/gsm8k", "main", split="train")
        test_ds = load_dataset("openai/gsm8k", "main", split="test")
        if args.limit:
            test_ds = test_ds.select(range(min(args.limit, len(test_ds))))
        train_rows = [dict(row) for row in train_ds]
        examples = list(test_ds)
    else:
        examples = load_gsm8k_examples("test", limit=args.limit)
        train_rows = []
    fewshot_rng = random.Random(args.fewshot_seed)
    for idx, ex in enumerate(tqdm(examples, desc=f"{args.mode} hf eval", mininterval=30)):
        if args.eval_style == "qwen_report":
            prompt = qwen_report_prompt_for_question(ex["question"], num_fewshot=args.num_fewshot)
            _, final_answer = split_gsm8k_answer(ex["answer"])
        elif args.eval_style == "lm_eval":
            fewshot_rows = sample_lm_eval_fewshots(
                train_rows,
                num_fewshot=args.num_fewshot,
                rng=fewshot_rng,
            )
            prompt = benchmark_prompt_for_question(ex["question"], fewshot_rows=fewshot_rows)
            _, final_answer = split_gsm8k_answer(ex["answer"])
        else:
            prompt = direct_prompt_for_question(ex.question) if args.mode == "direct" else prompt_for_question(ex.question)
            final_answer = ex.final_answer
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
        if args.eval_style != "project":
            text = trim_at_stop_strings(text)
        pred = extract_numeric_answer(text)
        gold = normalize_number(final_answer)
        ok = pred == gold
        correct += int(ok)
        rows.append({"idx": idx, "gold": gold, "pred": pred, "correct": int(ok), "text": text})
    return rows, correct, len(examples)


def evaluate_text_vllm(args: argparse.Namespace, model_path: str) -> tuple[list[dict[str, str | int]], int, int]:
    from tqdm import tqdm

    from coconut_qwen.data import (
        extract_numeric_answer,
        normalize_number,
        split_gsm8k_answer,
    )
    from vllm import LLM, SamplingParams

    examples, prompts = build_examples_and_prompts(args)
    max_new_tokens = args.max_new_tokens or (512 if args.eval_style != "project" else (64 if args.mode == "direct" else 160))
    llm_kwargs = {
        "model": model_path,
        "trust_remote_code": True,
        "dtype": args.dtype,
    }
    if args.max_model_len:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        stop=GSM8K_COT_STOP_STRINGS if args.eval_style == "qwen_report" else (GSM8K_STOP_STRINGS if args.eval_style == "lm_eval" else None),
    )
    outputs = llm.generate(prompts, sampling_params)

    rows: list[dict[str, str | int]] = []
    correct = 0
    for idx, (ex, out) in enumerate(tqdm(zip(examples, outputs), total=len(examples), desc=f"{args.mode} score", mininterval=30)):
        text = out.outputs[0].text
        if args.eval_style != "project":
            text = trim_at_stop_strings(text)
        pred = extract_numeric_answer(text)
        if args.eval_style != "project":
            _, final_answer = split_gsm8k_answer(ex["answer"])
        else:
            final_answer = ex.final_answer
        gold = normalize_number(final_answer)
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
    from coconut_qwen.modeling import BOT_TOKEN, CoconutConfig, CoconutForCausalLM, EOT_TOKEN
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
    max_new_tokens = args.max_new_tokens or (512 if args.eval_style == "qwen_report" else 128)
    stop_token_ids = (
        stop_token_sequences_for_strings(tokenizer, GSM8K_COT_STOP_STRINGS)
        if args.eval_style == "qwen_report"
        else None
    )

    rows: list[dict[str, str | int]] = []
    correct = 0
    if args.eval_style == "qwen_report":
        from datasets import load_dataset

        from coconut_qwen.data import split_gsm8k_answer

        test_ds = load_dataset("openai/gsm8k", "main", split="test")
        if args.limit:
            test_ds = test_ds.select(range(min(args.limit, len(test_ds))))
        examples = list(test_ds)
    else:
        examples = load_gsm8k_examples("test", limit=args.limit)
    for idx, ex in enumerate(tqdm(examples, desc="coconut hf eval", mininterval=30)):
        if args.eval_style == "qwen_report":
            prompt = qwen_report_prompt_for_question(ex["question"], num_fewshot=args.num_fewshot)
            prefix_ids = tokenizer(prompt + "\n" + BOT_TOKEN, add_special_tokens=True).input_ids
            _, final_answer = split_gsm8k_answer(ex["answer"])
        else:
            enc = encode_for_stage(ex, tokenizer, latent_steps=args.latent_steps, stage=999)
            prefix_ids = enc.prefix_ids
            final_answer = ex.final_answer
        prefix = to_tensor(prefix_ids, model.device)
        out_ids = model.generate_cached(
            prefix,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            stop_token_ids=stop_token_ids,
        )
        text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        if args.eval_style == "qwen_report":
            text = trim_at_stop_strings(text)
        pred = extract_numeric_answer(text)
        gold = normalize_number(final_answer)
        ok = pred == gold
        correct += int(ok)
        rows.append({"idx": idx, "gold": gold, "pred": pred, "correct": int(ok), "text": text})
    return rows, correct, len(examples)


def resolve_backend(args: argparse.Namespace) -> str:
    if args.backend != "auto":
        return args.backend
    if args.mode == "coconut":
        return "hf"
    return "vllm"


def validate_args(args: argparse.Namespace) -> None:
    if args.num_fewshot is None:
        args.num_fewshot = 4 if args.eval_style == "qwen_report" else 5
    if args.mode == "direct" and args.eval_style == "qwen_report":
        raise ValueError("--eval-style qwen_report is a CoT setting; use --mode cot.")
    if args.eval_style == "qwen_report" and args.model == "Qwen/Qwen3-0.6B":
        print(
            "warning: Qwen3 report GSM8K comparison is for Qwen/Qwen3-0.6B-Base, "
            "but --model is Qwen/Qwen3-0.6B.",
            file=sys.stderr,
        )
    if args.mode == "coconut" and args.eval_style == "lm_eval":
        raise ValueError("--eval-style lm_eval is only for text generation modes, not Coconut latent evaluation.")
    if args.mode == "coconut" and (getattr(args, "merged_model_dir", None) or getattr(args, "keep_merged_model", False)):
        raise ValueError("Merged-model options are only used with --backend vllm, not Coconut evaluation.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    backend = resolve_backend(args)
    if args.mode == "coconut" and backend == "vllm":
        raise ValueError("Coconut evaluation uses Hugging Face because it needs latent hidden-state feedback.")

    output = Path(args.output) if args.output else default_output_path(args)
    kept_merged_model = None
    if args.mode == "coconut":
        rows, correct, total = evaluate_coconut(args)
    elif backend == "vllm":
        with model_path_for_vllm(args) as (model_path, kept_merged_model):
            rows, correct, total = evaluate_text_vllm(args, model_path)
    else:
        rows, correct, total = evaluate_text_hf(args)

    acc = correct / max(1, total)
    summary = {
        "mode": args.mode,
        "model": args.model,
        "adapter": args.adapter,
        "merged_model": kept_merged_model,
        "latent_steps": args.latent_steps if args.mode == "coconut" else None,
        "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "eval_style": args.eval_style,
        "num_fewshot": args.num_fewshot if args.eval_style != "project" else None,
        "fewshot_seed": args.fewshot_seed if args.eval_style != "project" else None,
        "backend": "vllm_merged_lora" if backend == "vllm" and args.adapter else backend,
        "correct": correct,
        "total": total,
        "accuracy": acc,
    }
    write_outputs(output, rows=rows, summary=summary)
    print(f"accuracy={acc:.4f} ({correct}/{total})")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
