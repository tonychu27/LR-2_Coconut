from __future__ import annotations

import re
from dataclasses import dataclass

from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from .modeling import BOT_TOKEN, EOT_TOKEN, make_labels


ANSWER_RE = re.compile(r"####\s*([\-0-9,.$ ]+)")


@dataclass
class CoconutExample:
    question: str
    reasoning_steps: list[str]
    final_answer: str


@dataclass
class EncodedCoconutExample:
    prefix_ids: list[int]
    suffix_ids: list[int]
    labels: list[int]
    prompt: str
    target_text: str


def split_gsm8k_answer(answer: str) -> tuple[list[str], str]:
    final_match = ANSWER_RE.search(answer)
    final = final_match.group(1).strip() if final_match else answer.strip().splitlines()[-1]
    rationale = answer.split("####", 1)[0].strip()
    steps = [line.strip() for line in rationale.splitlines() if line.strip()]
    if not steps and rationale:
        steps = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", rationale) if piece.strip()]
    return steps, final


def load_gsm8k_examples(split: str = "train", limit: int | None = None) -> list[CoconutExample]:
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    examples = []
    for row in ds:
        steps, final = split_gsm8k_answer(row["answer"])
        examples.append(CoconutExample(row["question"], steps, final))
    return examples


def prompt_for_question(question: str) -> str:
    return (
        "Solve the grade-school math problem. Give concise reasoning, then finish with "
        "'#### <answer>'.\n\n"
        f"Problem: {question}\nAnswer:\n"
    )


def encode_for_stage(
    example: CoconutExample,
    tokenizer: PreTrainedTokenizerBase,
    *,
    latent_steps: int,
    stage: int,
    supervise_reasoning: bool = True,
) -> EncodedCoconutExample:
    """Encode one example for the staged Coconut curriculum.

    `stage` is the number of leading reasoning steps removed from language
    supervision and represented by latent states instead.
    """

    hidden_step_count = min(stage, len(example.reasoning_steps))
    visible_steps = example.reasoning_steps[hidden_step_count:]
    prompt = prompt_for_question(example.question)
    prefix_text = prompt + BOT_TOKEN
    suffix_text = EOT_TOKEN
    if visible_steps:
        suffix_text += "\n" + "\n".join(visible_steps)
    suffix_text += f"\n#### {example.final_answer}"

    prefix_ids = tokenizer(prefix_text, add_special_tokens=True).input_ids
    suffix_ids = tokenizer(suffix_text, add_special_tokens=False).input_ids
    target_text = suffix_text

    if supervise_reasoning:
        supervised_mask = [True] * len(suffix_ids)
    else:
        answer_text = f"\n#### {example.final_answer}"
        answer_ids = tokenizer(answer_text, add_special_tokens=False).input_ids
        supervised_mask = [False] * len(suffix_ids)
        if len(answer_ids) <= len(suffix_ids):
            start = len(suffix_ids) - len(answer_ids)
            supervised_mask[start:] = [True] * len(answer_ids)

    labels = make_labels(len(prefix_ids), latent_steps, suffix_ids, supervised_mask).squeeze(0).tolist()
    return EncodedCoconutExample(prefix_ids, suffix_ids, labels, prompt, target_text)


def extract_numeric_answer(text: str) -> str:
    match = ANSWER_RE.search(text)
    if match:
        return normalize_number(match.group(1))
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return normalize_number(nums[-1]) if nums else ""


def normalize_number(text: str) -> str:
    return text.strip().replace(",", "").replace("$", "")
