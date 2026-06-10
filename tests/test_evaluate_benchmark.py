import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_evaluate_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("evaluate_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_benchmark_output_is_trimmed_before_next_question():
    evaluate = load_evaluate_module()

    text = "The answer is 42.\nQuestion: A later generated example has 999 apples."

    assert evaluate.trim_at_stop_strings(text) == "The answer is 42.\n"


def test_lm_eval_prompt_uses_raw_gsm8k_answer_examples():
    evaluate = load_evaluate_module()

    prompt = evaluate.benchmark_prompt_for_question(
        "How many apples are left?",
        fewshot_rows=[
            {"question": "first", "answer": "Rationale text.\n#### 6"},
            {"question": "second", "answer": "More rationale.\n#### 5"},
        ],
    )

    assert prompt.count("\n\nQuestion:") == 2
    assert "Rationale text" in prompt
    assert "#### 6" in prompt
    assert prompt.endswith("Question: How many apples are left?\nAnswer:")


def test_lm_eval_fewshot_sampling_uses_seeded_random_examples():
    evaluate = load_evaluate_module()
    rows = [{"question": str(i), "answer": f"#### {i}"} for i in range(10)]

    first = evaluate.sample_lm_eval_fewshots(rows, num_fewshot=5, rng=__import__("random").Random(1234))
    second = evaluate.sample_lm_eval_fewshots(rows, num_fewshot=5, rng=__import__("random").Random(1234))

    assert first == second
    assert first != rows[:5]


def test_qwen_report_prompt_uses_four_shot_cot_examples():
    evaluate = load_evaluate_module()

    prompt = evaluate.qwen_report_prompt_for_question("How many apples are left?", num_fewshot=4)

    assert prompt.count("\n\nQ:") == 4
    assert "The answer is 6." in prompt
    assert prompt.endswith("Q: How many apples are left?\nA:")


def test_auto_backend_uses_vllm_for_text_modes_and_hf_for_coconut():
    evaluate = load_evaluate_module()

    assert (
        evaluate.resolve_backend(SimpleNamespace(backend="auto", mode="cot", adapter="adapter/path"))
        == "vllm"
    )
    assert (
        evaluate.resolve_backend(SimpleNamespace(backend="auto", mode="coconut", adapter="adapter/path"))
        == "hf"
    )


def test_qwen_report_eval_style_rejects_direct_mode():
    evaluate = load_evaluate_module()

    args = SimpleNamespace(
        num_fewshot=4,
        eval_style="qwen_report",
        mode="direct",
        model="Qwen/Qwen3-0.6B-Base",
    )

    with pytest.raises(ValueError, match="qwen_report is a CoT setting"):
        evaluate.validate_args(args)
