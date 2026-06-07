from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from peft import LoraConfig, get_peft_model


def maybe_enable_lora(
    model,
    enabled: bool,
    *,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
):
    if not enabled:
        return model
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_tokens", "lm_head"],
    )
    return get_peft_model(model, config)


def to_tensor(ids: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor([ids], dtype=torch.long, device=device)


def write_loss_artifacts(losses: list[float], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "loss_curve.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss"])
        for step, loss in enumerate(losses, start=1):
            writer.writerow([step, loss])

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("optimizer step")
    plt.ylabel("training loss")
    plt.title("Single-example Coconut overfit")
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=160)
    plt.close()
