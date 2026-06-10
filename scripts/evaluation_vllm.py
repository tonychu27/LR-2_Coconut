#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import evaluate


def main() -> None:
    if "--backend" not in sys.argv:
        sys.argv[1:1] = ["--backend", "vllm"]
    if "--model" not in sys.argv:
        sys.argv[1:1] = ["--model", "Qwen/Qwen3-0.6B-Base"]
    evaluate.main()


if __name__ == "__main__":
    main()
