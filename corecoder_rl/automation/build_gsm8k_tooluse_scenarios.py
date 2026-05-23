#!/usr/bin/env python3
"""Build Python-tool math scenarios from GSM8K."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DEFAULT_GSM8K = Path("/root/autodl-tmp/corecoder_rl/data/gsm8k/train.jsonl")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "tooluse_math_scenarios.jsonl"
_FINAL_RE = re.compile(r"####\s*([-+0-9,./]+)")


def extract_answer(answer: str) -> str:
    match = _FINAL_RE.search(answer)
    if match:
        return match.group(1).replace(",", "").strip()
    return answer.strip().splitlines()[-1].replace(",", "")


def build(input_path: Path, output_path: Path, limit: int) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            raw = json.loads(line)
            answer = extract_answer(raw.get("answer", ""))
            row = {
                "scenario_id": f"gsm8k-python-{count:04d}",
                "domain": "math_python_tool",
                "difficulty": "gsm8k",
                "requires_tool": True,
                "allowed_tools": ["python"],
                "question": raw["question"],
                "reference_answer": answer,
                "opening_user_message": (
                    "Solve this math word problem. Use the python tool for the arithmetic, "
                    "then give a concise final answer.\n\n" + raw["question"]
                ),
                "target_turns": 2,
                "checker": "numeric_exact_match",
            }
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            if limit and count >= limit:
                break
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_GSM8K)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=300)
    args = parser.parse_args()
    count = build(args.input, args.output, args.limit)
    print(json.dumps({"output": str(args.output), "count": count}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
