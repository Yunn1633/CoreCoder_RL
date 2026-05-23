#!/usr/bin/env python3
"""Build CoreCoder teacher/student scenarios from GSM8K-style data files."""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
import subprocess
from typing import Any


DEFAULT_LOCAL_GSM8K = "/root/CoreCoder_RL/corecoder-test/GSM8K.json"
DEFAULT_MODELSCOPE_DIR = "/root/autodl-tmp/corecoder_rl/datasets/gsm8k"
FINAL_ANSWER_RE = re.compile(r"####\s*([-+]?[\d,]+(?:\.\d+)?)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsm8k-path", default=DEFAULT_LOCAL_GSM8K)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--download-if-missing", action="store_true")
    parser.add_argument("--modelscope-dir", default=DEFAULT_MODELSCOPE_DIR)
    return parser.parse_args()


def maybe_download_gsm8k(path: str, target_dir: str) -> str:
    candidate = pathlib.Path(path)
    if candidate.exists():
        return str(candidate)
    modelscope = subprocess.run(
        ["bash", "-lc", "command -v modelscope"],
        capture_output=True,
        text=True,
    )
    if modelscope.returncode != 0:
        raise RuntimeError(
            f"GSM8K data not found at {path}, and modelscope CLI is not installed. "
            "Install modelscope or place GSM8K json/jsonl under --gsm8k-path."
        )
    pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "modelscope",
            "download",
            "--dataset",
            "AI-ModelScope/gsm8k",
            "--local_dir",
            target_dir,
        ],
        check=True,
    )
    found = find_gsm8k_file(target_dir)
    if not found:
        raise RuntimeError(f"Downloaded GSM8K but no usable json/jsonl file was found in {target_dir}")
    return found


def find_gsm8k_file(root: str) -> str | None:
    path = pathlib.Path(root)
    if path.is_file():
        return str(path)
    candidates: list[pathlib.Path] = []
    for pattern in ("**/*socratic*.jsonl", "**/*train*.jsonl", "**/*test*.jsonl", "**/*.jsonl", "**/*.json"):
        candidates.extend(sorted(path.glob(pattern)))
    for candidate in candidates:
        try:
            rows = load_gsm8k_rows(str(candidate), limit=3)
        except Exception:  # noqa: BLE001
            continue
        if rows:
            return str(candidate)
    return None


def load_gsm8k_rows(path: str, limit: int = 0) -> list[dict[str, str]]:
    source = pathlib.Path(path)
    if source.is_dir():
        found = find_gsm8k_file(path)
        if not found:
            raise RuntimeError(f"No GSM8K json/jsonl file found under {path}")
        source = pathlib.Path(found)
    rows: list[dict[str, str]] = []
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".jsonl":
        raw_rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        if isinstance(payload, list):
            raw_rows = payload
        elif isinstance(payload, dict):
            raw_rows = []
            for key in ("train", "test", "validation", "main", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    raw_rows.extend(value)
            if not raw_rows:
                raw_rows = list(payload.values()) if all(isinstance(v, dict) for v in payload.values()) else []
        else:
            raw_rows = []

    for item in raw_rows:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or item.get("ground_truth") or item.get("ground_truth_answer") or "").strip()
        if not question or not answer:
            continue
        final_answer = extract_final_answer(answer)
        rows.append({"question": question, "answer": answer, "final_answer": final_answer})
        if limit and len(rows) >= limit:
            break
    return rows


def extract_final_answer(answer: str) -> str:
    match = FINAL_ANSWER_RE.search(answer)
    if match:
        return match.group(1).replace(",", "")
    compact = answer.strip().splitlines()[-1].strip() if answer.strip() else ""
    return compact


def wrong_answer(final_answer: str, rng: random.Random) -> str:
    cleaned = final_answer.replace(",", "").strip()
    try:
        value = int(float(cleaned))
    except ValueError:
        return f"{cleaned}（我不太确定）" if cleaned else "我算不出来"
    offsets = [1, -1, 2, -2, 5, -5, 10]
    candidate = value + rng.choice(offsets)
    if candidate == value:
        candidate += 1
    return str(candidate)


def build_scenarios(rows: list[dict[str, str]], count: int, seed: int) -> list[dict[str, Any]]:
    if not rows:
        raise RuntimeError("No GSM8K rows available")
    rng = random.Random(seed)
    scenarios: list[dict[str, Any]] = []
    templates = [
        make_student_hint_scenario,
        make_student_check_scenario,
        make_teacher_grade_scenario,
        make_teacher_socratic_scenario,
    ]
    for idx in range(count):
        row = rng.choice(rows)
        maker = templates[idx % len(templates)]
        scenarios.append(maker(row, idx, rng))
    return scenarios


def scenario_base(row: dict[str, str], idx: int, domain: str, role_pattern: str) -> dict[str, Any]:
    return {
        "scenario_id": f"gsm8k-{idx + 1:05d}",
        "domain": domain,
        "role_pattern": role_pattern,
        "difficulty": "medium",
        "target_turns": 4,
        "stop_rules": ["single_problem", "context_budget", "resolved", "max_turns"],
        "tags": ["teacher-student", "gsm8k", "math", "safe"],
        "reference_question": row["question"],
        "reference_answer": row["answer"],
        "reference_final_answer": row["final_answer"],
        "context_policy": {
            "single_problem": True,
            "stop_when_near_limit": True,
        },
    }


def make_student_hint_scenario(row: dict[str, str], idx: int, rng: random.Random) -> dict[str, Any]:
    scenario = scenario_base(row, idx, "gsm8k_student_hint", "student_to_tutor")
    scenario["opening_user_message"] = (
        "我在做一道英文 GSM8K 数学题。请像老师一样用中文引导我，先帮我理解题意，"
        "再给分步提示；不要一上来只报最终答案。请只讨论这一道题，可以随着我的反馈多轮引导。\n\n"
        f"题目：{row['question']}"
    )
    scenario["feedback_types"] = ["requery", "directive", "evaluative"]
    return scenario


def make_student_check_scenario(row: dict[str, str], idx: int, rng: random.Random) -> dict[str, Any]:
    scenario = scenario_base(row, idx, "gsm8k_student_correction", "student_to_tutor")
    guess = wrong_answer(row["final_answer"], rng)
    scenario["opening_user_message"] = (
        "我像学生一样先交一个不确定的答案，请你检查我的思路，不要直接批评；"
        "如果我错了，请指出最可能错在哪一步，并用提问方式引导我改。"
        "请只讨论这一道题，可以随着我的反馈多轮引导。\n\n"
        f"题目：{row['question']}\n我现在算出来的答案是：{guess}"
    )
    scenario["student_guess"] = guess
    scenario["feedback_types"] = ["terminal_state", "directive", "evaluative"]
    return scenario


def make_teacher_grade_scenario(row: dict[str, str], idx: int, rng: random.Random) -> dict[str, Any]:
    scenario = scenario_base(row, idx, "gsm8k_teacher_grading", "teacher_to_grader")
    guess = wrong_answer(row["final_answer"], rng)
    scenario["opening_user_message"] = (
        "我是老师，下面是学生对一道 GSM8K 题的回答。请你像助教一样批改："
        "先判断答案是否正确，再给一句具体、友好的反馈，最后给一个下次检查的方法。"
        "请只讨论这一道题。如果我觉得评语不够具体，会继续要求你改。\n\n"
        f"题目：{row['question']}\n学生答案：{guess}\n参考最终答案：{row['final_answer']}"
    )
    scenario["student_guess"] = guess
    scenario["feedback_types"] = ["evaluative", "directive", "terminal_state"]
    return scenario


def make_teacher_socratic_scenario(row: dict[str, str], idx: int, rng: random.Random) -> dict[str, Any]:
    scenario = scenario_base(row, idx, "gsm8k_teacher_socratic", "teacher_to_grader")
    scenario["opening_user_message"] = (
        "请把下面这道 GSM8K 题设计成一段师生苏格拉底式短对话。"
        "学生一开始不要完全会做，老师通过多轮问题引导学生自己发现计算关系，"
        "语气要像真实课堂。请只讨论这一道题。\n\n"
        f"题目：{row['question']}"
    )
    scenario["feedback_types"] = ["preference", "requery", "evaluative"]
    return scenario


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    path = maybe_download_gsm8k(args.gsm8k_path, args.modelscope_dir) if args.download_if_missing else args.gsm8k_path
    rows = load_gsm8k_rows(path)
    scenarios = build_scenarios(rows, args.count, args.seed)
    write_jsonl(args.output, scenarios)
    print(json.dumps({"source": path, "rows": len(rows), "scenarios": len(scenarios), "output": args.output}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
