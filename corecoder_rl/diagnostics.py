from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{lineno}: invalid JSONL row: {exc}") from exc
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = int(round((len(values) - 1) * q))
    return float(values[idx])


def _fmt_ratio(numer: int, denom: int) -> str:
    if denom <= 0:
        return "0/0 (0.0%)"
    return f"{numer}/{denom} ({numer / denom * 100:.1f}%)"


def analyze_metrics(metrics_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metrics_rows:
        sessions[str(row.get("session_id", "unknown"))].append(row)

    turns_per_session = [len(rows) for rows in sessions.values()]
    response_tokens = [float(r.get("response_tokens") or 0) for r in metrics_rows]
    prompt_tokens = [float(r.get("prompt_tokens") or 0) for r in metrics_rows]
    rewards = [float(r.get("reward") or 0) for r in metrics_rows]
    loss_mask_on = [r for r in metrics_rows if int(r.get("loss_mask") or 0) == 1]
    no_next_state = [r for r in metrics_rows if not bool(r.get("has_next_state"))]
    zero_reward = [r for r in metrics_rows if float(r.get("reward") or 0) == 0.0]
    teacher_extra = [r for r in metrics_rows if bool(r.get("teacher_extra_prompt_present"))]
    prm_rows = [r for r in metrics_rows if r.get("prm_votes") not in (None, "", [])]
    action_types = Counter(str(r.get("step_action_type") or "unknown") for r in metrics_rows)
    scoring_sources = Counter(str(r.get("step_scoring_source") or "unknown") for r in metrics_rows)

    return {
        "rows": len(metrics_rows),
        "sessions": len(sessions),
        "turns_per_session_avg": _mean([float(x) for x in turns_per_session]),
        "turns_per_session_p95": _percentile([float(x) for x in turns_per_session], 0.95),
        "prompt_tokens_avg": _mean(prompt_tokens),
        "response_tokens_avg": _mean(response_tokens),
        "response_tokens_p95": _percentile(response_tokens, 0.95),
        "reward_avg": _mean(rewards),
        "effective_samples": len(loss_mask_on),
        "no_next_state": len(no_next_state),
        "zero_reward": len(zero_reward),
        "teacher_extra": len(teacher_extra),
        "prm_rows": len(prm_rows),
        "action_types": action_types,
        "scoring_sources": scoring_sources,
    }


def analyze_records(record_rows: list[dict[str, Any]]) -> dict[str, Any]:
    response_chars = [float(len(str(r.get("response_text") or ""))) for r in record_rows]
    next_state_missing = [r for r in record_rows if not r.get("next_state")]
    tool_rows = [r for r in record_rows if r.get("tool_calls")]
    return {
        "rows": len(record_rows),
        "response_chars_avg": _mean(response_chars),
        "response_chars_p95": _percentile(response_chars, 0.95),
        "next_state_missing": len(next_state_missing),
        "tool_rows": len(tool_rows),
    }


def build_recommendations(metrics: dict[str, Any], records: dict[str, Any], num_gpus: int) -> list[str]:
    recs: list[str] = []
    total = int(metrics.get("rows") or 0)
    effective = int(metrics.get("effective_samples") or 0)
    no_next_state = int(metrics.get("no_next_state") or 0)
    zero_reward = int(metrics.get("zero_reward") or 0)
    teacher_extra = int(metrics.get("teacher_extra") or 0)
    prm_rows = int(metrics.get("prm_rows") or 0)

    if num_gpus <= 1:
        recs.append("Single GPU: do not run full train_async. Use this machine for log diagnosis, feeder dry-runs, proxy smoke tests, or rollout-only debugging.")
        recs.append("If you must smoke-test generation, keep ROLLOUT_BATCH_SIZE <= 4, PRM_M=1, and disable teacher extra prompt/logprob first.")

    if total and effective / total < 0.5:
        recs.append("Effective samples are below 50%. Inspect no-next-state, zero rewards, and loss_mask=0 rows before increasing rollout batch size.")
    if total and no_next_state / total > 0.25:
        recs.append("Many rows have no next_state. Session finalization or tool/user follow-up is likely delaying usable samples.")
    if total and zero_reward / total > 0.40:
        recs.append("Zero reward ratio is high. PRM/rule scoring may be too conservative, or sessions are being finalized without clear feedback.")
    if total and teacher_extra / total > 0.50:
        recs.append("Teacher extra prompt is active for most samples. For throughput tests, disable CORECODER_TEACHER_EXTRA_PROMPT_ENABLED first.")
    if total and prm_rows / total > 0.50:
        recs.append("PRM is active for most samples. Measure baseline throughput with PRM disabled or PRM_M=1 before tuning SGLang.")
    if float(metrics.get("turns_per_session_p95") or 0) >= 6:
        recs.append("Session turn p95 is high. Large rollout-batch-size will amplify long-tail waiting; prefer 4-16 until rollout engines are scaled.")
    if int(records.get("next_state_missing") or 0) > 0:
        recs.append("Record file contains missing next_state entries. Check whether session_done is sent reliably at the end of each feeder session.")

    if not recs:
        recs.append("No obvious data-production issue found in the provided logs. Next step: compare pure policy rollout against PRM/teacher-enabled rollout.")
    return recs


def render_report(metrics_rows: list[dict[str, Any]], record_rows: list[dict[str, Any]], num_gpus: int) -> str:
    metrics = analyze_metrics(metrics_rows)
    records = analyze_records(record_rows)
    recs = build_recommendations(metrics, records, num_gpus)

    lines = [
        "CoreCoder_RL rollout doctor",
        "",
        "Metrics",
        f"- rows: {metrics['rows']}",
        f"- sessions: {metrics['sessions']}",
        f"- turns/session avg,p95: {metrics['turns_per_session_avg']:.2f}, {metrics['turns_per_session_p95']:.0f}",
        f"- prompt tokens avg: {metrics['prompt_tokens_avg']:.1f}",
        f"- response tokens avg,p95: {metrics['response_tokens_avg']:.1f}, {metrics['response_tokens_p95']:.0f}",
        f"- reward avg: {metrics['reward_avg']:.3f}",
        f"- effective samples: {_fmt_ratio(metrics['effective_samples'], metrics['rows'])}",
        f"- no next_state: {_fmt_ratio(metrics['no_next_state'], metrics['rows'])}",
        f"- zero reward: {_fmt_ratio(metrics['zero_reward'], metrics['rows'])}",
        f"- teacher extra prompt rows: {_fmt_ratio(metrics['teacher_extra'], metrics['rows'])}",
        f"- PRM rows: {_fmt_ratio(metrics['prm_rows'], metrics['rows'])}",
        "",
        "Breakdown",
        f"- action types: {dict(metrics['action_types'])}",
        f"- scoring sources: {dict(metrics['scoring_sources'])}",
    ]

    if records["rows"]:
        lines += [
            "",
            "Records",
            f"- rows: {records['rows']}",
            f"- response chars avg,p95: {records['response_chars_avg']:.1f}, {records['response_chars_p95']:.0f}",
            f"- missing next_state: {_fmt_ratio(records['next_state_missing'], records['rows'])}",
            f"- tool-call rows: {_fmt_ratio(records['tool_rows'], records['rows'])}",
        ]

    lines += ["", "Recommendations"]
    lines += [f"- {rec}" for rec in recs]
    return "\n".join(lines)


def add_doctor_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--metrics", type=Path, help="Path to corecoder_rl_metrics.jsonl")
    parser.add_argument("--records", type=Path, help="Path to CoreCoder record jsonl")
    parser.add_argument("--num-gpus", type=int, default=1, help="GPU count for recommendation context")


def run_doctor(args: argparse.Namespace) -> int:
    metrics_rows = _read_jsonl(args.metrics)
    record_rows = _read_jsonl(args.records)
    if not metrics_rows and not record_rows:
        raise SystemExit("Provide --metrics and/or --records with at least one JSONL row.")
    print(render_report(metrics_rows, record_rows, args.num_gpus))
    return 0
