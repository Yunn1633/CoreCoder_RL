#!/usr/bin/env python3
"""Feed Python-tool GSM8K-style agent trajectories into the CoreCoder RL proxy."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from corecoder_rl.tool_env import PYTHON_TOOL_SCHEMA, execute_python_tool
from corecoder_rl.tool_env.python_tool import parse_python_tool_arguments

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SCENARIO_BANK = SCRIPT_DIR / "tooluse_math_scenarios.jsonl"
DEFAULT_URL = "http://127.0.0.1:30000/v1/chat/completions"
DEFAULT_MODEL = "qwen3-4b"
_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")

TOOLUSE_SYSTEM_PROMPT = """You are a careful math agent.
You have exactly one tool named python. For arithmetic in word problems, call python first.
After observing the tool result, give a short final answer.
Do not invent tool results. Do not call any tool except python.
"""


def load_scenarios(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def post_turn(client: httpx.Client, args: argparse.Namespace, session_id: str, messages: list[dict[str, Any]], done: bool, metadata: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    payload = {
        "model": args.model,
        "session_id": session_id,
        "turn_type": "main",
        "session_done": done,
        "messages": messages,
        "tools": [PYTHON_TOOL_SCHEMA],
        "tool_choice": "auto",
        "temperature": args.temperature,
        "max_tokens": max_tokens,
    }
    payload.update({k: v for k, v in metadata.items() if v not in (None, "")})
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    last_error: Exception | None = None
    for attempt in range(args.request_retries + 1):
        try:
            resp = client.post(args.url, headers=headers, json=payload, timeout=None)
            if resp.status_code == 503 and attempt < args.request_retries:
                time.sleep(args.retry_delay)
                continue
            resp.raise_for_status()
            return resp.json()
        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError) as exc:
            last_error = exc
            if attempt >= args.request_retries:
                break
            time.sleep(args.retry_delay)
    if last_error is not None:
        raise last_error
    raise RuntimeError("request failed without response")


def assistant_message(output: dict[str, Any]) -> dict[str, Any]:
    choice = (output.get("choices") or [{}])[0]
    msg = dict(choice.get("message") or {})
    msg.setdefault("role", "assistant")
    if msg.get("content") is None:
        msg["content"] = ""
    return msg


def execute_tool_call(tool_call: dict[str, Any], timeout: float) -> tuple[dict[str, Any], dict[str, Any]]:
    fn = tool_call.get("function") or {}
    name = fn.get("name", "")
    call_id = tool_call.get("id") or f"call_{uuid.uuid4().hex[:8]}"
    if name != "python":
        content = json.dumps({"ok": False, "error": f"unsupported tool: {name}"}, ensure_ascii=False)
        return {"role": "tool", "tool_call_id": call_id, "name": name or "unknown", "content": content}, {"ok": False, "tool_name": name}
    code = parse_python_tool_arguments(fn.get("arguments"))
    result = execute_python_tool(code, timeout=timeout)
    return (
        {"role": "tool", "tool_call_id": call_id, "name": "python", "content": result.to_message_content()},
        {"ok": result.ok, "tool_name": "python", "tool_code": code, "tool_output": result.output, "tool_error": result.error},
    )


def normalize_number(text: str) -> str | None:
    nums = _NUM_RE.findall(str(text).replace(",", ""))
    if not nums:
        return None
    val = nums[-1]
    try:
        f = float(val)
        if abs(f - int(f)) < 1e-9:
            return str(int(f))
        return ("%.10f" % f).rstrip("0").rstrip(".")
    except ValueError:
        return val


def final_correct(final_text: str, reference: str) -> bool:
    return normalize_number(final_text) == normalize_number(reference)


def run_session(client: httpx.Client, args: argparse.Namespace, scenario: dict[str, Any]) -> dict[str, Any]:
    session_id = f"corecoder-tooluse-{scenario.get('scenario_id', 'scenario')}-{uuid.uuid4().hex[:8]}"
    messages = [
        {"role": "system", "content": TOOLUSE_SYSTEM_PROMPT},
        {"role": "user", "content": scenario["opening_user_message"]},
    ]
    metadata = {
        "scenario_id": scenario.get("scenario_id"),
        "student_mode": "tool_env",
        "student_model": "python",
        "feeder_id": args.feeder_id,
        "rl_method": args.rl_method,
        "prompt_source": "tooluse",
        "extra_prompt_model": "none",
        "requires_tool": True,
        "allowed_tools": "python",
        "question": scenario.get("question"),
        "reference_answer": scenario.get("reference_answer"),
        "checker": scenario.get("checker"),
    }
    first = post_turn(client, args, session_id, messages, False, metadata, args.tool_call_max_tokens)
    assistant = assistant_message(first)
    tool_calls = assistant.get("tool_calls") or []
    messages.append(assistant)

    tool_messages = []
    tool_results = []
    for call in tool_calls[: args.max_tool_calls]:
        tool_msg, tool_meta = execute_tool_call(call, timeout=args.tool_timeout)
        messages.append(tool_msg)
        tool_messages.append(tool_msg)
        tool_results.append(tool_meta)

    tool_success = bool(tool_results) and all(r.get("ok") for r in tool_results)
    metadata.update({
        "tool_call_count": len(tool_calls),
        "tool_success": tool_success,
        "tool_name": "python" if tool_calls else "none",
    })

    final = post_turn(client, args, session_id, messages, True, metadata, args.final_max_tokens)
    final_msg = assistant_message(final)
    messages.append(final_msg)
    is_correct = final_correct(final_msg.get("content", ""), scenario.get("reference_answer", ""))
    reward = 1.0 if is_correct and tool_success else 0.0

    return {
        "session_id": session_id,
        "scenario_id": scenario.get("scenario_id"),
        "question": scenario.get("question"),
        "reference_answer": scenario.get("reference_answer"),
        "tool_calls": tool_calls,
        "tool_messages": tool_messages,
        "tool_results": tool_results,
        "final_answer": final_msg.get("content", ""),
        "tool_success": tool_success,
        "final_correct": is_correct,
        "reward": reward,
        "messages": messages,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-bank", type=Path, default=Path(os.getenv("CORECODER_TOOLUSE_SCENARIO_BANK", DEFAULT_SCENARIO_BANK)))
    parser.add_argument("--url", default=os.getenv("CORECODER_RL_URL", DEFAULT_URL))
    parser.add_argument("--api-key", default=os.getenv("CORECODER_RL_API_KEY", os.getenv("SGLANG_API_KEY", "change-me")))
    parser.add_argument("--model", default=os.getenv("CORECODER_RL_MODEL", DEFAULT_MODEL))
    parser.add_argument("--limit", type=int, default=int(os.getenv("CORECODER_TOOLUSE_LIMIT", "4")))
    parser.add_argument("--output", type=Path, default=Path(os.getenv("CORECODER_TOOLUSE_OUTPUT", "/root/autodl-tmp/corecoder_rl/logs/corecoder_tooluse_trajectories.jsonl")))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("CORECODER_TOOLUSE_TEMPERATURE", "0.2")))
    parser.add_argument("--tool-timeout", type=float, default=float(os.getenv("CORECODER_TOOL_TIMEOUT", "3")))
    parser.add_argument("--max-tool-calls", type=int, default=int(os.getenv("CORECODER_MAX_TOOL_CALLS", "2")))
    parser.add_argument("--tool-call-max-tokens", type=int, default=int(os.getenv("CORECODER_TOOL_CALL_MAX_TOKENS", "512")))
    parser.add_argument("--final-max-tokens", type=int, default=int(os.getenv("CORECODER_TOOLUSE_FINAL_MAX_TOKENS", "512")))
    parser.add_argument("--feeder-id", default=os.getenv("CORECODER_FEEDER_ID", "tooluse_python_feeder"))
    parser.add_argument("--rl-method", default=os.getenv("CORECODER_RL_METHOD", "opsd_tooluse"))
    parser.add_argument("--request-retries", type=int, default=int(os.getenv("CORECODER_TOOLUSE_REQUEST_RETRIES", "8")))
    parser.add_argument("--retry-delay", type=float, default=float(os.getenv("CORECODER_TOOLUSE_RETRY_DELAY", "5")))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    scenarios = load_scenarios(args.scenario_bank, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(json.dumps({"scenarios": [s.get("scenario_id") for s in scenarios], "tools": [PYTHON_TOOL_SCHEMA]}, ensure_ascii=False))
        return 0

    with httpx.Client() as client, args.output.open("a", encoding="utf-8") as out:
        for scenario in scenarios:
            row = run_session(client, args, scenario)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(json.dumps({
                "fed": row["session_id"],
                "tool_calls": len(row["tool_calls"]),
                "tool_success": row["tool_success"],
                "final_correct": row["final_correct"],
                "reward": row["reward"],
            }, ensure_ascii=False), flush=True)
            time.sleep(0.2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
