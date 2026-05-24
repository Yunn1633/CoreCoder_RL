#!/usr/bin/env python3
"""Feed GSM8K Python-tool trajectories through the real CoreCoder Agent loop."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from corecoder import Agent
from corecoder.llm import LLMResponse, ToolCall
from corecoder.tools.base import Tool
from corecoder_rl.tool_env import execute_python_tool
from corecoder_rl.tool_env.python_tool import parse_python_tool_arguments

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SCENARIO_BANK = SCRIPT_DIR / "tooluse_math_scenarios.jsonl"
DEFAULT_URL = "http://127.0.0.1:30000/v1/chat/completions"
DEFAULT_MODEL = "qwen3-4b"
_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")

MATH_AGENT_TASK_PREFIX = """You are solving a math word problem as CoreCoder.
You may use the python tool when it is helpful for arithmetic, algebra, numeric checks, or small calculations.
Use a tool only when it helps. If the problem is simple enough, answer directly.
When you use a tool, wait for the tool result before giving the final answer.
Give a concise final answer.

Problem:
"""


class PythonMathTool(Tool):
    name = "python"
    description = "Run a small, safe Python snippet for arithmetic, algebra, numeric checks, or small data calculations."
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute. The last expression is returned when possible.",
            }
        },
        "required": ["code"],
    }

    def __init__(self, timeout: float = 3.0):
        self.timeout = timeout

    def execute(self, code: str = "") -> str:
        return execute_python_tool(code, timeout=self.timeout).to_message_content()


class CoreCoderRLProxyLLM:
    """CoreCoder LLM adapter that routes Agent.chat turns through the RL proxy."""

    def __init__(self, args: argparse.Namespace, session_id: str, metadata: dict[str, Any]):
        self.args = args
        self.session_id = session_id
        self.metadata = metadata
        self.client = httpx.Client(timeout=None)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def close(self):
        self.client.close()

    def chat(self, messages: list[dict], tools: list[dict] | None = None, on_token=None) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.args.model,
            "session_id": self.session_id,
            "turn_type": "main",
            "session_done": False,
            "messages": messages,
            "tools": tools or [],
            "tool_choice": "auto",
            "temperature": self.args.temperature,
            "max_tokens": self.args.max_tokens,
        }
        payload.update({k: v for k, v in self.metadata.items() if v not in (None, "")})
        data = self._post_json(self.args.url, payload)
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        usage = data.get("usage") or {}
        self.total_prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.total_completion_tokens += int(usage.get("completion_tokens") or 0)
        content = msg.get("content") or ""
        if content and on_token:
            on_token(content)
        parsed = []
        for raw in msg.get("tool_calls") or []:
            fn = raw.get("function") or {}
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"code": parse_python_tool_arguments(args)}
            parsed.append(ToolCall(id=raw.get("id") or f"call_{uuid.uuid4().hex[:8]}", name=fn.get("name") or "", arguments=args))
        return LLMResponse(content=content, tool_calls=parsed)

    def finalize(self, messages: list[dict], final_correct: bool, reward: float, final_answer: str, tool_summary: dict[str, Any]) -> dict[str, Any]:
        base_url = self.args.url.rsplit("/v1/chat/completions", 1)[0]
        url = base_url + "/corecoder/session_done"
        payload = {
            "session_id": self.session_id,
            "messages": messages,
            "final_correct": final_correct,
            "reward": reward,
            "final_answer": final_answer,
            **self.metadata,
            **tool_summary,
        }
        return self._post_json(url, payload)

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.args.api_key}"} if self.args.api_key else {}
        last_error: Exception | None = None
        for attempt in range(self.args.request_retries + 1):
            try:
                resp = self.client.post(url, headers=headers, json=payload, timeout=None)
                if resp.status_code == 503 and attempt < self.args.request_retries:
                    time.sleep(self.args.retry_delay)
                    continue
                resp.raise_for_status()
                return resp.json()
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError) as exc:
                last_error = exc
                if attempt >= self.args.request_retries:
                    break
                time.sleep(self.args.retry_delay)
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"request failed: {url}")


def load_scenarios(path: Path, limit: int, seed: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    rng = random.Random(seed) if seed is not None else random.SystemRandom()
    rng.shuffle(rows)
    return rows[:limit] if limit else rows


def build_user_message(scenario: dict[str, Any]) -> str:
    question = scenario.get("question") or scenario.get("opening_user_message", "")
    return MATH_AGENT_TASK_PREFIX + question


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


def collect_tool_summary(messages: list[dict[str, Any]]) -> dict[str, Any]:
    tool_calls = []
    tool_messages = []
    tool_results = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tool_calls.extend(msg.get("tool_calls") or [])
        elif msg.get("role") == "tool":
            tool_messages.append(msg)
            try:
                parsed = json.loads(msg.get("content") or "{}")
            except json.JSONDecodeError:
                parsed = {"ok": False, "output": msg.get("content", "")}
            tool_results.append({
                "ok": bool(parsed.get("ok")),
                "tool_name": msg.get("name") or "python",
                "tool_output": parsed.get("output", ""),
                "tool_error": parsed.get("error", ""),
            })
    tool_success = bool(tool_results) and all(r.get("ok") for r in tool_results)
    return {
        "tool_calls": tool_calls,
        "tool_messages": tool_messages,
        "tool_results": tool_results,
        "tool_call_count": len(tool_calls),
        "tool_success": tool_success,
        "tool_name": "python" if tool_calls else "none",
    }


def run_session(args: argparse.Namespace, scenario: dict[str, Any]) -> dict[str, Any]:
    session_id = f"corecoder-agent-tooluse-{scenario.get('scenario_id', 'scenario')}-{uuid.uuid4().hex[:8]}"
    metadata = {
        "scenario_id": scenario.get("scenario_id"),
        "student_mode": "corecoder_agent",
        "student_model": "corecoder_policy",
        "feeder_id": args.feeder_id,
        "rl_method": args.rl_method,
        "prompt_source": "corecoder_agent_tooluse",
        "extra_prompt_model": "none",
        "requires_tool": "optional",
        "allowed_tools": "python",
        "question": scenario.get("question"),
        "reference_answer": scenario.get("reference_answer"),
        "checker": scenario.get("checker"),
    }
    llm = CoreCoderRLProxyLLM(args, session_id, metadata)
    agent = Agent(
        llm=llm,
        tools=[PythonMathTool(timeout=args.tool_timeout)],
        max_context_tokens=args.max_context_tokens,
        max_rounds=args.max_agent_rounds,
    )
    try:
        final_answer = agent.chat(build_user_message(scenario))
        full_messages = agent._full_messages()
        tool_summary = collect_tool_summary(full_messages)
        is_correct = final_correct(final_answer, scenario.get("reference_answer", ""))
        reward = 1.0 if is_correct else 0.0
        llm.finalize(full_messages, is_correct, reward, final_answer, tool_summary)
        return {
            "session_id": session_id,
            "scenario_id": scenario.get("scenario_id"),
            "question": scenario.get("question"),
            "reference_answer": scenario.get("reference_answer"),
            "final_answer": final_answer,
            "final_correct": is_correct,
            "reward": reward,
            "messages": full_messages,
            **tool_summary,
        }
    finally:
        llm.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-bank", type=Path, default=Path(os.getenv("CORECODER_TOOLUSE_SCENARIO_BANK", DEFAULT_SCENARIO_BANK)))
    parser.add_argument("--url", default=os.getenv("CORECODER_RL_URL", DEFAULT_URL))
    parser.add_argument("--api-key", default=os.getenv("CORECODER_RL_API_KEY", os.getenv("SGLANG_API_KEY", "change-me")))
    parser.add_argument("--model", default=os.getenv("CORECODER_RL_MODEL", DEFAULT_MODEL))
    parser.add_argument("--limit", type=int, default=int(os.getenv("CORECODER_TOOLUSE_LIMIT", "4")))
    seed_env = os.getenv("CORECODER_TOOLUSE_SEED", "")
    parser.add_argument("--seed", type=int, default=int(seed_env) if seed_env else None)
    parser.add_argument("--output", type=Path, default=Path(os.getenv("CORECODER_TOOLUSE_OUTPUT", "/root/autodl-tmp/corecoder_rl/logs/corecoder_tooluse_trajectories.jsonl")))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("CORECODER_TOOLUSE_TEMPERATURE", "0.2")))
    parser.add_argument("--tool-timeout", type=float, default=float(os.getenv("CORECODER_TOOL_TIMEOUT", "3")))
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("CORECODER_TOOLUSE_MAX_TOKENS", "512")))
    parser.add_argument("--max-agent-rounds", type=int, default=int(os.getenv("CORECODER_MAX_AGENT_ROUNDS", "4")))
    parser.add_argument("--max-context-tokens", type=int, default=int(os.getenv("CORECODER_MAX_CONTEXT_TOKENS", "32768")))
    parser.add_argument("--feeder-id", default=os.getenv("CORECODER_FEEDER_ID", "corecoder_agent_tooluse_feeder"))
    parser.add_argument("--rl-method", default=os.getenv("CORECODER_RL_METHOD", "opsd_tooluse"))
    parser.add_argument("--request-retries", type=int, default=int(os.getenv("CORECODER_TOOLUSE_REQUEST_RETRIES", "8")))
    parser.add_argument("--retry-delay", type=float, default=float(os.getenv("CORECODER_TOOLUSE_RETRY_DELAY", "5")))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    scenarios = load_scenarios(args.scenario_bank, args.limit, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(json.dumps({"scenarios": [s.get("scenario_id") for s in scenarios], "mode": "corecoder_agent"}, ensure_ascii=False))
        return 0

    with args.output.open("a", encoding="utf-8") as out:
        for scenario in scenarios:
            row = run_session(args, scenario)
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
