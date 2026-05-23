#!/usr/bin/env python3
"""Drive safe synthetic CoreCoder sessions into the existing CoreCoder-RL data path."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

import websocket

from user_simulator_qwen import QwenUserSimulator
from user_simulator_rules import RuleBasedUserSimulator


DEFAULT_SCENARIO_BANK = str(pathlib.Path(__file__).resolve().parent / "scenario_bank.jsonl")
DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/corecoder_rl/automation/data"
DEFAULT_LOG_DIR = "/root/autodl-tmp/corecoder_rl/automation/logs"
DEFAULT_RL_URL = "http://127.0.0.1:30000/v1/chat/completions"
DEFAULT_GATEWAY_HOST = "127.0.0.1"
DEFAULT_GATEWAY_PORT = 18789
DEFAULT_MODEL = "qwen3-4b"
DEFAULT_RL_API_KEY = "change-me"
DEFAULT_GATEWAY_TOKEN = "corecoder-local-dev-token-20260419-control-ui"
DEFAULT_MIN_GPU_FREE_MIB = 6144
DEFAULT_GPU_POLL_SECONDS = 30.0
DEFAULT_MAX_CONCURRENT_SESSIONS = 2
DEFAULT_CONTEXT_TOKEN_LIMIT = 16384
DEFAULT_MAX_COMPLETION_TOKENS = 1024
DEFAULT_MIN_CONTEXT_HEADROOM_TOKENS = 1536


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default="rl-synth")
    parser.add_argument("--simulator", choices=["rules", "qwen"], default="qwen")
    parser.add_argument("--scenario-bank", default=DEFAULT_SCENARIO_BANK)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=0, help="Override per-scenario target turns.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--turn-delay-seconds", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--context-token-limit", type=int, default=int(os.getenv("CORECODER_SYNTH_CONTEXT_TOKEN_LIMIT", DEFAULT_CONTEXT_TOKEN_LIMIT)))
    parser.add_argument("--max-completion-tokens", type=int, default=int(os.getenv("CORECODER_SYNTH_MAX_COMPLETION_TOKENS", DEFAULT_MAX_COMPLETION_TOKENS)))
    parser.add_argument("--min-context-headroom-tokens", type=int, default=int(os.getenv("CORECODER_SYNTH_MIN_CONTEXT_HEADROOM_TOKENS", DEFAULT_MIN_CONTEXT_HEADROOM_TOKENS)))
    parser.add_argument("--max-concurrent-sessions", type=int, default=DEFAULT_MAX_CONCURRENT_SESSIONS)
    parser.add_argument("--min-gpu-free-mib", type=int, default=DEFAULT_MIN_GPU_FREE_MIB)
    parser.add_argument("--gpu-poll-seconds", type=float, default=DEFAULT_GPU_POLL_SECONDS)
    parser.add_argument("--memory-guard-timeout", type=float, default=0.0, help="0 means wait forever.")
    parser.add_argument("--gateway-host", default=DEFAULT_GATEWAY_HOST)
    parser.add_argument("--gateway-port", type=int, default=DEFAULT_GATEWAY_PORT)
    parser.add_argument(
        "--gateway-token",
        default=os.getenv("CORECODER_GATEWAY_TOKEN", DEFAULT_GATEWAY_TOKEN),
    )
    parser.add_argument("--rl-url", default=os.getenv("CORECODER_RL_URL", DEFAULT_RL_URL))
    parser.add_argument("--rl-api-key", default=os.getenv("CORECODER_RL_API_KEY", DEFAULT_RL_API_KEY))
    parser.add_argument("--model", default=os.getenv("CORECODER_RL_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--reset-gateway-session",
        action="store_true",
        help="Reset the CoreCoder gateway session after flushing RL data. Requires operator.admin scope.",
    )
    parser.add_argument(
        "--reset-gateway-session-before",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset the agent gateway session before each synthetic session so one session discusses one task.",
    )
    parser.add_argument(
        "--local-session-reset-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If gateway sessions.reset is unauthorized, rotate the local agent session store before the session.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def rough_token_count(text: str) -> int:
    # Mixed Chinese/English chat text is usually denser than pure English.
    return max(1, (len(text) + 2) // 3)


def estimate_transcript_tokens(transcript: list[dict[str, str]], current_user_message: str = "") -> int:
    total = 0
    for message in transcript:
        total += rough_token_count(str(message.get("content", ""))) + 8
    if current_user_message:
        total += rough_token_count(current_user_message) + 8
    return total


def read_agent_session_snapshot(agent_id: str, resolved_session_key: str | None, resolved_session_id: str | None) -> dict[str, Any]:
    store_path = pathlib.Path.home() / ".corecoder" / "agents" / agent_id / "sessions" / "sessions.json"
    snapshot: dict[str, Any] = {
        "source": str(store_path),
        "found": False,
        "session_key": resolved_session_key,
        "session_id": resolved_session_id,
    }
    if not store_path.exists():
        snapshot["reason"] = "session store missing"
        return snapshot
    try:
        store = json.loads(store_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        snapshot["reason"] = f"session store decode failed: {exc}"
        return snapshot

    candidates: list[tuple[str, dict[str, Any]]] = []
    if resolved_session_key and isinstance(store.get(resolved_session_key), dict):
        candidates.append((resolved_session_key, store[resolved_session_key]))
    if resolved_session_id:
        for key, value in store.items():
            if isinstance(value, dict) and value.get("sessionId") == resolved_session_id:
                candidates.append((key, value))
    main_key = f"agent:{agent_id}:main"
    if isinstance(store.get(main_key), dict):
        candidates.append((main_key, store[main_key]))
    if not candidates:
        snapshot["reason"] = "session not found"
        return snapshot

    key, meta = candidates[0]
    snapshot.update(
        {
            "found": True,
            "session_key": key,
            "session_id": meta.get("sessionId"),
            "context_tokens": meta.get("contextTokens"),
            "total_tokens": meta.get("totalTokens"),
            "total_tokens_fresh": meta.get("totalTokensFresh"),
            "status": meta.get("status"),
            "updated_at": meta.get("updatedAt"),
        }
    )
    prompt_report = meta.get("systemPromptReport") or {}
    system_prompt = prompt_report.get("systemPrompt") or {}
    if isinstance(system_prompt, dict):
        snapshot["system_prompt_chars"] = system_prompt.get("chars")

    session_file = meta.get("sessionFile")
    if isinstance(session_file, str) and pathlib.Path(session_file).exists():
        snapshot["session_file"] = session_file
        latest_compaction_tokens: int | None = None
        try:
            with pathlib.Path(session_file).open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "compaction" and isinstance(event.get("tokensBefore"), int):
                        latest_compaction_tokens = int(event["tokensBefore"])
        except OSError as exc:
            snapshot["session_file_error"] = str(exc)
        if latest_compaction_tokens is not None:
            snapshot["latest_compaction_tokens_before"] = latest_compaction_tokens
    return snapshot


def context_budget_status(
    *,
    args: argparse.Namespace,
    transcript: list[dict[str, str]],
    current_user_message: str,
    agent_id: str,
    resolved_session_key: str | None,
    resolved_session_id: str | None,
) -> dict[str, Any]:
    snapshot = read_agent_session_snapshot(agent_id, resolved_session_key, resolved_session_id)
    limit = int(snapshot.get("context_tokens") or args.context_token_limit)
    headroom_required = max(int(args.min_context_headroom_tokens), int(args.max_completion_tokens))
    budget = max(limit - headroom_required, 1)
    token_sources: list[int] = []
    if isinstance(snapshot.get("total_tokens"), int):
        token_sources.append(int(snapshot["total_tokens"]))
    if isinstance(snapshot.get("latest_compaction_tokens_before"), int):
        token_sources.append(int(snapshot["latest_compaction_tokens_before"]))
    system_prompt_chars = snapshot.get("system_prompt_chars")
    baseline = 0
    if isinstance(system_prompt_chars, int) and system_prompt_chars > 0:
        # CoreCoder injects tool/schema/bootstrap text around the visible prompt.
        # Keep this deliberately conservative because SGLang enforces the real
        # model context limit, not the user-visible transcript length.
        baseline = max(rough_token_count("x" * system_prompt_chars), 9000)
    estimate = baseline + estimate_transcript_tokens(transcript, current_user_message)
    token_sources.append(estimate)
    observed = max(token_sources)
    return {
        "limit": limit,
        "budget_before_next_completion": budget,
        "headroom_required": headroom_required,
        "observed_or_estimated_tokens": observed,
        "estimated_local_tokens": estimate,
        "snapshot": snapshot,
        "should_stop": observed >= budget,
    }


class MemoryGuard:
    """Throttle synthetic turns when local training GPUs are near capacity."""

    def __init__(
        self,
        min_free_mib: int,
        poll_seconds: float,
        timeout_seconds: float,
    ):
        self.min_free_mib = min_free_mib
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds
        self.pause_count = 0
        self.check_count = 0
        self._lock = threading.Lock()

    def read_gpu_memory(self) -> list[dict[str, int]]:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]
        try:
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=10)
        except (FileNotFoundError, subprocess.SubprocessError):
            return []
        rows: list[dict[str, int]] = []
        for line in proc.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 4:
                continue
            try:
                rows.append(
                    {
                        "index": int(parts[0]),
                        "total_mib": int(parts[1]),
                        "used_mib": int(parts[2]),
                        "free_mib": int(parts[3]),
                    }
                )
            except ValueError:
                continue
        return rows

    def wait_until_safe(self, label: str) -> dict[str, Any]:
        started = time.monotonic()
        local_pauses = 0
        last_snapshot: list[dict[str, int]] = []
        while True:
            snapshot = self.read_gpu_memory()
            last_snapshot = snapshot
            unsafe = [row for row in snapshot if row["free_mib"] < self.min_free_mib]
            with self._lock:
                self.check_count += 1
            if not unsafe:
                return {
                    "label": label,
                    "paused": local_pauses,
                    "wait_seconds": round(time.monotonic() - started, 3),
                    "gpu_memory": snapshot,
                }
            local_pauses += 1
            with self._lock:
                self.pause_count += 1
            free_desc = ", ".join(f"gpu{row['index']}={row['free_mib']}MiB" for row in unsafe)
            print(
                f"[memory-guard] {label}: waiting; below {self.min_free_mib} MiB free ({free_desc})",
                flush=True,
            )
            if self.timeout_seconds > 0 and time.monotonic() - started >= self.timeout_seconds:
                raise RuntimeError(
                    f"memory guard timed out for {label}; latest={json.dumps(last_snapshot)}"
                )
            time.sleep(self.poll_seconds)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"checks": self.check_count, "pauses": self.pause_count}


def ensure_gateway(host: str, port: int, timeout: float = 1.0) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        if sock.connect_ex((host, port)) != 0:
            raise RuntimeError(
                f"CoreCoder Gateway is not reachable at {host}:{port}. "
                "Start `corecoder gateway` before running the synth pipeline."
            )


def load_scenarios(path: str) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            scenarios.append(json.loads(line))
    if not scenarios:
        raise RuntimeError(f"No scenarios loaded from {path}")
    return scenarios


def make_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = f"/root/.local/bin:{env.get('PATH', '')}"
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = "127.0.0.1,localhost"
    return env


def extract_agent_response(raw_stdout: str) -> tuple[str, str | None, str | None]:
    stripped = raw_stdout.strip()
    if not stripped:
        return "", None, None

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped, None, None

    resolved_session_id = (
        payload.get("result", {})
        .get("meta", {})
        .get("agentMeta", {})
        .get("sessionId")
    ) or (
        payload.get("result", {})
        .get("meta", {})
        .get("systemPromptReport", {})
        .get("sessionId")
    )
    resolved_session_key = (
        payload.get("result", {})
        .get("meta", {})
        .get("systemPromptReport", {})
        .get("sessionKey")
    )

    result = payload.get("result", {}) if isinstance(payload.get("result"), dict) else {}
    direct_candidates = [
        result.get("reply"),
        result.get("text"),
        result.get("message"),
        result.get("output"),
        result.get("content"),
        result.get("assistant"),
    ]
    for value in direct_candidates:
        text = normalize_content_text(value)
        if text:
            return text, resolved_session_id, resolved_session_key

    queue: list[Any] = [result or payload]
    candidate_keys = ("reply", "text", "content", "message", "assistant", "output")
    ignored_keys = {
        "arguments",
        "name",
        "tool",
        "tool_calls",
        "sessionkey",
        "sessionid",
        "model",
        "provider",
        "status",
        "error",
        "errormessage",
    }
    while queue:
        item = queue.pop(0)
        if isinstance(item, str) and item.strip():
            return item.strip(), resolved_session_id, resolved_session_key
        if isinstance(item, dict):
            for key in candidate_keys:
                if key.lower() in ignored_keys:
                    continue
                value = item.get(key)
                text = normalize_content_text(value)
                if text:
                    return text, resolved_session_id, resolved_session_key
                if isinstance(value, dict):
                    queue.append(value)
            queue.extend(v for k, v in item.items() if k.lower() not in ignored_keys and isinstance(v, (dict, list)))
        elif isinstance(item, list):
            queue.extend(item)
    return stripped, resolved_session_id, resolved_session_key


def normalize_content_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                elif isinstance(item.get("content"), str):
                    parts.append(str(item["content"]))
        return "\n".join(part.strip() for part in parts if part and part.strip()).strip()
    if isinstance(value, dict):
        for key in ("text", "content", "message", "reply"):
            text = value.get(key)
            if isinstance(text, str) and text.strip():
                return text.strip()
    return ""


def run_agent_turn(
    *,
    agent_id: str,
    session_id: str,
    message: str,
    timeout: int,
    env: dict[str, str],
) -> tuple[str, str | None, str | None]:
    cmd = [
        "corecoder",
        "agent",
        "--agent",
        agent_id,
        "--session-id",
        session_id,
        "--message",
        message,
        "--json",
        "--verbose",
        "off",
        "--timeout",
        str(timeout),
    ]
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            proc = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout + 30,
            )
            reply, resolved_session_id, resolved_session_key = extract_agent_response(proc.stdout)
            if not reply:
                raise RuntimeError(f"CoreCoder returned empty output for session {session_id}")
            if is_transient_corecoder_error(reply):
                raise RuntimeError(f"CoreCoder transient error reply: {reply[:300]}")
            return reply, resolved_session_id, resolved_session_key
        except (subprocess.SubprocessError, RuntimeError) as exc:
            last_error = exc
            if attempt >= 3:
                break
            print(
                f"  warning: CoreCoder turn failed for {session_id}; retry {attempt}/3: {exc}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(10 * attempt)
    raise RuntimeError(f"CoreCoder turn failed after retries for session {session_id}: {last_error}")


def is_transient_corecoder_error(reply: str) -> bool:
    lowered = reply.strip().lower()
    transient_markers = [
        "http 500",
        "internal server error",
        "503 service unavailable",
        "no_available_workers",
        "all circuits open",
    ]
    return any(marker in lowered for marker in transient_markers)


def looks_like_non_answer(reply: str, scenario: dict[str, Any]) -> bool:
    text = reply.strip()
    if not text:
        return True
    lowered = text.lower()
    if "session_status" in lowered or "unknown sessionid" in lowered:
        return True
    if "tool_calls" in lowered or "\"tool\"" in lowered:
        return True
    if scenario.get("role_pattern") == "teacher_to_grader":
        natural_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff" or ch.isalpha())
        if natural_chars < 12:
            return True
    return False


def repair_prompt_for_non_answer(scenario: dict[str, Any]) -> str:
    if scenario.get("role_pattern") == "teacher_to_grader":
        return (
            "你刚才没有给出实际批改内容。请不要调用工具，直接用自然语言回答："
            "判断学生答案是否正确，并给出具体、友好、能帮助学生改进的反馈。"
        )
    return (
        "你刚才没有给出实际讲解。请不要调用工具，直接用自然语言继续回答这道题，"
        "像老师一样引导我理解。"
    )


def flush_session(
    *,
    session_id: str,
    rl_url: str,
    rl_api_key: str,
    model: str,
    retries: int = 20,
    retry_sleep_seconds: float = 3.0,
) -> None:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "[automation] close session"}],
        "turn_type": "side",
        "session_id": session_id,
        "session_done": True,
        "stream": False,
    }
    req = urllib.request.Request(
        rl_url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {rl_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            with opener.open(req, timeout=60) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Failed to flush session {session_id}: HTTP {resp.status}")
            return
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            time.sleep(retry_sleep_seconds)
    raise RuntimeError(f"Failed to flush session {session_id}: {last_error}")


def reset_gateway_session(
    *,
    gateway_host: str,
    gateway_port: int,
    gateway_token: str,
    session_key: str,
    timeout: int = 20,
) -> None:
    url = f"ws://{gateway_host}:{gateway_port}"
    ws = websocket.create_connection(url, timeout=timeout)
    try:
        while True:
            msg = json.loads(ws.recv())
            if msg.get("type") == "event" and msg.get("event") == "connect.challenge":
                break
            if msg.get("type") == "res":
                break

        connect_id = f"connect-{uuid.uuid4().hex}"
        ws.send(
            json.dumps(
                {
                    "type": "req",
                    "id": connect_id,
                    "method": "connect",
                    "params": {
                        "minProtocol": 3,
                        "maxProtocol": 3,
                        "client": {
                            "id": "corecoder-ios",
                            "displayName": "rl-synth reset",
                            "version": "dev",
                            "platform": "dev",
                            "mode": "ui",
                            "instanceId": "rl-synth-reset",
                        },
                        "locale": "zh-CN",
                        "userAgent": "rl-synth-reset",
                        "role": "operator",
                        "scopes": ["operator.read", "operator.write", "operator.admin"],
                        "caps": [],
                        "auth": {"token": gateway_token},
                    },
                }
            )
        )
        while True:
            connect_resp = json.loads(ws.recv())
            if connect_resp.get("type") == "event":
                continue
            if connect_resp.get("type") == "res" and connect_resp.get("id") == connect_id:
                if not connect_resp.get("ok", False):
                    raise RuntimeError(f"gateway connect failed: {connect_resp}")
                break

        req_id = f"reset-{uuid.uuid4().hex}"
        ws.send(
            json.dumps(
                {
                    "type": "req",
                    "id": req_id,
                    "method": "sessions.reset",
                    "params": {"key": session_key, "reason": "reset"},
                }
            )
        )
        while True:
            msg = json.loads(ws.recv())
            if msg.get("type") == "res" and msg.get("id") == req_id:
                if not msg.get("ok", False):
                    raise RuntimeError(f"sessions.reset failed: {msg}")
                return
    finally:
        try:
            ws.close()
        except Exception:  # noqa: BLE001
            pass


def rotate_local_agent_sessions(agent_id: str) -> dict[str, str]:
    sessions_dir = pathlib.Path.home() / ".corecoder" / "agents" / agent_id / "sessions"
    if not sessions_dir.exists():
        sessions_dir.mkdir(parents=True, exist_ok=True)
        return {"action": "created", "sessions_dir": str(sessions_dir)}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup_dir = sessions_dir.with_name(f"sessions.backup-{stamp}")
    suffix = 1
    while backup_dir.exists():
        suffix += 1
        backup_dir = sessions_dir.with_name(f"sessions.backup-{stamp}-{suffix}")
    shutil.move(str(sessions_dir), str(backup_dir))
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return {"action": "rotated", "sessions_dir": str(sessions_dir), "backup_dir": str(backup_dir)}


def make_simulator(kind: str, seed: int, dry_run: bool = False):
    if kind == "rules" or dry_run:
        return RuleBasedUserSimulator(seed=seed)
    return QwenUserSimulator()


def write_jsonl(path: pathlib.Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_simulator_output(result: Any) -> tuple[str, bool, str, dict[str, Any]]:
    if isinstance(result, tuple) and len(result) >= 4:
        return str(result[0]), bool(result[1]), str(result[2]), dict(result[3])
    if isinstance(result, tuple) and len(result) >= 2:
        return str(result[0]), bool(result[1]), "requery", {"raw": list(result)}
    return str(result), False, "requery", {"raw": result}


def run_synthetic_session(
    *,
    index: int,
    total: int,
    scenario: dict[str, Any],
    args: argparse.Namespace,
    env: dict[str, str],
    output_dir: pathlib.Path,
    sessions_path: pathlib.Path,
    write_lock: threading.Lock,
    memory_guard: MemoryGuard,
) -> dict[str, Any]:
    simulator = make_simulator(args.simulator, args.seed + index, dry_run=args.dry_run)
    logical_session_id = f"rl-synth-{scenario['scenario_id']}-{uuid.uuid4().hex[:8]}"
    resolved_session_id: str | None = None
    resolved_session_key: str | None = None
    transcript: list[dict[str, str]] = []
    turn_rows: list[dict[str, Any]] = []
    target_turns = args.max_turns or int(scenario.get("target_turns", 3))
    current_user_message = str(scenario["opening_user_message"])
    status = "ok"
    error = ""
    stop_reason = ""
    assistant_turns = 0
    feedback_counts: dict[str, int] = {}
    guard_pauses_start = memory_guard.stats()["pauses"]
    session_reset: dict[str, Any] = {}
    repair_count = 0

    print(f"[{index}/{total}] session={logical_session_id} scenario={scenario['scenario_id']}")
    try:
        if args.reset_gateway_session_before and not args.dry_run:
            pre_session_key = f"agent:{args.agent}:main"
            try:
                reset_gateway_session(
                    gateway_host=args.gateway_host,
                    gateway_port=args.gateway_port,
                    gateway_token=args.gateway_token,
                    session_key=pre_session_key,
                )
                session_reset = {"method": "gateway", "session_key": pre_session_key, "ok": True}
            except Exception as reset_exc:  # noqa: BLE001
                if not args.local_session_reset_fallback:
                    raise
                session_reset = {
                    "method": "local-store-fallback",
                    "session_key": pre_session_key,
                    "gateway_error": str(reset_exc),
                    "ok": True,
                    "local": rotate_local_agent_sessions(args.agent),
                }
                print(
                    f"  warning: gateway reset failed; rotated local session store: {reset_exc}",
                    file=sys.stderr,
                    flush=True,
                )
            resolved_session_key = pre_session_key
        for turn_index in range(1, target_turns + 1):
            context_before = context_budget_status(
                args=args,
                transcript=transcript,
                current_user_message=current_user_message,
                agent_id=args.agent,
                resolved_session_key=resolved_session_key,
                resolved_session_id=resolved_session_id,
            )
            if context_before["should_stop"]:
                stop_reason = "context_budget_before_turn"
                print(
                    f"  stopping before turn {turn_index}: context budget "
                    f"{context_before['observed_or_estimated_tokens']}/{context_before['limit']}",
                    flush=True,
                )
                break
            guard_before = memory_guard.wait_until_safe(f"{logical_session_id}:before-turn-{turn_index}")
            transcript.append({"role": "user", "content": current_user_message})
            if args.dry_run:
                assistant_reply = f"[dry-run] assistant reply for {scenario['scenario_id']} turn {turn_index}"
                turn_session_id = logical_session_id
                turn_session_key = None
            else:
                assistant_reply, turn_session_id, turn_session_key = run_agent_turn(
                    agent_id=args.agent,
                    session_id=logical_session_id,
                    message=current_user_message,
                    timeout=args.timeout,
                    env=env,
                )
                if turn_session_id:
                    resolved_session_id = turn_session_id
                if turn_session_key:
                    resolved_session_key = turn_session_key
                if looks_like_non_answer(assistant_reply, scenario):
                    repair_user_message = repair_prompt_for_non_answer(scenario)
                    transcript.append({"role": "assistant", "content": assistant_reply})
                    transcript.append({"role": "user", "content": repair_user_message})
                    assistant_reply, turn_session_id, turn_session_key = run_agent_turn(
                        agent_id=args.agent,
                        session_id=logical_session_id,
                        message=repair_user_message,
                        timeout=args.timeout,
                        env=env,
                    )
                    repair_count += 1
                    if turn_session_id:
                        resolved_session_id = turn_session_id
                    if turn_session_key:
                        resolved_session_key = turn_session_key
            guard_after = memory_guard.wait_until_safe(f"{logical_session_id}:after-turn-{turn_index}")
            transcript.append({"role": "assistant", "content": assistant_reply})
            assistant_turns += 1
            context_after = context_budget_status(
                args=args,
                transcript=transcript,
                current_user_message="",
                agent_id=args.agent,
                resolved_session_key=resolved_session_key,
                resolved_session_id=resolved_session_id,
            )

            feedback_type = "opening" if turn_index == 1 else "unknown"
            simulator_raw: dict[str, Any] = {}
            finish_after_turn = False
            next_user_message = ""
            if turn_index < target_turns:
                next_user_message, finish_after_turn, feedback_type, simulator_raw = normalize_simulator_output(
                    simulator.next_message(
                        scenario=scenario,
                        transcript=transcript,
                        assistant_reply=assistant_reply,
                        turn_index=turn_index,
                    )
                )
                feedback_counts[feedback_type] = feedback_counts.get(feedback_type, 0) + 1

            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "logical_session_id": logical_session_id,
                "resolved_session_id": resolved_session_id,
                "resolved_session_key": resolved_session_key,
                "scenario_id": scenario["scenario_id"],
                "domain": scenario.get("domain"),
                "turn_index": turn_index,
                "user": current_user_message,
                "assistant": assistant_reply,
                "next_user": next_user_message,
                "feedback_type": feedback_type,
                "simulator": args.simulator,
                "simulator_raw": simulator_raw,
                "gpu_before": guard_before["gpu_memory"],
                "gpu_after": guard_after["gpu_memory"],
                "context_before": context_before,
                "context_after": context_after,
                "repair_count": repair_count,
            }
            turn_rows.append(row)
            with write_lock:
                write_jsonl(sessions_path, row)

            if turn_index >= target_turns:
                stop_reason = stop_reason or "target_turns"
                break
            if context_after["should_stop"]:
                stop_reason = "context_budget_after_turn"
                print(
                    f"  stopping after turn {turn_index}: context budget "
                    f"{context_after['observed_or_estimated_tokens']}/{context_after['limit']}",
                    flush=True,
                )
                break
            current_user_message = next_user_message or "请再用一句话说明最关键的下一步。"
            if args.turn_delay_seconds > 0:
                time.sleep(args.turn_delay_seconds)
            if finish_after_turn and turn_index >= max(target_turns - 1, 1):
                stop_reason = "simulator_finish_after_next_turn"
                continue

        if not args.dry_run and assistant_turns > 0:
            flush_session(
                session_id=resolved_session_id or logical_session_id,
                rl_url=args.rl_url,
                rl_api_key=args.rl_api_key,
                model=args.model,
            )
            if resolved_session_key and args.reset_gateway_session:
                try:
                    reset_gateway_session(
                        gateway_host=args.gateway_host,
                        gateway_port=args.gateway_port,
                        gateway_token=args.gateway_token,
                        session_key=resolved_session_key,
                    )
                except Exception as reset_exc:  # noqa: BLE001
                    error = f"cleanup warning: {reset_exc}"
                    print(f"  warning: {error}", file=sys.stderr)
        if status == "ok" and assistant_turns == 0:
            status = "skipped"
            stop_reason = stop_reason or "no_turns"
    except Exception as exc:  # noqa: BLE001
        status = "error"
        error = str(exc)
        print(f"  error: {error}", file=sys.stderr)

    transcript_path = output_dir / "transcripts" / f"{logical_session_id}.json"
    transcript_payload = {
        "logical_session_id": logical_session_id,
        "resolved_session_id": resolved_session_id,
        "resolved_session_key": resolved_session_key,
        "scenario": scenario,
        "status": status,
        "error": error,
        "stop_reason": stop_reason,
        "session_reset": session_reset,
        "repair_count": repair_count,
        "turns": turn_rows,
        "transcript": transcript,
    }
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(json.dumps(transcript_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    guard_pauses_end = memory_guard.stats()["pauses"]
    summary = {
        "logical_session_id": logical_session_id,
        "resolved_session_id": resolved_session_id,
        "resolved_session_key": resolved_session_key,
        "scenario_id": scenario["scenario_id"],
        "status": status,
        "assistant_turns": assistant_turns,
        "simulator": args.simulator,
        "feedback_counts": feedback_counts,
        "memory_guard_pauses": guard_pauses_end - guard_pauses_start,
        "transcript_path": str(transcript_path),
        "stop_reason": stop_reason,
        "session_reset": session_reset,
        "repair_count": repair_count,
        "error": error,
    }
    print(f"  status={status} assistant_turns={assistant_turns}")
    return summary


def main() -> int:
    args = parse_args()
    if args.max_concurrent_sessions < 1:
        raise RuntimeError("--max-concurrent-sessions must be >= 1")
    if not args.dry_run:
        ensure_gateway(args.gateway_host, args.gateway_port)
    env = make_env()
    output_dir = pathlib.Path(args.output_dir)
    log_dir = pathlib.Path(args.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    scenarios = load_scenarios(args.scenario_bank)
    rng = random.Random(args.seed)

    sessions_path = output_dir / "sessions.jsonl"
    summary_path = output_dir / "summary.json"
    selected = [rng.choice(scenarios) for _ in range(args.sessions)]

    session_summaries: list[dict[str, Any]] = []
    write_lock = threading.Lock()
    memory_guard = MemoryGuard(
        min_free_mib=args.min_gpu_free_mib,
        poll_seconds=args.gpu_poll_seconds,
        timeout_seconds=args.memory_guard_timeout,
    )
    print(
        f"Loaded {len(scenarios)} scenarios, running {len(selected)} synthetic sessions "
        f"with concurrency={args.max_concurrent_sessions}, simulator={args.simulator}."
    )

    with ThreadPoolExecutor(max_workers=args.max_concurrent_sessions) as executor:
        futures = [
            executor.submit(
                run_synthetic_session,
                index=index,
                total=len(selected),
                scenario=scenario,
                args=args,
                env=env,
                output_dir=output_dir,
                sessions_path=sessions_path,
                write_lock=write_lock,
                memory_guard=memory_guard,
            )
            for index, scenario in enumerate(selected, start=1)
        ]
        for future in as_completed(futures):
            session_summaries.append(future.result())

    completed = [row for row in session_summaries if row["status"] == "ok"]
    total_turns = sum(int(row["assistant_turns"]) for row in session_summaries)
    feedback_counts: dict[str, int] = {}
    for row in session_summaries:
        for key, value in row.get("feedback_counts", {}).items():
            feedback_counts[key] = feedback_counts.get(key, 0) + int(value)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "simulator": args.simulator,
        "sessions_requested": args.sessions,
        "sessions_completed": len(completed),
        "sessions_failed": sum(1 for row in session_summaries if row["status"] != "ok"),
        "avg_assistant_turns": (total_turns / len(session_summaries)) if session_summaries else 0.0,
        "feedback_counts": feedback_counts,
        "memory_guard": memory_guard.stats(),
        "min_gpu_free_mib": args.min_gpu_free_mib,
        "max_concurrent_sessions": args.max_concurrent_sessions,
        "items": session_summaries,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["sessions_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
