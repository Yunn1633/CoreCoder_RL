"""Lightweight ToolPRM-style intra-call scoring for CoreCoder tool use.

This module is intentionally model-agnostic. It does not replace the existing
step-wise tool RL reward; it records finer process labels inside a tool_call so
later runs can train or plug in a learned ToolPRM.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


_DANGEROUS_PYTHON_RE = re.compile(
    r"(__|\bimport\s+os\b|\bimport\s+subprocess\b|\bopen\s*\(|\beval\s*\(|\bexec\s*\(|\bsystem\s*\()",
    re.IGNORECASE,
)


@dataclass
class ToolPRMResult:
    enabled: bool = True
    tool_name_correct: bool | None = None
    arg_valid: bool | None = None
    arg_value_score: float | None = None
    param_finish: bool | None = None
    execution_success: bool | None = None
    observation_useful: bool | None = None
    func_finish: bool | None = None
    total_finish: bool | None = None
    next_action_needed: bool | None = None
    toolprm_score: float = 0.0
    toolprm_reward_hint: int = 0
    stages: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "toolprm_enabled": self.enabled,
            "toolprm_score": self.toolprm_score,
            "toolprm_reward_hint": self.toolprm_reward_hint,
            "tool_name_correct": self.tool_name_correct,
            "arg_valid": self.arg_valid,
            "arg_value_score": self.arg_value_score,
            "param_finish": self.param_finish,
            "execution_success": self.execution_success,
            "observation_useful": self.observation_useful,
            "func_finish": self.func_finish,
            "total_finish": self.total_finish,
            "next_action_needed": self.next_action_needed,
            "toolprm_stages": self.stages,
            "toolprm_reasons": self.reasons,
        }


def _parse_args(raw: Any) -> tuple[dict[str, Any], bool]:
    if isinstance(raw, dict):
        return raw, True
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return (parsed, isinstance(parsed, dict))
        except json.JSONDecodeError:
            return {}, False
    return {}, False


def _normalise_allowed_tools(allowed_tools: Any) -> set[str]:
    if allowed_tools is None:
        return {"python"}
    if isinstance(allowed_tools, str):
        pieces = re.split(r"[,\s]+", allowed_tools.strip())
        return {p for p in pieces if p} or {"python"}
    if isinstance(allowed_tools, (list, tuple, set)):
        return {str(x) for x in allowed_tools if str(x)} or {"python"}
    return {"python"}


def _parse_observation(text: str) -> tuple[bool, str, str]:
    text = text or ""
    try:
        parsed = json.loads(text)
        ok = bool(parsed.get("ok")) and not parsed.get("error")
        output = str(parsed.get("output", ""))
        error = str(parsed.get("error", ""))
        return ok, output, error
    except Exception:
        lowered = text.lower()
        ok = bool(text.strip()) and "error" not in lowered and "traceback" not in lowered
        return ok, text.strip(), ""


class ToolStepPRM:
    """Heuristic ToolPRM facade for function/tool-call process labels.

    The output mirrors ToolPRM-style intra-call granularity: function name,
    argument value, parameter finish, function finish, and total finish. A
    learned/generative PRM can later implement the same ``score`` interface.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def score(
        self,
        *,
        action_text: str,
        tool_calls: list[dict[str, Any]] | None,
        observation_text: str,
        allowed_tools: Any = None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return ToolPRMResult(enabled=False).to_dict()

        calls = tool_calls or []
        allowed = _normalise_allowed_tools(allowed_tools)
        result = ToolPRMResult()
        result.stages = {
            "FUNC_NAME": None,
            "ARG_VALUE": [],
            "PARAM_FINISH": None,
            "FUNC_FINISH": None,
            "TOTAL_FINISH": None,
        }

        if not calls:
            result.reasons.append("no_tool_call")
            return result.to_dict()

        name_scores: list[bool] = []
        arg_scores: list[bool] = []
        param_scores: list[bool] = []
        for call in calls:
            fn = call.get("function") if isinstance(call, dict) else None
            fn = fn if isinstance(fn, dict) else {}
            name = str(fn.get("name", ""))
            args, parsed_ok = _parse_args(fn.get("arguments"))
            name_ok = name in allowed
            name_scores.append(name_ok)
            if not name_ok:
                result.reasons.append(f"tool_name_not_allowed:{name}")

            arg_ok = parsed_ok
            arg_value_score = 1.0 if parsed_ok else 0.0
            if name == "python":
                code = str(args.get("code", "")) if parsed_ok else ""
                has_code = bool(code.strip())
                safe_code = not bool(_DANGEROUS_PYTHON_RE.search(code))
                arg_ok = parsed_ok and has_code and safe_code
                arg_value_score = 1.0 if arg_ok else 0.0
                if not has_code:
                    result.reasons.append("python_code_missing")
                if not safe_code:
                    result.reasons.append("python_code_unsafe")
            arg_scores.append(arg_ok)
            param_scores.append(arg_ok)
            result.stages["ARG_VALUE"].append({
                "tool_name": name,
                "arg_valid": arg_ok,
                "arg_value_score": arg_value_score,
                "arguments": args,
            })

        exec_ok, output, error = _parse_observation(observation_text)
        useful = exec_ok and bool(output.strip())
        func_finish = all(name_scores) and all(param_scores) and exec_ok
        total_finish = func_finish
        next_action_needed = exec_ok

        result.tool_name_correct = all(name_scores)
        result.arg_valid = all(arg_scores)
        result.arg_value_score = sum(x["arg_value_score"] for x in result.stages["ARG_VALUE"]) / max(1, len(result.stages["ARG_VALUE"]))
        result.param_finish = all(param_scores)
        result.execution_success = exec_ok
        result.observation_useful = useful
        result.func_finish = func_finish
        result.total_finish = total_finish
        result.next_action_needed = next_action_needed
        result.stages["FUNC_NAME"] = result.tool_name_correct
        result.stages["PARAM_FINISH"] = result.param_finish
        result.stages["FUNC_FINISH"] = result.func_finish
        result.stages["TOTAL_FINISH"] = result.total_finish
        if error:
            result.reasons.append(f"execution_error:{error[:200]}")
        if exec_ok and not useful:
            result.reasons.append("empty_observation")

        components = [
            float(result.tool_name_correct),
            float(result.arg_valid),
            float(result.execution_success),
            float(result.observation_useful),
            float(result.func_finish),
        ]
        result.toolprm_score = sum(components) / len(components)
        result.toolprm_reward_hint = 1 if result.toolprm_score >= 0.8 else (-1 if result.toolprm_score <= 0.4 else 0)
        return result.to_dict()


if __name__ == "__main__":
    prm = ToolStepPRM()
    demo = prm.score(
        action_text="",
        tool_calls=[{"function": {"name": "python", "arguments": "{\"code\": \"2+2\"}"}}],
        observation_text='{"ok": true, "output": "4"}',
        allowed_tools="python",
    )
    print(json.dumps(demo, ensure_ascii=False, indent=2))
