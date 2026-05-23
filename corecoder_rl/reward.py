from __future__ import annotations

import re
from typing import Any

_NEG = re.compile(r"redo|retry|again|fix|wrong|incorrect|not that|change|modify|revise|failed|error", re.I)
_POS = re.compile(r"thanks|thank you|great|good|works|done|correct|success", re.I)


def flatten_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(i.get("text", "")) for i in content if isinstance(i, dict) and i.get("type") == "text")
    return "" if content is None else str(content)


def rule_next_state_reward(next_state: dict[str, Any] | None) -> tuple[float, str]:
    if not next_state:
        return 0.0, "no_next_state"
    role = str(next_state.get("role", "user"))
    text = flatten_message_content(next_state.get("content"))
    low = text.lower()
    if role == "tool":
        if any(x in low for x in ("error", "traceback", "exception", "failed")):
            return -1.0, "tool_error"
        return (1.0, "tool_success") if text.strip() else (0.0, "empty_tool_result")
    if _NEG.search(text):
        return -1.0, "user_correction"
    if _POS.search(text):
        return 1.0, "user_positive"
    return 0.0, "ambiguous"
