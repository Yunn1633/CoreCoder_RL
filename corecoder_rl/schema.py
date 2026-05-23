from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import time


@dataclass
class TurnRecord:
    session_id: str
    turn_index: int
    prompt_messages: list[dict[str, Any]]
    assistant_response: dict[str, Any]
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    next_state: dict[str, Any] | None = None
    reward: float | None = None
    loss_mask: int = 1
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    def to_json(self) -> dict[str, Any]:
        return asdict(self)
