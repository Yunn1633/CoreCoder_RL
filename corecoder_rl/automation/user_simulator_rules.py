#!/usr/bin/env python3
"""Rule-based user simulator for safe synthetic CoreCoder dialogues."""

from __future__ import annotations

from typing import Any
import random


class RuleBasedUserSimulator:
    """Generate lightweight follow-up user turns without another GPU model."""

    def __init__(self, seed: int = 1234):
        self._rng = random.Random(seed)

    def next_message(
        self,
        scenario: dict[str, Any],
        transcript: list[dict[str, str]],
        assistant_reply: str,
        turn_index: int,
    ) -> tuple[str, bool]:
        """Return (message, should_finish_after_this_turn)."""
        target_turns = int(scenario.get("target_turns", 3))
        domain = str(scenario.get("domain", "general"))
        normalized_reply = assistant_reply.strip()

        if turn_index >= max(target_turns - 1, 1):
            return self._build_closing_message(domain, normalized_reply), True

        return self._build_follow_up(domain, transcript, normalized_reply), False

    def _build_follow_up(
        self,
        domain: str,
        transcript: list[dict[str, str]],
        assistant_reply: str,
    ) -> str:
        too_short = len(assistant_reply) < 80
        asked_list = any(ch in assistant_reply for ch in ["1.", "2.", "3.", "-", "先", "然后"])
        recent_user = transcript[-1]["content"] if transcript else ""

        generic_prompts = [
            "请你把刚才的结论再压缩成一个可以直接执行的检查顺序。",
            "如果我只想先做最关键的一步，你建议先做哪一步，为什么？",
            "你能补一个最常见失败点和对应修复方法吗？",
        ]

        by_domain = {
            "environment_setup": [
                "如果我现在只看终端输出，最值得先观察哪三行？",
                "能不能把你的建议拆成『先确认』『再定位』『最后修复』三步？",
            ],
            "documentation": [
                "请再举一个运行依赖和编译依赖的对比例子。",
                "如果我只按最小可运行去装，哪些依赖可以后补？",
            ],
            "operations": [
                "如果问题是偶发性的，你会建议我补哪些监控或日志点？",
                "能不能给一个从网络、鉴权到模型服务的排查顺序？",
            ],
            "code_explanation": [
                "请再用一个更具体的小例子说明一下。",
                "如果我完全不看代码，只记住一句话，应该记住什么？",
            ],
            "cli_usage": [
                "如果我想把它做成脚本循环运行，最少要固定哪些参数？",
                "能不能补一个更适合自动化调用的命令示例？",
            ],
            "resource_planning": [
                "如果我只有有限磁盘，你建议保留最近多少个 checkpoint？",
                "除了 checkpoint，还有哪些目录会悄悄占空间？",
            ],
        }

        candidates = list(by_domain.get(domain, [])) + generic_prompts
        if too_short:
            candidates.insert(0, "你刚才说得有点短，请展开一点，并给出更具体的判断标准。")
        if not asked_list:
            candidates.insert(0, "请把答案整理成更清晰的分点列表，方便我直接照着做。")
        if recent_user and recent_user in candidates:
            candidates = [c for c in candidates if c != recent_user] or generic_prompts
        return self._rng.choice(candidates)

    def _build_closing_message(self, domain: str, assistant_reply: str) -> str:
        if domain in {"operations", "environment_setup"}:
            return "明白了，这已经足够我继续排查了。请再用一句话总结最关键的动作，然后我们结束这轮。"
        if len(assistant_reply) < 50:
            return "好的，请最后再补一句最重要的结论，我们就结束这轮。"
        return "好的，我已经理解了。请最后用一句话概括核心结论，然后结束这轮。"
