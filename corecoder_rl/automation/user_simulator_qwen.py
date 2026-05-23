#!/usr/bin/env python3
"""OpenAI-compatible user simulator for synthetic CoreCoder-RL conversations."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any


class QwenUserSimulator:
    """Generate next-state user turns from an OpenAI-compatible endpoint.

    The simulator is intended to run against an external small model so this
    process does not allocate local GPU memory.  It emits only the next user
    utterance; metadata is inferred locally so the transcript stays natural.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int = 120,
        max_tokens: int | None = None,
    ):
        self.base_url = (base_url or os.getenv("CORECODER_SYNTH_USER_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("CORECODER_SYNTH_USER_API_KEY", "")
        self.model = model or os.getenv("CORECODER_SYNTH_USER_MODEL", "")
        self.timeout = timeout
        self.max_tokens = int(max_tokens or os.getenv("CORECODER_SYNTH_USER_MAX_TOKENS", "256"))
        if not self.base_url or not self.model:
            raise ValueError(
                "QwenUserSimulator requires CORECODER_SYNTH_USER_BASE_URL and "
                "CORECODER_SYNTH_USER_MODEL."
            )

    def next_message(
        self,
        scenario: dict[str, Any],
        transcript: list[dict[str, str]],
        assistant_reply: str,
        turn_index: int,
    ) -> tuple[str, bool, str, dict[str, Any]]:
        target_turns = int(scenario.get("target_turns", 3))
        should_finish = turn_index >= max(target_turns - 1, 1)
        feedback_types = scenario.get("feedback_types") or [
            "evaluative",
            "directive",
            "requery",
            "preference",
            "terminal_state",
        ]
        mode = self._scenario_mode(scenario)
        system_prompt = self._system_prompt(mode, should_finish)
        user_prompt = {
            "mode": mode,
            "scenario_id": scenario.get("scenario_id"),
            "domain": scenario.get("domain"),
            "role_pattern": scenario.get("role_pattern"),
            "reference_question": scenario.get("reference_question"),
            "reference_final_answer": scenario.get("reference_final_answer"),
            "turn_index": turn_index,
            "should_finish": should_finish,
            "assistant_reply": assistant_reply,
            "transcript": transcript[-6:],
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
            ],
            "temperature": 0.7,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        message = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        next_message = self._clean_plain_message(message)
        if not next_message:
            next_message = self._fallback_message(mode, should_finish)
        feedback_type = self._infer_feedback_type(next_message, feedback_types)
        raw = {
            "response": data,
            "content": message,
            "mode": mode,
            "plain_message": next_message,
            "inferred_feedback_type": feedback_type,
        }
        return next_message, should_finish, feedback_type, raw

    @staticmethod
    def _scenario_mode(scenario: dict[str, Any]) -> str:
        role_pattern = str(scenario.get("role_pattern", "")).lower()
        domain = str(scenario.get("domain", "")).lower()
        if "teacher_to" in role_pattern or domain.startswith("gsm8k_teacher") or domain.startswith("teacher_"):
            return "teacher_mode"
        return "student_mode"

    @staticmethod
    def _system_prompt(mode: str, should_finish: bool) -> str:
        common = (
            "请用中文输出。你只输出下一轮真实用户话语，不要输出分析，不要输出评分，"
            "不要输出 JSON，不要加引号，不要加标签。保持安全，不要求破坏性操作、窃取密钥或修改仓库。"
            "只围绕当前这一道题或当前这一份作业继续反馈。"
        )
        if mode == "teacher_mode":
            finish_hint = "如果回答还可以，请表达满意或要求整理成最终评语。" if should_finish else ""
            return (
                "你正在模拟一个真实老师。你使用 CoreCoder 批改学生作业。"
                "你希望反馈具体、友好、能帮助学生改进。"
                "你不喜欢过于冷淡、只给分数、没有解释的批改。"
                "你看到 CoreCoder 的回答后，请生成下一轮真实用户反馈。"
                "如果评语不够具体或不够友好，请自然地指出。"
                f"{finish_hint}"
                "话语要自然、简短，像老师真的在继续提要求。"
                f"{common}"
            )
        finish_hint = "如果答案还可以，请表达满意或提出一个小修改。" if should_finish else ""
        return (
            "你正在模拟一个真实学生。你正在使用 AI 帮你完成作业。"
            "你不希望答案看起来像 AI 生成。"
            "你偏好自然、简短、不那么模板化的表达。"
            "你看到 CoreCoder 的回答后，请生成下一轮真实用户反馈。"
            "如果答案太 AI-like，请自然地指出。"
            f"{finish_hint}"
            "话语要像学生本人在聊天，不要像评测员。"
            f"{common}"
        )

    @staticmethod
    def _clean_plain_message(message: str) -> str:
        cleaned = message.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict) and isinstance(parsed.get("message"), str):
            cleaned = parsed["message"].strip()
        return cleaned.strip().strip("\"'")

    @staticmethod
    def _fallback_message(mode: str, should_finish: bool) -> str:
        if mode == "teacher_mode":
            return "这个方向可以，再帮我整理成一段具体、友好的最终评语。" if should_finish else "这段有点笼统，能不能更具体地指出学生哪一步做得好、哪里要改？"
        return "这样可以，但帮我改得更像我自己写的，别太像 AI。" if should_finish else "我有点懂了，不过能不能说得更自然短一点？"

    @staticmethod
    def _infer_feedback_type(message: str, allowed: list[str]) -> str:
        lowered = message.lower()
        if any(token in message for token in ["太像 AI", "像AI", "模板", "自然", "口语", "不像我"]):
            guess = "preference"
        elif any(token in message for token in ["能不能", "帮我", "改", "整理", "具体", "友好", "短一点"]):
            guess = "directive"
        elif any(token in message for token in ["可以", "满意", "对了", "懂了", "不错", "这样行"]):
            guess = "evaluative"
        elif any(token in message for token in ["最后", "最终", "就这样", "不用再"]):
            guess = "terminal_state"
        else:
            guess = "requery"
        return guess if guess in set(allowed) else (allowed[0] if allowed else "requery")
