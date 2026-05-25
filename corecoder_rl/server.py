import asyncio
import collections
import json
import logging
import os
import queue
import re
import threading
import time
from itertools import count
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_RESET = "\033[0m"
logger = logging.getLogger(__name__)

_BOXED_RE = re.compile(r"\\boxed\{([-+]?\d)\}")

_NON_STANDARD_BODY_KEYS = {"session_id", "session_done", "turn_type", "scenario_id", "student_mode", "student_model", "feeder_id", "rl_method", "prompt_source", "extra_prompt_model", "requires_tool", "allowed_tools", "tool_call_count", "tool_success", "tool_name", "final_correct", "reference_answer", "question", "checker"}


def _flatten_message_content(content):
    """Extract plain text from multimodal content lists."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
        return " ".join(parts) if parts else ""
    return str(content) if content is not None else ""


def _normalize_messages_for_template(messages: list[dict]) -> list[dict]:
    """Make messages compatible with the chat template.

    - developer → system (templates only know 'system')
    - multimodal content lists → plain text strings
    - tool_call arguments: JSON string → dict (for Jinja2 |items)
    """
    out = []
    for msg in messages:
        m = dict(msg)
        if m.get("role") == "developer":
            m["role"] = "system"
        raw = m.get("content")
        if not isinstance(raw, str) and raw is not None:
            m["content"] = _flatten_message_content(raw)
        if m.get("tool_calls"):
            m["tool_calls"] = [_normalize_tool_call(tc) for tc in m["tool_calls"]]
        out.append(m)
    return out


def _normalize_tool_call(tc: dict) -> dict:
    """Ensure tool_call.function.arguments is a dict so Jinja2 |items works."""
    tc = dict(tc)
    fn = tc.get("function")
    if isinstance(fn, dict):
        fn = dict(fn)
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                fn["arguments"] = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                fn["arguments"] = {}
        tc["function"] = fn
    return tc


def _extract_logprobs_from_chat_response(choice: dict[str, Any]) -> list[float]:
    logprobs_obj = choice.get("logprobs")
    if not isinstance(logprobs_obj, dict):
        return []
    content = logprobs_obj.get("content")
    if not isinstance(content, list):
        return []
    return [float(item.get("logprob", 0.0)) for item in content if isinstance(item, dict)]


def _build_prm_judge_prompt(response_text: str, next_state_text: str, next_state_role: str = "user") -> list[dict]:
    system = (
        "You are a process reward model (PRM) evaluating an AI assistant.\n"
        "You will see the assistant's output and the subsequent next state.\n"
        "Your task: decide whether the assistant's output **successfully fulfilled** the user's intent "
        "at that step, using the next state as evidence.\n\n"
        "## Understanding the next state's role\n"
        "- role='user': A reply from the user.\n"
        "- role='tool': The return value of a tool the assistant invoked. "
        "This content was NOT available before the assistant's action — "
        "it exists BECAUSE the assistant called the tool. "
        "A successful, non-error tool output means the assistant's action worked correctly "
        "and should be scored positively.\n\n"
        "## Scoring rules\n"
        "- \\boxed{1} (good): The next state shows the task progressed as expected — "
        "e.g. the user moves on, says thanks, the environment confirms success, "
        "or a tool returns a successful, non-error result.\n"
        "- \\boxed{-1} (bad): The next state signals the assistant's output was wrong, "
        "incomplete, or unwanted. **Key negative signals include:**\n"
        "  * The user asks the assistant to **redo, retry, or repeat** the same action "
        "(\"do it again\", \"try again\", \"one more time\").\n"
        "  * The user requests a **correction or modification** to what the assistant just did "
        "(\"change X to Y\", \"no, I meant …\", \"not that, …\", \"please fix …\").\n"
        "  * The user **rephrases or restates** the same request, implying the assistant "
        "did not understand or execute it correctly.\n"
        "  * The environment returns an **error, failure, or unexpected result** caused "
        "by the assistant's action.\n"
        "- \\boxed{0} (neutral): The next state is ambiguous — e.g. the user gives an "
        "unrelated follow-up that neither confirms nor denies success, or there is "
        "insufficient information to judge.\n\n"
        "## Important\n"
        "A change request IS negative feedback — it means the previous output did not "
        "meet the user's need. Do NOT treat it as a neutral new instruction.\n\n"
        "Think step-by-step, then give your final score inside \\boxed{}."
    )
    user = (
        f"## Assistant output\n{response_text}\n\n"
        f"## Next state [role: {next_state_role}]\n{next_state_text}\n\n"
        "First, classify the next state: is it (a) positive progression, "
        "(b) a correction / redo / change request, or (c) ambiguous? "
        "Then assign \\boxed{1}, \\boxed{-1}, or \\boxed{0}."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_prm_score(text: str) -> int | None:
    matches = _BOXED_RE.findall(text)
    if not matches:
        return None
    val = int(matches[-1])
    if val in (1, -1, 0):
        return val
    return None


def _majority_vote(scores: list[int | None]) -> float:
    valid = [s for s in scores if s is not None]
    if not valid:
        return 0.0
    counter = collections.Counter(valid)
    top = counter.most_common(1)[0]
    if list(counter.values()).count(top[1]) > 1:
        return 0.0
    return float(top[0])


def _extract_topk_logprobs_from_meta(meta: dict[str, Any], response_len: int) -> tuple[list[list[float]], list[list[int]]]:
    raw_rows = meta.get("input_top_logprobs", [])
    if not isinstance(raw_rows, list) or response_len <= 0:
        return [], []
    rows = raw_rows[-response_len:]
    topk_log_probs: list[list[float]] = []
    topk_indices: list[list[int]] = []
    for row in rows:
        vals: list[float] = []
        ids: list[int] = []
        if isinstance(row, list):
            for item in row:
                if isinstance(item, (list, tuple)) and len(item) >= 2 and item[0] is not None and item[1] is not None:
                    try:
                        vals.append(float(item[0]))
                        ids.append(int(item[1]))
                    except (TypeError, ValueError):
                        continue
        topk_log_probs.append(vals)
        topk_indices.append(ids)
    return topk_log_probs, topk_indices


async def reward_func(args, sample_or_samples, **kwargs):
    if isinstance(sample_or_samples, list):
        return [{"score": s.reward.get("score", 0.0) if isinstance(s.reward, dict) else 0.0}
                for s in sample_or_samples]
    s = sample_or_samples
    return {"score": s.reward.get("score", 0.0) if isinstance(s.reward, dict) else 0.0}


async def generate(args, sample: Sample, sampling_params, evaluation: bool = False) -> Sample:
    tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    messages = sample.prompt if isinstance(sample.prompt, list) else [{"role": "user", "content": str(sample.prompt)}]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    payload = {
        "input_ids": input_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        output = response.json()
    text = output.get("text", "")
    meta = output.get("meta_info", {})
    pairs = meta.get("output_token_logprobs", [])
    if isinstance(pairs, list) and pairs:
        token_ids = [int(p[1]) for p in pairs if isinstance(p, (list, tuple)) and len(p) >= 2]
        logprobs = [float(p[0]) for p in pairs if isinstance(p, (list, tuple)) and len(p) >= 2]
    else:
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        logprobs = [0.0] * len(token_ids)
    sample.tokens = input_ids + token_ids
    sample.response = text
    sample.response_length = len(token_ids)
    sample.rollout_log_probs = logprobs
    sample.loss_mask = [1] * len(token_ids)
    sample.status = Sample.Status.COMPLETED
    return sample


class CoreCoderAPIServer:
    """Proxy between CoreCoder and SGLang for RL training data collection.

    CoreCoder sends ``X-Session-Id`` and ``X-Turn-Type`` headers with every
    request.  The proxy simply forwards to SGLang, and when ``turn_type``
    is ``"main"`` it tokenises the full prompt+response and submits a
    training sample.  Side tasks (``turn_type != "main"``) are forwarded
    but produce no training data.
    """

    def __init__(self, args, output_queue: queue.Queue, submission_enabled: threading.Event):
        self.args = args
        self.output_queue = output_queue
        self.submission_enabled = submission_enabled
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.sglang_chat_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/v1/chat/completions"
        self.sglang_health_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/health"
        self.expected_api_key = os.getenv("SGLANG_API_KEY", "")
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "30000"))
        self.served_model_name = os.getenv("SERVED_MODEL_NAME", "qwen3-8b")

        self._index_counter = count(0)
        self._group_counter = count(0)
        self._turn_counts: dict[str, int] = {}
        self._pending_records: dict[str, dict] = {}
        self._prm_tasks: dict[str, dict[int, asyncio.Task]] = {}  # session → {turn → task}
        self._pending_turn_data: dict[str, dict[int, dict]] = {}  # session → {turn → turn_data}
        self._session_metadata: dict[str, dict[str, Any]] = {}  # session -> feeder/scenario metadata
        self._session_effective: dict[str, int] = {}  # session → count of samples with loss_mask=[1]

        self._teacher_logprob_enabled = os.getenv("CORECODER_TEACHER_LOGPROB_ENABLED", "1").lower() not in {"0", "false", "no", "off"}
        self._teacher_logprob_prompt = os.getenv(
            "CORECODER_TEACHER_LOGPROB_PROMPT",
            "You are an expert teacher model. Use the teacher extra instruction and task context "
            "to evaluate the same student-generated trajectory. Do not generate a new response here."
        )
        self._teacher_use_reference_answer = os.getenv(
            "CORECODER_TEACHER_USE_REFERENCE_ANSWER", "1"
        ).lower() not in {"0", "false", "no", "off"}
        self._teacher_extra_prompt_enabled = os.getenv(
            "CORECODER_TEACHER_EXTRA_PROMPT_ENABLED", "1"
        ).lower() not in {"0", "false", "no", "off"}
        self._teacher_extra_prompt_model = os.getenv("CORECODER_TEACHER_EXTRA_PROMPT_MODEL", "deepseek-v4-flash")
        self._teacher_extra_prompt_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
        self._teacher_extra_prompt_api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not self._teacher_extra_prompt_api_key:
            try:
                self._teacher_extra_prompt_api_key = open("/root/.corecoder_deepseek_api_key", encoding="utf-8").read().strip()
            except OSError:
                self._teacher_extra_prompt_api_key = ""
        self._teacher_extra_prompt_cache: dict[str, str] = {}
        self._teacher_topk = int(os.getenv("CORECODER_OPSD_TEACHER_TOPK", "32"))

        self._prm_enabled = getattr(args, "prm_enable", False)
        self._prm_m = int(os.getenv("PRM_M", getattr(args, "prm_m", 3)))
        self._prm_temperature = float(getattr(args, "prm_temperature", 0.6))
        self._prm_max_tokens = int(getattr(args, "prm_max_new_tokens", 4096))
        prm_ip = getattr(args, "prm_router_ip", None)
        prm_port = getattr(args, "prm_router_port", None)
        self._prm_url = f"http://{prm_ip}:{prm_port}/generate" if prm_ip and prm_port else ""
        self._prm_tokenizer = None
        if self._prm_enabled:
            prm_path = getattr(args, "prm_model_path", None) or args.hf_checkpoint
            self._prm_tokenizer = load_tokenizer(prm_path, trust_remote_code=True)
            logger.info("[CoreCoder] PRM enabled: url=%s m=%d", self._prm_url, self._prm_m)

        self._eval_scores: list[float] = []
        self._eval_scores_lock = threading.Lock()

        self._record_file = os.getenv("CORECODER_RECORD_FILE", "") if os.getenv("CORECODER_RECORD_ENABLED", "0") == "1" else ""
        if self._record_file:
            os.makedirs(os.path.dirname(self._record_file), exist_ok=True)
            open(self._record_file, "w").close()
            logger.info("[CoreCoder] record file initialized (cleared): %s", self._record_file)

        self._prm_record_file = os.getenv("CORECODER_PRM_RECORD_FILE", "")
        if not self._prm_record_file and self._record_file and self._prm_enabled:
            base, ext = os.path.splitext(self._record_file)
            self._prm_record_file = f"{base}_prm{ext}"
        if self._prm_record_file:
            os.makedirs(os.path.dirname(self._prm_record_file), exist_ok=True)
            open(self._prm_record_file, "w").close()
            logger.info("[CoreCoder] PRM record file initialized (cleared): %s", self._prm_record_file)

        default_metrics = ""
        if self._record_file:
            default_metrics = os.path.join(os.path.dirname(self._record_file), "corecoder_rl_metrics.jsonl")
        self._metrics_file = os.getenv("CORECODER_METRICS_FILE", default_metrics)
        if self._metrics_file:
            os.makedirs(os.path.dirname(self._metrics_file), exist_ok=True)
            logger.info("[CoreCoder] metrics file append target: %s", self._metrics_file)

        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self.app = self._build_app()

    # ------------------------------------------------------------------ app
    def _build_app(self) -> FastAPI:
        app = FastAPI(title="CoreCoder SLIME Proxy")
        app.state.owner = self

        @app.get("/healthz")
        async def healthz():
            return {"ok": True}

        @app.post("/v1/chat/completions")
        async def chat_completions(
            request: Request,
            authorization: str | None = Header(default=None),
            x_session_id: str | None = Header(default=None),
            x_turn_type: str | None = Header(default=None),
            x_session_done: str | None = Header(default=None),
        ):
            owner: CoreCoderAPIServer = request.app.state.owner
            await owner._check_auth(authorization)
            if not owner.submission_enabled.is_set():
                raise HTTPException(status_code=503, detail="submission paused for weight update")

            body = await request.json()
            session_id = x_session_id or body.get("session_id") or "unknown"
            turn_type = (x_turn_type or body.get("turn_type") or "side").strip().lower()
            session_done = (
                (x_session_done and x_session_done.strip().lower() in {"1", "true", "yes", "on"})
                or str(body.get("session_done", "")).strip().lower() in {"1", "true", "yes", "on"}
            )

            stream = bool(body.get("stream", False))
            result = await owner._handle_request(
                body, session_id=session_id, turn_type=turn_type, session_done=session_done,
            )
            if stream:
                return StreamingResponse(owner._stream_response(result), media_type="text/event-stream")
            return JSONResponse(content=result["response"])

        @app.post("/corecoder/session_done")
        async def corecoder_session_done(request: Request, authorization: str | None = Header(default=None)):
            owner: CoreCoderAPIServer = request.app.state.owner
            await owner._check_auth(authorization)
            body = await request.json()
            return JSONResponse(content=owner._finalize_agent_session(body))

        return app

    async def _check_auth(self, authorization: str | None):
        if not self.expected_api_key:
            return
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        token = authorization.split(" ", 1)[1].strip()
        if token != self.expected_api_key:
            raise HTTPException(status_code=401, detail="invalid api key")

    def _metadata_from_body(self, body: dict[str, Any]) -> dict[str, Any]:
        request_metadata = {
            "scenario_id": body.get("scenario_id"),
            "student_mode": body.get("student_mode"),
            "student_model": body.get("student_model"),
            "feeder_id": body.get("feeder_id"),
            "rl_method": body.get("rl_method"),
            "prompt_source": body.get("prompt_source"),
            "extra_prompt_model": body.get("extra_prompt_model"),
            "requires_tool": body.get("requires_tool"),
            "allowed_tools": body.get("allowed_tools"),
            "tool_call_count": body.get("tool_call_count"),
            "tool_success": body.get("tool_success"),
            "tool_name": body.get("tool_name"),
            "final_correct": body.get("final_correct"),
            "reference_answer": body.get("reference_answer"),
            "question": body.get("question"),
            "checker": body.get("checker"),
        }
        return {k: v for k, v in request_metadata.items() if v not in (None, "")}

    def _finalize_agent_session(self, body: dict[str, Any]) -> dict[str, Any]:
        session_id = body.get("session_id") or "unknown"
        metadata = self._metadata_from_body(body)
        if metadata:
            self._session_metadata.setdefault(session_id, {}).update(metadata)
        try:
            score = float(body.get("reward"))
        except (TypeError, ValueError):
            score = 1.0 if bool(body.get("final_correct")) else 0.0
        pending = self._pending_turn_data.get(session_id, {})
        if pending:
            turn_num = max(pending)
            pending[turn_num]["deterministic_score"] = score
            pending[turn_num]["has_next_state"] = True
            pending[turn_num].setdefault("metadata", {}).update(metadata)
        self._flush_pending_record(session_id, None)
        self._maybe_submit_ready_samples(session_id, force_no_prm=True)
        eff = self._session_effective.pop(session_id, 0)
        self._turn_counts.pop(session_id, None)
        logger.info("[CoreCoder] session=%s finalized by CoreCoder Agent (score=%.1f effective_samples=%d)", session_id, score, eff)
        return {"ok": True, "session_id": session_id, "score": score, "effective_samples": eff}

    # ---------------------------------------------------- record file
    def _flush_pending_record(self, session_id: str, next_state):
        """Write out the buffered record for *session_id* with its next_state and fire PRM."""
        rec = self._pending_records.pop(session_id, None)
        if rec is None:
            return
        rec["next_state"] = next_state
        if next_state:
            ns_role = next_state.get("role", "?")
            ns_content = _flatten_message_content(next_state.get("content"))
            logger.info(
                f"{_GREEN}[CoreCoder] session={session_id} turn={rec['turn']} "
                f"next_state role={ns_role} len={len(ns_content)}: "
                f"{ns_content[:200]}{_RESET}"
            )
            if not self._score_tool_next_state(session_id, rec["turn"], next_state):
                self._fire_prm_scoring(session_id, rec["turn"], rec["response_text"], next_state)
        if self._record_file:
            try:
                with open(self._record_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except OSError as e:
                logger.warning("[CoreCoder] failed to write record: %s", e)

    def _buffer_record(self, session_id: str, turn_num: int, messages: list,
                       prompt_text: str, response_text: str, tool_calls: list):
        if not self._record_file:
            return
        self._pending_records[session_id] = {
            "session_id": session_id,
            "turn": turn_num,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "messages": messages,
            "prompt_text": prompt_text,
            "response_text": response_text,
            "tool_calls": tool_calls or None,
        }

    def _append_prm_record(self, session_id: str, turn_num: int,
                           score: float, votes: list, representative: str):
        if not self._prm_record_file:
            return
        rec: dict[str, Any] = {
            "session_id": session_id,
            "turn": turn_num,
            "score": score,
            "votes": votes,
        }
        if score != 0.0 and representative:
            rec["representative_eval"] = representative
        try:
            with open(self._prm_record_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning("[CoreCoder] failed to write PRM record: %s", e)

    def drain_eval_scores(self) -> list[float]:
        with self._eval_scores_lock:
            scores = list(self._eval_scores)
            self._eval_scores.clear()
            return scores

    def reset_eval_scores(self):
        with self._eval_scores_lock:
            self._eval_scores.clear()

    # ---------------------------------------------------- record purge
    def purge_record_files(self):
        """Clear all record JSONL files. Called when training starts."""
        for path, label in [
            (self._record_file, "record"),
            (self._prm_record_file, "PRM record"),
        ]:
            if not path:
                continue
            try:
                open(path, "w").close()
                logger.info("[CoreCoder] %s file purged: %s", label, path)
            except OSError as e:
                logger.warning("[CoreCoder] failed to purge %s file: %s", label, e)

    # ---------------------------------------------------- PRM scoring
    async def _query_prm_once(self, judge_prompt: str, vote_id: int) -> tuple[int | None, str]:
        if not self._prm_url:
            return None, ""
        payload = {
            "text": judge_prompt,
            "sampling_params": {
                "temperature": self._prm_temperature,
                "top_p": 1.0,
                "top_k": -1,
                "max_new_tokens": self._prm_max_tokens,
                "skip_special_tokens": False,
                "no_stop_trim": True,
                "spaces_between_special_tokens": False,
            },
            "return_logprob": False,
        }
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                resp = await client.post(self._prm_url, json=payload)
                resp.raise_for_status()
                data = resp.json()
            raw = data.get("text", data) if isinstance(data, dict) else str(data)
            if isinstance(raw, list):
                raw = raw[0] if raw else ""
            return _parse_prm_score(str(raw)), str(raw)
        except Exception as e:
            logger.warning("[CoreCoder] PRM query failed (vote %d): %s", vote_id, e)
            return None, ""

    async def _prm_evaluate(self, session_id: str, turn_num: int,
                            response_text: str, next_state) -> dict:
        ns_text = _flatten_message_content(next_state.get("content")) if next_state else ""
        ns_role = next_state.get("role", "user") if next_state else "user"
        msgs = _build_prm_judge_prompt(response_text, ns_text, ns_role)
        if self._prm_tokenizer:
            judge_prompt = self._prm_tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
        else:
            judge_prompt = "\n".join(m["content"] for m in msgs)

        results = await asyncio.gather(
            *[self._query_prm_once(judge_prompt, i) for i in range(self._prm_m)]
        )
        scores = [r[0] for r in results]
        final = _majority_vote(scores)

        representative = ""
        if final != 0.0:
            for s, text in results:
                if s is not None and s == int(final):
                    representative = text
                    break

        votes_display = [s if s is not None else "fail" for s in scores]
        logger.info(
            f"{_CYAN}[CoreCoder] PRM session={session_id} turn={turn_num} "
            f"votes={votes_display} → score={final}{_RESET}"
        )
        self._append_prm_record(session_id, turn_num, final, votes_display, representative)
        return {"score": final, "votes": votes_display, "representative_eval": representative}

    def _score_tool_next_state(self, session_id: str, turn_num: int, next_state) -> bool:
        """Apply deterministic process reward for tool observations.

        A non-error tool result is exactly the environmental confirmation we need
        for tool-call training, so this path bypasses PRM and submits the
        assistant tool_call tokens directly into OPSD.
        """
        if not next_state or next_state.get("role") != "tool":
            return False
        td = self._pending_turn_data.get(session_id, {}).get(turn_num)
        if td is None:
            return True
        content = _flatten_message_content(next_state.get("content"))
        ok = False
        try:
            parsed = json.loads(content)
            ok = bool(parsed.get("ok")) and not parsed.get("error")
        except Exception:
            ok = "error" not in content.lower() and bool(content.strip())
        td["has_next_state"] = True
        td["deterministic_score"] = 1.0 if ok else -1.0
        logger.info("[CoreCoder] tool next_state session=%s turn=%d deterministic_score=%.1f", session_id, turn_num, td["deterministic_score"])
        self._maybe_submit_ready_samples(session_id, force_no_prm=True)
        return True

    def _fire_prm_scoring(self, session_id: str, turn_num: int,
                          response_text: str, next_state):
        if not self._prm_enabled or not next_state:
            return
        task = asyncio.create_task(
            self._prm_evaluate(session_id, turn_num, response_text, next_state)
        )
        task.add_done_callback(self._task_done_cb)
        task.add_done_callback(lambda _t: self._maybe_submit_ready_samples(session_id))
        self._prm_tasks.setdefault(session_id, {})[turn_num] = task
        td = self._pending_turn_data.get(session_id, {}).get(turn_num)
        if td is not None:
            td["has_next_state"] = True

    # ---------------------------------------------------- teacher trajectory scoring
    def _render_messages_for_teacher(self, messages: list, response_msg: dict | None = None) -> str:
        rows = []
        for msg in list(messages) + ([response_msg] if response_msg else []):
            if not msg:
                continue
            role = msg.get("role", "?")
            content = _flatten_message_content(msg.get("content"))
            reasoning = msg.get("reasoning_content")
            if reasoning:
                content = f"[reasoning] {reasoning}\n[content] {content}"
            calls = msg.get("tool_calls")
            if calls:
                content += "\n[tool_calls] " + json.dumps(calls, ensure_ascii=False)
            rows.append(f"{role}: {content}")
        return "\n\n".join(rows)

    def _build_teacher_extra_prompt_messages(self, messages: list, response_msg: dict, metadata: dict[str, Any]) -> list[dict]:
        question = metadata.get("question", "")
        reference = metadata.get("reference_answer", "") if self._teacher_use_reference_answer else ""
        allowed_tools = metadata.get("allowed_tools", "python")
        trajectory = self._render_messages_for_teacher(messages, response_msg)
        system = (
            "You are a teacher model helping improve a math tool-use agent. "
            "Given the task, available tools, optional reference answer, and the student's trajectory so far, "
            "write one short extra instruction that would help the student produce a better answer. "
            "Focus on whether a tool is useful, what to compute/check, and how to avoid mistakes. "
            "Do not solve the whole problem. Do not include analysis or hidden reasoning. "
            "Do not replace the student's response. Output only the extra instruction."
        )
        user = (
            f"Task question:\n{question}\n\n"
            f"Reference answer, visible only to you:\n{reference}\n\n"
            f"Available tools:\n{allowed_tools}\n\n"
            f"Student trajectory so far:\n{trajectory}\n\n"
            "Extra instruction:"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    async def _generate_teacher_extra_prompt(self, messages: list, response_msg: dict, metadata: dict[str, Any]) -> str:
        if not self._teacher_extra_prompt_enabled:
            return ""
        cache_key = json.dumps({
            "messages": messages,
            "response": response_msg,
            "question": metadata.get("question"),
            "reference_answer": metadata.get("reference_answer") if self._teacher_use_reference_answer else None,
            "allowed_tools": metadata.get("allowed_tools"),
        }, ensure_ascii=False, sort_keys=True, default=str)
        if cache_key in self._teacher_extra_prompt_cache:
            return self._teacher_extra_prompt_cache[cache_key]
        if not self._teacher_extra_prompt_api_key:
            logger.warning("[CoreCoder] teacher extra prompt disabled: missing DEEPSEEK_API_KEY")
            return ""
        payload = {
            "model": self._teacher_extra_prompt_model,
            "messages": self._build_teacher_extra_prompt_messages(messages, response_msg, metadata),
            "temperature": 0.2,
            "max_tokens": 512,
        }
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"{self._teacher_extra_prompt_base_url}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self._teacher_extra_prompt_api_key}"},
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
            msg = ((data.get("choices") or [{}])[0].get("message") or {})
            text = (msg.get("content") or "").strip()
            if not text:
                text = (msg.get("reasoning_content") or "").strip()
            text = " ".join(text.split())[:1000]
            self._teacher_extra_prompt_cache[cache_key] = text
            logger.info("[CoreCoder] teacher extra prompt (%s): %s", self._teacher_extra_prompt_model, text[:300])
            return text
        except Exception as e:
            logger.warning("[CoreCoder] teacher extra prompt generation failed: %s", e)
            return ""

    def _build_teacher_context(self, metadata: dict[str, Any], extra_prompt: str = "") -> str:
        parts = [self._teacher_logprob_prompt]
        if extra_prompt:
            parts.append(f"Teacher extra instruction:\n{extra_prompt}")
        question = metadata.get("question")
        if question:
            parts.append(f"Task question:\n{question}")
        if self._teacher_use_reference_answer and metadata.get("reference_answer"):
            parts.append(f"Reference answer, visible only to the teacher:\n{metadata['reference_answer']}")
        if metadata.get("allowed_tools"):
            parts.append(f"Allowed tools: {metadata['allowed_tools']}")
        rubric = metadata.get("checker") or metadata.get("rubric")
        if rubric:
            parts.append(f"Rubric/checker: {rubric}")
        return "\n\n".join(str(p) for p in parts if p is not None)

    def _build_teacher_scoring_ids(self, messages: list, response_msg: dict, tools, metadata: dict[str, Any], extra_prompt: str = "") -> tuple[list[int], list[int]]:
        system_msg = {"role": "system", "content": self._build_teacher_context(metadata, extra_prompt)}
        norm_msgs = _normalize_messages_for_template([system_msg] + messages)
        norm_resp = _normalize_messages_for_template([response_msg])[0]
        full_norm = norm_msgs + [norm_resp]
        prompt_text = self.tokenizer.apply_chat_template(
            norm_msgs, tools=tools, tokenize=False, add_generation_prompt=True,
        )
        full_text = self.tokenizer.apply_chat_template(
            full_norm, tools=tools, tokenize=False, add_generation_prompt=False,
        )
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        return prompt_ids, full_ids

    async def _score_teacher_distribution(self, messages: list, response_msg: dict, tools, response_len: int, metadata: dict[str, Any] | None = None) -> tuple[list[float], list[list[float]], list[list[int]]]:
        if not self._teacher_logprob_enabled or response_len <= 0:
            return [], [], []
        try:
            metadata = metadata or {}
            extra_prompt = await self._generate_teacher_extra_prompt(messages, response_msg, metadata)
            if extra_prompt:
                metadata["teacher_extra_prompt"] = extra_prompt
                metadata["teacher_extra_prompt_model"] = self._teacher_extra_prompt_model
            prompt_ids, full_ids = self._build_teacher_scoring_ids(messages, response_msg, tools, metadata, extra_prompt)
            if len(full_ids) <= len(prompt_ids):
                return [], [], []
            payload = {
                "input_ids": full_ids,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 0,
                    "skip_special_tokens": False,
                    "no_stop_trim": True,
                    "spaces_between_special_tokens": False,
                },
                "return_logprob": True,
                "logprob_start_len": max(0, len(prompt_ids) - 1),
                "top_logprobs_num": max(0, self._teacher_topk),
            }
            url = f"http://{self.args.sglang_router_ip}:{self.args.sglang_router_port}/generate"
            async with httpx.AsyncClient(timeout=None) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
            meta = data.get("meta_info", {}) if isinstance(data, dict) else {}
            pairs = meta.get("input_token_logprobs", [])
            vals = [float(p[0]) for p in pairs if isinstance(p, (list, tuple)) and len(p) >= 2 and p[0] is not None]
            if len(vals) >= response_len:
                vals = vals[-response_len:]
            elif vals:
                logger.warning("[CoreCoder] teacher logprob length short: got=%d want=%d", len(vals), response_len)
                vals = vals + [vals[-1]] * (response_len - len(vals))
            topk_log_probs, topk_indices = _extract_topk_logprobs_from_meta(meta, response_len)
            return vals, topk_log_probs, topk_indices
        except Exception as e:
            logger.warning("[CoreCoder] teacher distribution scoring failed: %s", e)
        return [], [], []

    # ---------------------------------------------------- request handling
    async def _handle_request(
        self,
        body: dict[str, Any],
        session_id: str,
        turn_type: str,
        session_done: bool,
    ) -> dict[str, Any]:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages must be a non-empty list")

        request_metadata = self._metadata_from_body(body)
        if request_metadata:
            self._session_metadata.setdefault(session_id, {}).update(request_metadata)

        tools = body.get("tools")

        forward_body = {k: v for k, v in body.items() if k not in _NON_STANDARD_BODY_KEYS}
        forward_body["stream"] = False
        forward_body.pop("stream_options", None)
        forward_body["logprobs"] = True
        forward_body["top_logprobs"] = 1
        if "model" not in forward_body:
            forward_body["model"] = self.served_model_name

        async with httpx.AsyncClient(timeout=None) as client:
            sglang_resp = await client.post(self.sglang_chat_url, json=forward_body)
            if sglang_resp.status_code != 200:
                logger.error("[CoreCoder] SGLang returned %d: %s", sglang_resp.status_code, sglang_resp.text[:1000])
                sglang_resp.raise_for_status()
            try:
                output = sglang_resp.json()
            except Exception:
                logger.error("[CoreCoder] SGLang non-JSON body: %s", sglang_resp.text[:1000])
                raise

        choice = output.get("choices", [{}])[0]
        assistant_msg = choice.get("message", {})
        tool_calls = assistant_msg.get("tool_calls") or []

        content = assistant_msg.get("content") or ""
        reasoning = assistant_msg.get("reasoning_content") or ""
        logger.info(
            f"{_YELLOW}[CoreCoder] [{turn_type}] session={session_id} "
            f"prompt_msgs={len(messages)}{_RESET}"
        )
        logger.info(
            f"{_RED}[CoreCoder] [{turn_type}] session={session_id} "
            f"thinking={len(reasoning)} chars, response:\n{content}{_RESET}"
        )
        if tool_calls:
            logger.info("[CoreCoder] tool_calls: %s", json.dumps(tool_calls, ensure_ascii=False)[:500])

        if turn_type == "main":
            if session_id in self._pending_records and messages:
                self._flush_pending_record(session_id, messages[-1])

            response_msg = dict(assistant_msg)
            if response_msg.get("content") is None:
                response_msg["content"] = ""

            norm_msgs = _normalize_messages_for_template(messages)
            norm_resp = _normalize_messages_for_template([response_msg])[0]
            full_norm = norm_msgs + [norm_resp]

            prompt_text = self.tokenizer.apply_chat_template(
                norm_msgs, tools=tools, tokenize=False, add_generation_prompt=True,
            )
            full_text = self.tokenizer.apply_chat_template(
                full_norm, tools=tools, tokenize=False, add_generation_prompt=False,
            )

            if full_text.startswith(prompt_text):
                response_text = full_text[len(prompt_text):]
            else:
                logger.warning("[CoreCoder] prompt_text is not a prefix of full_text, using full_text as response")
                response_text = full_text

            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            response_ids = self.tokenizer(response_text, add_special_tokens=False)["input_ids"]

            if not response_ids and not response_text.strip():
                logger.info("[CoreCoder] MAIN session=%s → empty response, skipping", session_id)
                output["session_id"] = session_id
                return {"response": output}

            response_logprobs = _extract_logprobs_from_chat_response(choice)
            if len(response_logprobs) > len(response_ids):
                response_logprobs = response_logprobs[: len(response_ids)]
            elif len(response_logprobs) < len(response_ids):
                response_logprobs = response_logprobs + [0.0] * (len(response_ids) - len(response_logprobs))

            self._turn_counts[session_id] = self._turn_counts.get(session_id, 0) + 1
            turn_num = self._turn_counts[session_id]

            metadata = dict(self._session_metadata.get(session_id, {}))
            teacher_logprobs, teacher_topk_logprobs, teacher_topk_indices = await self._score_teacher_distribution(messages, response_msg, tools, len(response_ids), metadata)
            if len(teacher_logprobs) != len(response_ids):
                teacher_logprobs = response_logprobs[:]
                logger.warning("[CoreCoder] falling back to rollout logprobs as teacher targets for session=%s turn=%d", session_id, turn_num)
            if (
                len(teacher_topk_logprobs) != len(response_ids)
                or len(teacher_topk_indices) != len(response_ids)
                or any(not row for row in teacher_topk_logprobs)
                or any(not row for row in teacher_topk_indices)
            ):
                teacher_topk_logprobs = [[lp] for lp in teacher_logprobs]
                teacher_topk_indices = [[tok] for tok in response_ids]

            turn_data = {
                "turn": turn_num,
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "response_logprobs": response_logprobs,
                "teacher_logprobs": teacher_logprobs,
                "teacher_topk_logprobs": teacher_topk_logprobs,
                "teacher_topk_indices": teacher_topk_indices,
                "prompt_text": prompt_text,
                "response_text": response_text,
                "metadata": metadata,
            }

            logger.info(
                "[CoreCoder] MAIN session=%s turn=%d prompt_tokens=%d response_tokens=%d",
                session_id, turn_num, len(prompt_ids), len(response_ids),
            )
            self._buffer_record(session_id, turn_num, messages, prompt_text, response_text, tool_calls)
            self._pending_turn_data.setdefault(session_id, {})[turn_num] = turn_data
            self._maybe_submit_ready_samples(session_id)
        else:
            logger.info("[CoreCoder] SIDE session=%s → skipped (no training data)", session_id)

        if session_done:
            self._flush_pending_record(session_id, None)
            self._maybe_submit_ready_samples(session_id, force_no_prm=True)
            eff = self._session_effective.pop(session_id, 0)
            self._turn_counts.pop(session_id, None)
            logger.info("[CoreCoder] session=%s done → cleaned up (effective_samples=%d)", session_id, eff)

        output["session_id"] = session_id
        return {"response": output}

    # ------------------------------------------------- sample submission
    def _maybe_submit_ready_samples(self, session_id: str, force_no_prm: bool = False):
        """Submit turns whose PRM is done (or PRM not needed).

        force_no_prm: also submit turns that have no PRM task yet (used at
        session end for the last turn which will never get a next_state).
        """
        prm_tasks = self._prm_tasks.get(session_id, {})
        pending = self._pending_turn_data.get(session_id, {})
        for turn_num in sorted(list(pending.keys())):
            task = prm_tasks.get(turn_num)
            if not self._prm_enabled:
                pass  # no PRM → submit immediately
            elif task is not None and not task.done():
                continue  # PRM still running
            elif task is None and not force_no_prm:
                continue  # waiting for next_state to fire PRM
            turn_data = pending.pop(turn_num)
            prm_result = None
            if task is not None and task.done():
                try:
                    prm_result = task.result()
                except Exception:
                    pass
                prm_tasks.pop(turn_num, None)
            self._safe_create_task(
                self._submit_turn_sample(turn_data, session_id, prm_result)
            )

    def _append_metrics_record(self, session_id: str, turn_data: dict[str, Any],
                               prm_result: dict | None, score: float, exclude: bool,
                               sample: Sample):
        if not self._metrics_file:
            return
        metadata = dict(self._session_metadata.get(session_id, {}))
        metadata.update(turn_data.get("metadata") or {})
        row: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": session_id,
            "turn": turn_data.get("turn"),
            "sample_index": sample.index,
            "group_index": sample.group_index,
            "scenario_id": metadata.get("scenario_id"),
            "student_mode": metadata.get("student_mode"),
            "student_model": metadata.get("student_model"),
            "feeder_id": metadata.get("feeder_id"),
            "rl_method": metadata.get("rl_method"),
            "prompt_source": metadata.get("prompt_source"),
            "extra_prompt_model": metadata.get("extra_prompt_model"),
            "reward": score,
            "loss_mask": 0 if exclude else 1,
            "has_next_state": bool(turn_data.get("has_next_state", False)),
            "prompt_tokens": len(turn_data.get("prompt_ids", [])),
            "response_tokens": len(turn_data.get("response_ids", [])),
            "response_chars": len(turn_data.get("response_text", "")),
            "student_logprob_mean": (sum(turn_data.get("response_logprobs", []) or [0.0]) / max(1, len(turn_data.get("response_logprobs", [])))),
            "teacher_logprob_mean": (sum(turn_data.get("teacher_logprobs", []) or [0.0]) / max(1, len(turn_data.get("teacher_logprobs", [])))),
            "prm_votes": (prm_result or {}).get("votes"),
            "prm_representative_eval": bool((prm_result or {}).get("representative_eval")),
            "requires_tool": metadata.get("requires_tool"),
            "allowed_tools": metadata.get("allowed_tools"),
            "tool_call_count": metadata.get("tool_call_count"),
            "tool_success": metadata.get("tool_success"),
            "tool_name": metadata.get("tool_name"),
            "final_correct": metadata.get("final_correct"),
            "reference_answer_present": bool(metadata.get("reference_answer")),
            "question_present": bool(metadata.get("question")),
            "checker": metadata.get("checker"),
            "teacher_extra_prompt_present": bool(metadata.get("teacher_extra_prompt")),
            "teacher_extra_prompt_model": metadata.get("teacher_extra_prompt_model"),
        }
        try:
            with open(self._metrics_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning("[CoreCoder] failed to append metrics record: %s", e)

    async def _submit_turn_sample(self, turn_data: dict[str, Any], session_id: str,
                                  prm_result: dict | None):
        prompt_ids = turn_data["prompt_ids"]
        response_ids = turn_data["response_ids"]

        has_next_state = turn_data.get("has_next_state", False)
        if prm_result:
            score = prm_result["score"]
        elif "deterministic_score" in turn_data:
            score = float(turn_data["deterministic_score"])
        else:
            score = 0.0

        with self._eval_scores_lock:
            self._eval_scores.append(score)

        exclude = not has_next_state or score == 0.0
        # Guarantee at least one sample per session contributes to training.
        # If this session has produced zero effective samples so far and this
        # turn has a next_state (i.e. it was PRM-evaluated, just scored 0),
        # promote it to effective.
        if exclude and has_next_state and self._session_effective.get(session_id, 0) == 0:
            exclude = False
            logger.info("[CoreCoder] promoting session=%s turn with score=0 → loss_mask=1 (at-least-one guarantee)", session_id)

        sample = Sample()
        sample.prompt = turn_data["prompt_text"]
        sample.response = turn_data["response_text"]
        sample.tokens = prompt_ids + response_ids
        sample.response_length = len(response_ids)
        sample.loss_mask = [0] * len(response_ids) if exclude else [1] * len(response_ids)
        sample.rollout_log_probs = turn_data["response_logprobs"]
        sample.teacher_log_probs = turn_data.get("teacher_logprobs") or turn_data["response_logprobs"]
        sample.teacher_topk_log_probs = turn_data.get("teacher_topk_logprobs")
        sample.teacher_topk_indices = turn_data.get("teacher_topk_indices")
        sample.status = Sample.Status.COMPLETED
        sample.index = next(self._index_counter)
        sample.group_index = next(self._group_counter)
        sample.reward = {"score": score}

        if not exclude:
            self._session_effective[session_id] = self._session_effective.get(session_id, 0) + 1

        logger.info(
            "[CoreCoder] submitted sample session=%s index=%d score=%.1f exclude=%s prompt_len=%d response_len=%d",
            session_id, sample.index, score, exclude, len(prompt_ids), len(response_ids),
        )
        self._append_metrics_record(session_id, turn_data, prm_result, score, exclude, sample)
        await asyncio.to_thread(self.output_queue.put, (sample.group_index, [sample]))

    def _safe_create_task(self, coro):
        task = asyncio.create_task(coro)
        task.add_done_callback(self._task_done_cb)

    @staticmethod
    def _task_done_cb(task: asyncio.Task):
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("[CoreCoder] background task failed: %s", exc, exc_info=exc)

    # ----------------------------------------------------------- streaming
    async def _stream_response(self, result: dict[str, Any]):
        payload = result["response"]
        choice = payload.get("choices", [{}])[0]
        message = choice.get("message", {})
        delta = {"role": "assistant", "content": message.get("content", "") or ""}
        if message.get("tool_calls"):
            delta["tool_calls"] = message["tool_calls"]
        chunk_base = {
            "id": payload.get("id", ""),
            "object": "chat.completion.chunk",
            "created": payload.get("created", int(time.time())),
            "model": payload.get("model", ""),
            "session_id": payload.get("session_id", ""),
        }
        first = {**chunk_base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
        final = {**chunk_base, "choices": [{"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason", "stop")}]}
        yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    # ------------------------------------------------------------- lifecycle
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="info")
        self._server = uvicorn.Server(config=config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        self._readiness_thread = threading.Thread(target=self._wait_for_sglang_ready, daemon=True)
        self._readiness_thread.start()

    def _wait_for_sglang_ready(self):
        while True:
            try:
                r = httpx.get(self.sglang_health_url, timeout=5)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(3)
        logger.info("[CoreCoder] policy server ready")

        if self._prm_enabled and self._prm_url:
            prm_health = self._prm_url.rsplit("/", 1)[0] + "/health"
            while True:
                try:
                    r = httpx.get(prm_health, timeout=5)
                    if r.status_code == 200:
                        break
                except Exception:
                    pass
                time.sleep(3)
            logger.info("[CoreCoder] PRM server ready")

        time.sleep(8)
        prm_line = ""
        if self._prm_enabled:
            prm_line = f"\n  PRM enabled: {self._prm_url} (m={self._prm_m})"
        banner = (
            f"\n{'=' * 70}\n"
            f"  [CoreCoder] your model is fired up and ready to roll\n"
            f"  proxy {self.host}:{self.port} -> SGLang "
            f"{self.args.sglang_router_ip}:{self.args.sglang_router_port}"
            f"{prm_line}\n"
            f"{'=' * 70}\n"
        )
        logger.info(f"{_GREEN}{banner}{_RESET}")

    def stop(self):
        if self._server is not None:
            self._server.should_exit = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
