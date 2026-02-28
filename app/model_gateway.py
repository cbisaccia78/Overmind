"""Model inference gateway.

Generates tool-call decisions and supervisor advisory directives using an
agent's configured model, then persists interaction telemetry for auditability.
"""

from __future__ import annotations

import json
import os
import re
import time
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import urlsplit, urlunsplit
from typing import Any, Callable

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - dependency optional at import time
    OpenAI = None  # type: ignore[assignment]

from .repository import Repository


class ModelGateway:
    """Model inference facade that records request/response telemetry.

    The gateway returns normalized tool-call decisions (plus optional
    supervisor advisory payloads) and persists corresponding `model_calls`
    rows for auditing.
    """

    def __init__(
        self,
        repo: Repository,
        openai_tools_provider: (
            Callable[
                [list[str]],
                list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, str]],
            ]
            | None
        ) = None,
    ):
        """Create a model gateway.

        Args:
            repo: Repository used to persist model call audit records.
            openai_tools_provider: Optional callable that returns OpenAI-compatible
                tool definitions for a list of tool names.
        """
        self.repo = repo
        self.openai_tools_provider = openai_tools_provider

    def infer(
        self,
        *,
        task: str,
        agent: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Infer a single tool call from a task for an agent model.

        Args:
            task: Task text to plan.
            agent: Agent row/configuration.
            context: Run/step planning context.

        Returns:
            A normalized payload with `ok`, `tool_name`, `args`, usage, and errors.
        """
        # Backward-compatible fallback: schema/UI require `model`, but older DB rows
        # may still be missing it.
        model = str(agent.get("model") or "stub-v1")
        allowed_tools = [str(name) for name in (agent.get("tools_allowed") or [])]
        for internal_tool in ("ask_user", "final_answer"):
            if internal_tool not in allowed_tools:
                allowed_tools.append(internal_tool)
        request_json = {
            "task": task,
            "agent_id": agent.get("id"),
            "model": model,
            "context": context,
        }
        run_id = str(context.get("run_id") or "").strip()
        self._emit_run_event(
            run_id=run_id,
            event_type="model.infer.started",
            payload={
                "run_id": run_id,
                "kind": "action",
                "model": model,
            },
        )

        start = time.monotonic()
        try:
            model_name = str(model or "").strip().lower()
            if self._should_use_deepseek(model=model, allowed_tools=allowed_tools):
                response_json = self._infer_with_deepseek(
                    task=task,
                    model=model,
                    allowed_tools=allowed_tools,
                    context=context,
                    include_meta=True,
                )
            elif model_name.startswith("deepseek"):
                response_json = self._infer_with_model(task=task, model=model)
            elif self._should_use_openai(allowed_tools):
                response_json = self._infer_with_openai(
                    task=task,
                    model=model,
                    allowed_tools=allowed_tools,
                    context=context,
                    include_meta=True,
                )
            else:
                response_json = self._infer_with_model(task=task, model=model)
            usage_json = self._estimate_usage(task=task, response=response_json)
            result: dict[str, Any] = {
                "ok": True,
                "model": model,
                "tool_name": response_json["tool_name"],
                "args": response_json["args"],
                "response": response_json,
                "usage": usage_json,
            }
            self._emit_model_output_events(
                run_id=run_id,
                model=model,
                stream_kind="action",
                response=response_json,
            )
            error = None
        except Exception as exc:  # noqa: BLE001
            response_json = {}
            usage_json = self._estimate_usage(task=task, response=response_json)
            result = {
                "ok": False,
                "model": model,
                "error": {
                    "code": "model_error",
                    "message": str(exc),
                },
                "response": response_json,
                "usage": usage_json,
            }
            error = str(exc)
            self._emit_run_event(
                run_id=run_id,
                event_type="model.infer.failed",
                payload={
                    "run_id": run_id,
                    "kind": "action",
                    "model": model,
                    "error": error,
                },
            )

        latency_ms = int((time.monotonic() - start) * 1000)
        if self.repo is not None:
            self.repo.create_model_call(
                run_id=context.get("run_id"),
                agent_id=agent.get("id"),
                model=model,
                request_json=request_json,
                response_json=result.get("response", {}),
                usage_json=result.get("usage", {}),
                error=error,
                latency_ms=latency_ms,
            )
        result["latency_ms"] = latency_ms
        return result

    def advise_supervisor(
        self,
        *,
        task: str,
        agent: dict[str, Any],
        context: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Return structured advisory planning hints for supervisor usage.

        The returned advice is always sanitized and bounded. The deterministic
        supervisor remains authoritative for budgets and transitions.
        """
        model = str(agent.get("model") or "stub-v1")
        request_json = {
            "kind": "supervisor_advice",
            "task": task,
            "agent_id": agent.get("id"),
            "model": model,
            "context": context,
            "state": state,
        }
        run_id = str(context.get("run_id") or "").strip()
        self._emit_run_event(
            run_id=run_id,
            event_type="model.infer.started",
            payload={
                "run_id": run_id,
                "kind": "supervisor",
                "model": model,
            },
        )

        start = time.monotonic()
        try:
            model_name = str(model or "").strip().lower()
            if model_name.startswith("deepseek") and os.getenv("DEEPSEEK_API_KEY"):
                raw = self._advise_with_openai_compatible(
                    model=model,
                    provider="deepseek",
                    api_key=os.getenv("DEEPSEEK_API_KEY") or "",
                    endpoint=os.getenv(
                        "OVERMIND_DEEPSEEK_CHAT_COMPLETIONS_URL",
                        "https://api.deepseek.com/v1/chat/completions",
                    ),
                    task=task,
                    state=state,
                )
            elif os.getenv("OPENAI_API_KEY"):
                raw = self._advise_with_openai_responses(
                    model=model,
                    api_key=os.getenv("OPENAI_API_KEY") or "",
                    endpoint=os.getenv(
                        "OVERMIND_OPENAI_RESPONSES_URL",
                        "https://api.openai.com/v1/responses",
                    ),
                    task=task,
                    state=state,
                )
            else:
                raw = self._advise_with_model(task=task, state=state)

            advice = self._sanitize_supervisor_advice(raw)
            response_json = {"advice": advice}
            usage_json = self._estimate_usage(task=task, response=response_json)
            result: dict[str, Any] = {
                "ok": True,
                "model": model,
                "advice": advice,
                "response": response_json,
                "usage": usage_json,
            }
            self._emit_model_output_events(
                run_id=run_id,
                model=model,
                stream_kind="supervisor",
                advice=advice,
                response=response_json,
            )
            error = None
        except Exception as exc:  # noqa: BLE001
            fallback = self._sanitize_supervisor_advice(
                self._advise_with_model(task=task, state=state)
            )
            response_json = {"advice": fallback}
            usage_json = self._estimate_usage(task=task, response=response_json)
            result = {
                "ok": False,
                "model": model,
                "advice": fallback,
                "error": {"code": "model_error", "message": str(exc)},
                "response": response_json,
                "usage": usage_json,
            }
            error = str(exc)
            self._emit_run_event(
                run_id=run_id,
                event_type="model.infer.failed",
                payload={
                    "run_id": run_id,
                    "kind": "supervisor",
                    "model": model,
                    "error": error,
                },
            )

        latency_ms = int((time.monotonic() - start) * 1000)
        if self.repo is not None:
            self.repo.create_model_call(
                run_id=context.get("run_id"),
                agent_id=agent.get("id"),
                model=model,
                request_json=request_json,
                response_json=result.get("response", {}),
                usage_json=result.get("usage", {}),
                error=error,
                latency_ms=latency_ms,
            )
        result["latency_ms"] = latency_ms
        return result

    def summarize_context(
        self,
        *,
        task: str,
        agent: dict[str, Any],
        context: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a compact rolling context summary for long-running loops."""
        model = str(agent.get("model") or "stub-v1")
        request_json = {
            "kind": "context_summary",
            "task": task,
            "agent_id": agent.get("id"),
            "model": model,
            "context": context,
            "state": state,
        }
        run_id = str(context.get("run_id") or "").strip()
        self._emit_run_event(
            run_id=run_id,
            event_type="model.infer.started",
            payload={
                "run_id": run_id,
                "kind": "context_summary",
                "model": model,
            },
        )

        start = time.monotonic()
        try:
            model_name = str(model or "").strip().lower()
            if model_name.startswith("deepseek") and os.getenv("DEEPSEEK_API_KEY"):
                raw = self._summarize_context_with_openai_compatible(
                    model=model,
                    provider="deepseek",
                    api_key=os.getenv("DEEPSEEK_API_KEY") or "",
                    endpoint=os.getenv(
                        "OVERMIND_DEEPSEEK_CHAT_COMPLETIONS_URL",
                        "https://api.deepseek.com/v1/chat/completions",
                    ),
                    task=task,
                    state=state,
                )
            elif os.getenv("OPENAI_API_KEY"):
                raw = self._summarize_context_with_openai_responses(
                    model=model,
                    api_key=os.getenv("OPENAI_API_KEY") or "",
                    endpoint=os.getenv(
                        "OVERMIND_OPENAI_RESPONSES_URL",
                        "https://api.openai.com/v1/responses",
                    ),
                    task=task,
                    state=state,
                )
            else:
                raw = self._summarize_context_with_model(task=task, state=state)

            summary = self._sanitize_context_summary(raw)
            response_json = {"summary": summary}
            usage_json = self._estimate_usage(task=task, response=response_json)
            result: dict[str, Any] = {
                "ok": True,
                "model": model,
                "summary": summary,
                "response": response_json,
                "usage": usage_json,
            }
            self._emit_model_output_events(
                run_id=run_id,
                model=model,
                stream_kind="context_summary",
                response=response_json,
            )
            error = None
        except Exception as exc:  # noqa: BLE001
            fallback = self._sanitize_context_summary(
                self._summarize_context_with_model(task=task, state=state)
            )
            response_json = {"summary": fallback}
            usage_json = self._estimate_usage(task=task, response=response_json)
            result = {
                "ok": False,
                "model": model,
                "summary": fallback,
                "error": {"code": "model_error", "message": str(exc)},
                "response": response_json,
                "usage": usage_json,
            }
            error = str(exc)
            self._emit_run_event(
                run_id=run_id,
                event_type="model.infer.failed",
                payload={
                    "run_id": run_id,
                    "kind": "context_summary",
                    "model": model,
                    "error": error,
                },
            )

        latency_ms = int((time.monotonic() - start) * 1000)
        if self.repo is not None:
            self.repo.create_model_call(
                run_id=context.get("run_id"),
                agent_id=agent.get("id"),
                model=model,
                request_json=request_json,
                response_json=result.get("response", {}),
                usage_json=result.get("usage", {}),
                error=error,
                latency_ms=latency_ms,
            )
        result["latency_ms"] = latency_ms
        return result

    def _emit_run_event(
        self,
        *,
        run_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        """Persist a run event when run context and repository are available."""
        if self.repo is None or not run_id:
            return
        self.repo.create_event(run_id, event_type, payload)

    def _emit_model_output_events(
        self,
        *,
        run_id: str,
        model: str,
        stream_kind: str,
        response: dict[str, Any] | None = None,
        advice: dict[str, Any] | None = None,
    ) -> None:
        """Emit model output chunks as run events for live UI streaming."""
        if self.repo is None or not run_id:
            return

        segments = self._collect_model_output_segments(
            response=response,
            advice=advice,
        )
        if not segments:
            return

        stream_base = f"{stream_kind}:{time.time_ns()}"
        for seg_idx, segment in enumerate(segments):
            channel = str(segment.get("channel") or "output")
            text = str(segment.get("text") or "")
            if not text:
                continue
            stream_id = f"{stream_base}:{seg_idx}"
            self._emit_run_event(
                run_id=run_id,
                event_type="model.output.started",
                payload={
                    "run_id": run_id,
                    "stream_id": stream_id,
                    "kind": stream_kind,
                    "model": model,
                    "channel": channel,
                },
            )
            for chunk_idx, chunk in enumerate(self._chunk_text(text, chunk_size=56)):
                self._emit_run_event(
                    run_id=run_id,
                    event_type="model.output.delta",
                    payload={
                        "run_id": run_id,
                        "stream_id": stream_id,
                        "kind": stream_kind,
                        "model": model,
                        "channel": channel,
                        "delta": chunk,
                        "idx": chunk_idx,
                    },
                )
            self._emit_run_event(
                run_id=run_id,
                event_type="model.output.completed",
                payload={
                    "run_id": run_id,
                    "stream_id": stream_id,
                    "kind": stream_kind,
                    "model": model,
                    "channel": channel,
                    "total_chars": len(text),
                },
            )

    @staticmethod
    def _chunk_text(text: str, chunk_size: int) -> list[str]:
        """Split text into fixed-size chunks for streaming events."""
        if chunk_size <= 0:
            return [text]
        if not text:
            return []
        return [text[idx : idx + chunk_size] for idx in range(0, len(text), chunk_size)]

    def _collect_model_output_segments(
        self,
        *,
        response: dict[str, Any] | None,
        advice: dict[str, Any] | None,
    ) -> list[dict[str, str]]:
        """Build streamable text segments from model response payloads."""
        segments: list[dict[str, str]] = []
        payload = response if isinstance(response, dict) else {}
        reasoning_content = self._streamable_text(payload.get("reasoning_content"))
        assistant_content = self._streamable_text(payload.get("assistant_content"))
        if reasoning_content:
            segments.append({"channel": "reasoning", "text": reasoning_content})
        if assistant_content:
            segments.append({"channel": "assistant", "text": assistant_content})

        if isinstance(advice, dict) and advice:
            advice_text = self._streamable_text(
                json.dumps(advice, indent=2, ensure_ascii=True)
            )
            if advice_text:
                segments.append({"channel": "advice", "text": advice_text})

        summary_payload = payload.get("summary")
        if isinstance(summary_payload, dict) and summary_payload:
            summary_text = self._streamable_text(
                json.dumps(summary_payload, indent=2, ensure_ascii=True)
            )
            if summary_text:
                segments.append({"channel": "summary", "text": summary_text})

        if segments:
            return segments

        tool_name = str(payload.get("tool_name") or "").strip()
        args_payload = payload.get("args")
        if not tool_name:
            return segments

        args_text = ""
        if isinstance(args_payload, dict) and args_payload:
            args_text = json.dumps(args_payload, ensure_ascii=True, sort_keys=True)
            if len(args_text) > 400:
                args_text = args_text[:400].rstrip() + "..."
        summary = f"Selected tool: {tool_name}"
        if args_text:
            summary = f"{summary} {args_text}"
        segments.append({"channel": "decision", "text": summary})
        return segments

    @staticmethod
    def _streamable_text(value: Any) -> str:
        """Normalize arbitrary payload values into compact stream text."""
        text = str(value or "").strip()
        if not text:
            return ""
        if len(text) <= 2000:
            return text
        return text[:2000].rstrip() + "..."

    def _should_use_openai(self, allowed_tools: list[str]) -> bool:
        """Return whether OpenAI tool-calling should be used.

        Args:
            allowed_tools: Tool names allowed for the current agent.

        Returns:
            True when an OpenAI API key is configured and at least one tool is
            available to expose.
        """
        return bool(
            os.getenv("OPENAI_API_KEY")
            and allowed_tools
            and self.openai_tools_provider is not None
        )

    def _advise_with_openai_responses(
        self,
        *,
        model: str,
        api_key: str,
        endpoint: str,
        task: str,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Request structured supervisor advice via OpenAI Responses API."""
        payload = {
            "model": model,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": self._supervisor_system_prompt(),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": self._supervisor_user_prompt(task=task, state=state),
                        }
                    ],
                },
            ],
            "temperature": 0,
        }
        payload, used_reasoning = self._apply_reasoning_request_tuning(
            payload=payload,
            model=model,
            provider="openai",
        )
        timeout_s = int(os.getenv("OVERMIND_OPENAI_TIMEOUT_S", "10"))
        try:
            data = self._request_openai_responses(
                payload=payload,
                api_key=api_key,
                endpoint=endpoint,
                timeout_s=timeout_s,
            )
        except RuntimeError:
            if not used_reasoning:
                raise
            data = self._request_openai_responses(
                payload=self._strip_reasoning_request_tuning(payload),
                api_key=api_key,
                endpoint=endpoint,
                timeout_s=timeout_s,
            )
        return self._parse_supervisor_json_response(data)

    def _advise_with_openai_compatible(
        self,
        *,
        model: str,
        provider: str,
        api_key: str,
        endpoint: str,
        task: str,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Request structured supervisor advice via OpenAI-compatible chat API."""
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self._supervisor_system_prompt()},
                {
                    "role": "user",
                    "content": self._supervisor_user_prompt(task=task, state=state),
                },
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        payload, used_reasoning = self._apply_reasoning_request_tuning(
            payload=payload,
            model=model,
            provider=provider,
        )
        timeout_s = int(os.getenv("OVERMIND_OPENAI_TIMEOUT_S", "10"))
        try:
            data = self._request_openai_api(
                payload=payload,
                api_key=api_key,
                endpoint=endpoint,
                timeout_s=timeout_s,
            )
        except RuntimeError:
            if not used_reasoning:
                raise
            data = self._request_openai_api(
                payload=self._strip_reasoning_request_tuning(payload),
                api_key=api_key,
                endpoint=endpoint,
                timeout_s=timeout_s,
            )
        return self._parse_supervisor_json_response(data)

    def _summarize_context_with_openai_responses(
        self,
        *,
        model: str,
        api_key: str,
        endpoint: str,
        task: str,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Request structured context summary via OpenAI Responses API."""
        payload = {
            "model": model,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": self._context_summary_system_prompt(),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": self._context_summary_user_prompt(
                                task=task,
                                state=state,
                            ),
                        }
                    ],
                },
            ],
            "temperature": 0,
        }
        payload, used_reasoning = self._apply_reasoning_request_tuning(
            payload=payload,
            model=model,
            provider="openai",
        )
        timeout_s = int(os.getenv("OVERMIND_OPENAI_TIMEOUT_S", "10"))
        try:
            data = self._request_openai_responses(
                payload=payload,
                api_key=api_key,
                endpoint=endpoint,
                timeout_s=timeout_s,
            )
        except RuntimeError:
            if not used_reasoning:
                raise
            data = self._request_openai_responses(
                payload=self._strip_reasoning_request_tuning(payload),
                api_key=api_key,
                endpoint=endpoint,
                timeout_s=timeout_s,
            )
        return self._parse_supervisor_json_response(data)

    def _summarize_context_with_openai_compatible(
        self,
        *,
        model: str,
        provider: str,
        api_key: str,
        endpoint: str,
        task: str,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Request structured context summary via OpenAI-compatible chat API."""
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self._context_summary_system_prompt()},
                {
                    "role": "user",
                    "content": self._context_summary_user_prompt(task=task, state=state),
                },
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        payload, used_reasoning = self._apply_reasoning_request_tuning(
            payload=payload,
            model=model,
            provider=provider,
        )
        timeout_s = int(os.getenv("OVERMIND_OPENAI_TIMEOUT_S", "10"))
        try:
            data = self._request_openai_api(
                payload=payload,
                api_key=api_key,
                endpoint=endpoint,
                timeout_s=timeout_s,
            )
        except RuntimeError:
            if not used_reasoning:
                raise
            data = self._request_openai_api(
                payload=self._strip_reasoning_request_tuning(payload),
                api_key=api_key,
                endpoint=endpoint,
                timeout_s=timeout_s,
            )
        return self._parse_supervisor_json_response(data)

    @staticmethod
    def _supervisor_system_prompt() -> str:
        return (
            "You are a supervisor planning assistant for a browser agent.\n"
            "Return only one JSON object with keys:\n"
            "mode, phase, rationale, micro_plan, success_criteria.\n"
            "Rules:\n"
            "- mode must be one of exploration, execution, recovery.\n"
            "- micro_plan must be 1-5 short actionable steps.\n"
            "- success_criteria must be 1-4 measurable checks.\n"
            "- Do not include tool calls, code, or prose outside JSON."
        )

    @staticmethod
    def _supervisor_user_prompt(task: str, state: dict[str, Any]) -> str:
        compact_state = json.dumps(state, separators=(",", ":"), ensure_ascii=True)
        if len(compact_state) > 5000:
            compact_state = compact_state[:5000] + "..."
        return (
            "Goal:\n"
            f"{task.strip()}\n\n"
            "Current state snapshot (JSON):\n"
            f"{compact_state}\n\n"
            "Return the next advisory directive JSON."
        )

    @staticmethod
    def _context_summary_system_prompt() -> str:
        return (
            "You summarize long-running agent context into compact structured memory.\n"
            "Return only one JSON object with keys:\n"
            "objective_status, progress_summary, completed_milestones, open_issues, "
            "attempted_paths, constraints, next_focus.\n"
            "Rules:\n"
            "- Keep statements factual and concise.\n"
            "- Preserve concrete identifiers (tool names, URLs, file paths, error codes) when present.\n"
            "- Do not invent facts; if unknown, say unknown or leave list empty.\n"
            "- completed_milestones/open_issues/attempted_paths/constraints: 0-6 short items each.\n"
            "- next_focus: 1-4 actionable items.\n"
            "- No markdown, no prose outside JSON."
        )

    @staticmethod
    def _context_summary_user_prompt(task: str, state: dict[str, Any]) -> str:
        compact_state = json.dumps(state, separators=(",", ":"), ensure_ascii=True)
        if len(compact_state) > 7000:
            compact_state = compact_state[:7000] + "..."
        return (
            "Goal:\n"
            f"{task.strip()}\n\n"
            "Context state to summarize (JSON):\n"
            f"{compact_state}\n\n"
            "Return compact rolling context summary JSON."
        )

    def _parse_supervisor_json_response(self, data: dict[str, Any]) -> dict[str, Any]:
        """Parse JSON content from chat-completions or responses payloads."""
        content = self._extract_response_text(data)
        if not content:
            raise RuntimeError("openai response missing content")
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        candidate = self._extract_first_json_object(content)
        if not candidate:
            raise RuntimeError("openai response missing valid json object")
        parsed = json.loads(candidate)
        if not isinstance(parsed, dict):
            raise RuntimeError("openai response json must be object")
        return parsed

    @staticmethod
    def _extract_response_text(data: dict[str, Any]) -> str:
        """Extract plain text from chat-completions or responses payloads."""
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            message = None
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                        continue
                    if isinstance(item, dict):
                        text = item.get("text")
                        if isinstance(text, str):
                            parts.append(text)
                text = "\n".join(parts).strip()
                if text:
                    return text

        top_level_text = data.get("output_text")
        if isinstance(top_level_text, str) and top_level_text.strip():
            return top_level_text.strip()

        output = data.get("output")
        if not isinstance(output, list):
            return ""

        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").strip().lower()
            if item_type == "message":
                content = item.get("content")
                if isinstance(content, str):
                    if content.strip():
                        parts.append(content.strip())
                    continue
                if isinstance(content, list):
                    for chunk in content:
                        if isinstance(chunk, str):
                            if chunk.strip():
                                parts.append(chunk.strip())
                            continue
                        if not isinstance(chunk, dict):
                            continue
                        chunk_type = str(chunk.get("type") or "").strip().lower()
                        if chunk_type.startswith("reasoning"):
                            continue
                        text = chunk.get("text")
                        if isinstance(text, str) and text.strip():
                            parts.append(text.strip())
                continue
            if item_type == "output_text":
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()

    @staticmethod
    def _extract_first_json_object(text: str) -> str:
        """Return first top-level JSON object substring if present."""
        raw = str(text or "")
        start = raw.find("{")
        if start < 0:
            return ""
        depth = 0
        in_string = False
        escaped = False
        for idx, ch in enumerate(raw[start:], start=start):
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
                continue
            if ch == "}":
                depth -= 1
                if depth == 0:
                    return raw[start : idx + 1]
        return ""

    def _advise_with_model(self, *, task: str, state: dict[str, Any]) -> dict[str, Any]:
        """Deterministic fallback advisory planner for local/dev operation."""
        directive = state.get("directive")
        if not isinstance(directive, dict):
            directive = {}
        mode = str(directive.get("mode") or "execution").strip().lower()
        phase = str(directive.get("phase") or "execute_objective").strip()
        rationale = str(
            directive.get("rationale") or "deterministic advisory fallback"
        ).strip()
        micro_plan = directive.get("micro_plan")
        success_criteria = directive.get("success_criteria")

        lowered = str(task or "").lower()
        if "login" in lowered or "sign in" in lowered:
            mode = "execution"
            phase = "authenticate"
            rationale = "goal implies authentication workflow"
        elif "error" in lowered or "fix" in lowered:
            mode = "recovery"
            phase = "recover_progress"
            rationale = "goal implies remediation or recovery path"

        if not isinstance(micro_plan, list) or not micro_plan:
            micro_plan = [
                "Take one objective-aligned action.",
                "Validate immediate effect.",
                "Switch path if no progress signal appears.",
            ]
        if not isinstance(success_criteria, list) or not success_criteria:
            success_criteria = [
                "state changes in objective-relevant way",
                "validation indicates progress",
            ]

        return {
            "mode": mode,
            "phase": phase,
            "rationale": rationale,
            "micro_plan": micro_plan,
            "success_criteria": success_criteria,
        }

    def _summarize_context_with_model(
        self, *, task: str, state: dict[str, Any]
    ) -> dict[str, Any]:
        """Deterministic fallback context summary for local/dev operation."""
        previous_summary = state.get("previous_summary")
        if not isinstance(previous_summary, dict):
            previous_summary = {}

        older_history = state.get("older_history")
        if not isinstance(older_history, list):
            older_history = []

        completed_milestones: list[str] = []
        open_issues: list[str] = []
        attempted_paths: list[str] = []
        constraints: list[str] = []

        for item in older_history:
            if not isinstance(item, dict):
                continue
            step_type = str(item.get("step_type") or "")
            tool_name = str(item.get("tool_name") or "")
            status = str(item.get("status") or "")
            if step_type == "tool" and tool_name:
                attempted_paths.append(f"{tool_name} -> {status or 'unknown'}")
                if status == "ok":
                    completed_milestones.append(f"{tool_name} succeeded")
                elif status == "failed":
                    error = str(item.get("error") or "").strip()
                    if error:
                        open_issues.append(error)
                    else:
                        open_issues.append(f"{tool_name} failed")
            elif step_type == "ask_user":
                constraints.append("workflow requested user input")

        objective_status = str(previous_summary.get("objective_status") or "").strip()
        if not objective_status:
            objective_status = f"working toward: {str(task or '').strip()[:180]}"
        progress_summary = str(previous_summary.get("progress_summary") or "").strip()
        if not progress_summary:
            progress_summary = (
                f"tracked {len(older_history)} prior step(s); latest retained as recent raw history"
            )

        next_focus_items = []
        raw_next_focus = previous_summary.get("next_focus")
        if isinstance(raw_next_focus, list):
            for item in raw_next_focus:
                text = str(item or "").strip()
                if text:
                    next_focus_items.append(text)
                if len(next_focus_items) >= 4:
                    break
        if not next_focus_items:
            next_focus_items = [
                "use recent raw history to choose one materially different next action"
            ]

        return {
            "objective_status": objective_status,
            "progress_summary": progress_summary,
            "completed_milestones": completed_milestones[:6],
            "open_issues": open_issues[:6],
            "attempted_paths": attempted_paths[:6],
            "constraints": constraints[:6],
            "next_focus": next_focus_items[:4],
        }

    @staticmethod
    def _sanitize_supervisor_advice(advice: dict[str, Any] | None) -> dict[str, Any]:
        """Clamp advisory output to safe deterministic schema bounds."""
        payload = advice if isinstance(advice, dict) else {}
        mode = str(payload.get("mode") or "execution").strip().lower()
        if mode not in {"exploration", "execution", "recovery"}:
            mode = "execution"

        def _clean_text(value: Any, max_len: int) -> str:
            text = " ".join(str(value or "").split()).strip()
            if not text:
                return ""
            if len(text) <= max_len:
                return text
            return text[:max_len].rstrip() + "..."

        def _clean_items(value: Any, max_items: int) -> list[str]:
            if not isinstance(value, list):
                return []
            items: list[str] = []
            for item in value:
                text = _clean_text(item, 180)
                if not text:
                    continue
                items.append(text)
                if len(items) >= max_items:
                    break
            return items

        phase = _clean_text(payload.get("phase"), 64) or "execute_objective"
        rationale = _clean_text(payload.get("rationale"), 220) or "no rationale provided"
        micro_plan = _clean_items(payload.get("micro_plan"), 5) or [
            "Take one objective-aligned action.",
            "Validate immediate effect.",
        ]
        success_criteria = _clean_items(payload.get("success_criteria"), 4) or [
            "state changes in objective-relevant way"
        ]
        return {
            "mode": mode,
            "phase": phase,
            "rationale": rationale,
            "micro_plan": micro_plan,
            "success_criteria": success_criteria,
        }

    @staticmethod
    def _sanitize_context_summary(summary: dict[str, Any] | None) -> dict[str, Any]:
        """Clamp context summary output to bounded, prompt-safe schema."""
        payload = summary if isinstance(summary, dict) else {}

        def _clean_text(value: Any, max_len: int) -> str:
            text = " ".join(str(value or "").split()).strip()
            if not text:
                return ""
            if len(text) <= max_len:
                return text
            return text[:max_len].rstrip() + "..."

        def _clean_items(value: Any, max_items: int, max_len: int = 180) -> list[str]:
            if not isinstance(value, list):
                return []
            items: list[str] = []
            for item in value:
                text = _clean_text(item, max_len)
                if not text:
                    continue
                items.append(text)
                if len(items) >= max_items:
                    break
            return items

        objective_status = (
            _clean_text(payload.get("objective_status"), 240) or "objective status unknown"
        )
        progress_summary = (
            _clean_text(payload.get("progress_summary"), 280)
            or "no material progress summary available"
        )
        completed_milestones = _clean_items(payload.get("completed_milestones"), 6)
        open_issues = _clean_items(payload.get("open_issues"), 6)
        attempted_paths = _clean_items(payload.get("attempted_paths"), 6)
        constraints = _clean_items(payload.get("constraints"), 6)
        next_focus = _clean_items(payload.get("next_focus"), 4)
        if not next_focus:
            next_focus = ["pick one next action that changes task-relevant state"]

        return {
            "objective_status": objective_status,
            "progress_summary": progress_summary,
            "completed_milestones": completed_milestones,
            "open_issues": open_issues,
            "attempted_paths": attempted_paths,
            "constraints": constraints,
            "next_focus": next_focus,
        }

    def _should_use_deepseek(self, *, model: str, allowed_tools: list[str]) -> bool:
        """Return whether DeepSeek tool-calling should be used."""
        model_name = str(model or "").strip().lower()
        return bool(
            model_name.startswith("deepseek")
            and os.getenv("DEEPSEEK_API_KEY")
            and allowed_tools
            and self.openai_tools_provider is not None
        )

    def _infer_with_openai(
        self,
        *,
        task: str,
        model: str,
        allowed_tools: list[str],
        context: dict[str, Any] | None = None,
        include_meta: bool = False,
    ) -> dict[str, Any]:
        """Infer a tool call via OpenAI Responses API."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI tool calling")
        endpoint = os.getenv(
            "OVERMIND_OPENAI_RESPONSES_URL",
            "https://api.openai.com/v1/responses",
        )
        return self._infer_with_openai_compatible(
            task=task,
            model=model,
            provider="openai",
            allowed_tools=allowed_tools,
            api_key=api_key,
            endpoint=endpoint,
            context=context,
            include_meta=include_meta,
        )

    def _infer_with_deepseek(
        self,
        *,
        task: str,
        model: str,
        allowed_tools: list[str],
        context: dict[str, Any] | None = None,
        include_meta: bool = False,
    ) -> dict[str, Any]:
        """Infer a tool call via DeepSeek's OpenAI-compatible API."""
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required for DeepSeek tool calling")
        endpoint = os.getenv(
            "OVERMIND_DEEPSEEK_CHAT_COMPLETIONS_URL",
            "https://api.deepseek.com/v1/chat/completions",
        )
        return self._infer_with_openai_compatible(
            task=task,
            model=model,
            provider="deepseek",
            allowed_tools=allowed_tools,
            api_key=api_key,
            endpoint=endpoint,
            context=context,
            include_meta=include_meta,
        )

    def _infer_with_openai_compatible(
        self,
        *,
        task: str,
        model: str,
        provider: str,
        allowed_tools: list[str],
        api_key: str,
        endpoint: str,
        context: dict[str, Any] | None = None,
        include_meta: bool = False,
    ) -> dict[str, Any]:
        """Infer one tool call using provider-specific OpenAI-compatible APIs."""
        if provider == "openai":
            return self._infer_with_openai_responses_api(
                task=task,
                model=model,
                allowed_tools=allowed_tools,
                api_key=api_key,
                endpoint=endpoint,
                include_meta=include_meta,
            )
        return self._infer_with_chat_completions_compatible(
            task=task,
            model=model,
            provider=provider,
            allowed_tools=allowed_tools,
            api_key=api_key,
            endpoint=endpoint,
            context=context,
            include_meta=include_meta,
        )

    def _infer_with_openai_responses_api(
        self,
        *,
        task: str,
        model: str,
        allowed_tools: list[str],
        api_key: str,
        endpoint: str,
        include_meta: bool,
    ) -> dict[str, Any]:
        """Infer one tool call using OpenAI Responses API."""
        tools, alias_map = self._get_openai_tools(allowed_tools)
        if not tools:
            raise RuntimeError("No allowed tools available for OpenAI tool calling")
        response_tools = self._to_responses_tools(tools)
        base_input = self._build_openai_responses_input(task)
        payload = {
            "model": model,
            "input": base_input,
            "tools": response_tools,
            "tool_choice": "auto",
            "temperature": 0,
        }
        payload, used_reasoning = self._apply_reasoning_request_tuning(
            payload=payload,
            model=model,
            provider="openai",
        )

        timeout_s = int(os.getenv("OVERMIND_OPENAI_TIMEOUT_S", "10"))
        try:
            data = self._request_openai_responses(
                payload=payload,
                api_key=api_key,
                endpoint=endpoint,
                timeout_s=timeout_s,
            )
        except RuntimeError:
            if not used_reasoning:
                raise
            data = self._request_openai_responses(
                payload=self._strip_reasoning_request_tuning(payload),
                api_key=api_key,
                endpoint=endpoint,
                timeout_s=timeout_s,
            )
        try:
            tool_name, args, raw_tool_call = self._parse_openai_responses_tool_call(
                data=data,
                allowed_tools=allowed_tools,
                alias_map=alias_map,
            )
        except RuntimeError as exc:
            if "openai response missing valid tool call" not in str(exc).lower():
                raise
            retry_payload = {
                "model": model,
                "input": [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "Select exactly one function call and return no plain text."
                                ),
                            }
                        ],
                    },
                    *base_input,
                ],
                "tools": response_tools,
                "tool_choice": "required",
                "temperature": 0,
            }
            retry_payload, retry_used_reasoning = self._apply_reasoning_request_tuning(
                payload=retry_payload,
                model=model,
                provider="openai",
            )
            try:
                retry_data = self._request_openai_responses(
                    payload=retry_payload,
                    api_key=api_key,
                    endpoint=endpoint,
                    timeout_s=timeout_s,
                )
            except RuntimeError:
                if not retry_used_reasoning:
                    raise
                retry_data = self._request_openai_responses(
                    payload=self._strip_reasoning_request_tuning(retry_payload),
                    api_key=api_key,
                    endpoint=endpoint,
                    timeout_s=timeout_s,
                )
            tool_name, args, raw_tool_call = self._parse_openai_responses_tool_call(
                data=retry_data,
                allowed_tools=allowed_tools,
                alias_map=alias_map,
            )
            data = retry_data
        result = {"tool_name": tool_name, "args": args}
        if include_meta:
            result.update(
                self._build_openai_responses_tool_response_metadata(
                    data=data,
                    raw_tool_call=raw_tool_call,
                )
            )
        return result

    def _infer_with_chat_completions_compatible(
        self,
        *,
        task: str,
        model: str,
        provider: str,
        allowed_tools: list[str],
        api_key: str,
        endpoint: str,
        context: dict[str, Any] | None = None,
        include_meta: bool = False,
    ) -> dict[str, Any]:
        """Infer one tool call using a chat completions API."""
        tools, alias_map = self._get_openai_tools(allowed_tools)
        if not tools:
            raise RuntimeError("No allowed tools available for OpenAI tool calling")
        base_messages = self._build_openai_messages(
            task=task,
            provider=provider,
            context=context,
        )
        payload = {
            "model": model,
            "messages": base_messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0,
        }
        payload, used_reasoning = self._apply_reasoning_request_tuning(
            payload=payload,
            model=model,
            provider=provider,
        )

        timeout_s = int(os.getenv("OVERMIND_OPENAI_TIMEOUT_S", "10"))
        try:
            data = self._request_openai_api(
                payload=payload,
                api_key=api_key,
                endpoint=endpoint,
                timeout_s=timeout_s,
            )
        except RuntimeError:
            if not used_reasoning:
                raise
            data = self._request_openai_api(
                payload=self._strip_reasoning_request_tuning(payload),
                api_key=api_key,
                endpoint=endpoint,
                timeout_s=timeout_s,
            )
        try:
            tool_name, args, raw_tool_call, message = self._parse_openai_tool_call(
                data=data,
                allowed_tools=allowed_tools,
                alias_map=alias_map,
            )
        except RuntimeError as exc:
            if "openai response missing valid tool call" not in str(exc).lower():
                raise
            retry_payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Select exactly one function call and return no plain text."
                        ),
                    },
                    *base_messages,
                ],
                "tools": tools,
                "tool_choice": "required",
                "temperature": 0,
            }
            retry_payload, retry_used_reasoning = self._apply_reasoning_request_tuning(
                payload=retry_payload,
                model=model,
                provider=provider,
            )
            try:
                retry_data = self._request_openai_api(
                    payload=retry_payload,
                    api_key=api_key,
                    endpoint=endpoint,
                    timeout_s=timeout_s,
                )
            except RuntimeError:
                if not retry_used_reasoning:
                    raise
                retry_data = self._request_openai_api(
                    payload=self._strip_reasoning_request_tuning(retry_payload),
                    api_key=api_key,
                    endpoint=endpoint,
                    timeout_s=timeout_s,
                )
            tool_name, args, raw_tool_call, message = self._parse_openai_tool_call(
                data=retry_data,
                allowed_tools=allowed_tools,
                alias_map=alias_map,
            )
        result = {"tool_name": tool_name, "args": args}
        if include_meta:
            result.update(
                self._build_openai_tool_response_metadata(
                    message=message,
                    raw_tool_call=raw_tool_call,
                )
            )
        return result

    @staticmethod
    def _build_openai_responses_input(task: str) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": task}],
            }
        ]

    @staticmethod
    def _to_responses_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert chat-completions function tool schema into responses schema."""
        converted: list[dict[str, Any]] = []
        for item in tools:
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "").strip().lower() != "function":
                converted.append(dict(item))
                continue
            function = item.get("function")
            if not isinstance(function, dict):
                converted.append(dict(item))
                continue
            name = str(function.get("name") or "").strip()
            if not name:
                continue
            payload: dict[str, Any] = {"type": "function", "name": name}
            description = function.get("description")
            if isinstance(description, str) and description.strip():
                payload["description"] = description
            parameters = function.get("parameters")
            if isinstance(parameters, dict):
                payload["parameters"] = parameters
            strict = function.get("strict")
            if isinstance(strict, bool):
                payload["strict"] = strict
            converted.append(payload)
        return converted

    def _request_openai_responses(
        self,
        *,
        payload: dict[str, Any],
        api_key: str,
        endpoint: str,
        timeout_s: int,
    ) -> dict[str, Any]:
        """Request OpenAI Responses API via the official OpenAI client."""
        if OpenAI is None:
            raise RuntimeError("openai package is required for OpenAI model calls")
        base_url = self._responses_base_url(endpoint)
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_s,
        )
        try:
            response = client.responses.create(**payload)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"openai request failed: {exc}") from exc
        return self._sdk_object_to_dict(response)

    @staticmethod
    def _responses_base_url(endpoint: str) -> str:
        """Resolve OpenAI client base URL from a Responses endpoint URL."""
        raw = str(endpoint or "").strip()
        if not raw:
            return "https://api.openai.com/v1"

        parsed = urlsplit(raw)
        if not parsed.scheme or not parsed.netloc:
            return "https://api.openai.com/v1"

        path = parsed.path.rstrip("/")
        if path.endswith("/responses"):
            path = path[: -len("/responses")]
        if not path:
            path = "/v1"
        return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))

    @staticmethod
    def _sdk_object_to_dict(value: Any) -> dict[str, Any]:
        """Convert SDK response objects to plain dictionaries."""
        if isinstance(value, dict):
            return dict(value)
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            output = to_dict()
            if isinstance(output, dict):
                return output
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            output = model_dump()
            if isinstance(output, dict):
                return output
        raise RuntimeError("openai response missing structured payload")

    @staticmethod
    def _request_openai_api(
        *,
        payload: dict[str, Any],
        api_key: str,
        endpoint: str,
        timeout_s: int,
    ) -> dict[str, Any]:
        req = urlrequest.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urlerror.URLError, TimeoutError) as exc:
            raise RuntimeError(f"openai request failed: {exc}") from exc

    @staticmethod
    def _parse_openai_responses_tool_call(
        *,
        data: dict[str, Any],
        allowed_tools: list[str],
        alias_map: dict[str, str],
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        output = data.get("output")
        if not isinstance(output, list):
            raise RuntimeError("openai response missing valid tool call")

        for item in output:
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "").strip().lower() != "function_call":
                continue
            raw_tool_name = str(item.get("name") or "")
            tool_name = alias_map.get(raw_tool_name, raw_tool_name)
            args_raw = item.get("arguments") or "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "openai response missing valid tool call"
                ) from exc
            if tool_name not in allowed_tools:
                raise RuntimeError(f"openai selected disallowed tool '{tool_name}'")
            if not isinstance(args, dict):
                raise RuntimeError("openai tool call arguments must be an object")
            raw_tool_call = ModelGateway._normalize_responses_tool_call(item)
            return tool_name, args, raw_tool_call

        raise RuntimeError("openai response missing valid tool call")

    @staticmethod
    def _normalize_responses_tool_call(item: dict[str, Any]) -> dict[str, Any]:
        name = str(item.get("name") or "").strip()
        call_id = str(item.get("call_id") or item.get("id") or "").strip()
        if not call_id:
            call_id = f"call_{name.replace('.', '_')}" if name else "call_function"
        args_payload = item.get("arguments")
        if isinstance(args_payload, str):
            args_json = args_payload
        else:
            args_json = json.dumps(args_payload or {}, separators=(",", ":"), ensure_ascii=True)
        return {
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": args_json,
            },
        }

    @staticmethod
    def _parse_openai_tool_call(
        *,
        data: dict[str, Any],
        allowed_tools: list[str],
        alias_map: dict[str, str],
    ) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
        try:
            message = data["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []
            first_call = tool_calls[0]
            raw_tool_name = str(first_call["function"]["name"])
            tool_name = alias_map.get(raw_tool_name, raw_tool_name)
            args_raw = first_call["function"].get("arguments") or "{}"
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except (KeyError, IndexError, TypeError, AttributeError, json.JSONDecodeError) as exc:
            raise RuntimeError("openai response missing valid tool call") from exc

        if tool_name not in allowed_tools:
            raise RuntimeError(f"openai selected disallowed tool '{tool_name}'")
        if not isinstance(args, dict):
            raise RuntimeError("openai tool call arguments must be an object")
        return tool_name, args, dict(first_call), dict(message)

    def _build_openai_messages(
        self,
        *,
        task: str,
        provider: str,
        context: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Build request messages, adding DeepSeek tool follow-up context when needed."""
        user_message = {"role": "user", "content": task}
        if provider != "deepseek":
            return [user_message]

        continuation = self._build_deepseek_tool_continuation_messages(context=context)
        if continuation:
            return [*continuation, user_message]
        return [user_message]

    def _build_deepseek_tool_continuation_messages(
        self, *, context: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        """Return assistant/tool replay messages for DeepSeek reasoning follow-up calls."""
        if self.repo is None or not isinstance(context, dict):
            return []

        run_id = str(context.get("run_id") or "").strip()
        history = context.get("history")
        if not run_id or not isinstance(history, list) or not history:
            return []

        last_tool_step: dict[str, Any] | None = None
        for item in reversed(history):
            if not isinstance(item, dict):
                continue
            if str(item.get("step_type") or "") == "tool":
                last_tool_step = item
                break
        if last_tool_step is None:
            return []

        last_input = dict(last_tool_step.get("input") or {})
        last_output = dict(last_tool_step.get("output") or {})
        last_tool_name = str(last_input.get("tool_name") or "")
        last_args = dict(last_input.get("args") or {})
        if not last_tool_name:
            return []

        model_calls = self.repo.list_model_calls(run_id)
        if not model_calls:
            return []
        last_response = dict(model_calls[-1].get("response_json") or {})
        if str(last_response.get("tool_name") or "") != last_tool_name:
            return []

        reasoning_content = self._extract_reasoning_text(
            last_response.get("reasoning_content")
        )
        if not reasoning_content:
            return []
        assistant_content = self._extract_reasoning_text(
            last_response.get("assistant_content")
        )

        normalized_tool_call = self._normalize_tool_call_for_replay(
            raw_tool_call=last_response.get("raw_tool_call"),
            fallback_tool_name=last_tool_name,
            fallback_args=last_args,
        )
        if not normalized_tool_call:
            return []

        assistant_message = {
            "role": "assistant",
            "content": self._merge_reasoning_and_assistant_content(
                reasoning_content=reasoning_content,
                assistant_content=assistant_content,
            ),
            "tool_calls": [normalized_tool_call],
        }
        tool_message = {
            "role": "tool",
            "tool_call_id": normalized_tool_call["id"],
            "content": self._serialize_tool_output_for_model(last_output),
        }
        return [assistant_message, tool_message]

    @staticmethod
    def _extract_reasoning_text(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        return text

    @staticmethod
    def _merge_reasoning_and_assistant_content(
        *, reasoning_content: str, assistant_content: str
    ) -> str:
        if reasoning_content and assistant_content:
            return f"{reasoning_content}\n{assistant_content}"
        return reasoning_content or assistant_content or ""

    @staticmethod
    def _normalize_tool_call_for_replay(
        *,
        raw_tool_call: Any,
        fallback_tool_name: str,
        fallback_args: dict[str, Any],
    ) -> dict[str, Any]:
        call = dict(raw_tool_call) if isinstance(raw_tool_call, dict) else {}
        function_payload = (
            dict(call.get("function")) if isinstance(call.get("function"), dict) else {}
        )

        tool_name = str(function_payload.get("name") or fallback_tool_name or "").strip()
        if not tool_name:
            return {}
        call_id = str(call.get("id") or "").strip()
        if not call_id:
            call_id = f"call_{tool_name.replace('.', '_')}"

        args_payload = function_payload.get("arguments")
        if isinstance(args_payload, str):
            args_json = args_payload
        else:
            if not isinstance(args_payload, dict):
                args_payload = fallback_args
            args_json = json.dumps(args_payload, separators=(",", ":"), ensure_ascii=True)
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": tool_name, "arguments": args_json},
        }

    @staticmethod
    def _serialize_tool_output_for_model(output: dict[str, Any], max_len: int = 6000) -> str:
        try:
            rendered = json.dumps(output, separators=(",", ":"), ensure_ascii=True)
        except TypeError:
            rendered = str(output)
        if len(rendered) <= max_len:
            return rendered
        return rendered[:max_len] + "..."

    def _build_openai_tool_response_metadata(
        self,
        *,
        message: dict[str, Any],
        raw_tool_call: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract provider-agnostic metadata from a tool-call completion."""
        metadata: dict[str, Any] = {}
        assistant_content = self._extract_assistant_content_from_message(message)
        reasoning_content = self._extract_reasoning_content_from_message(message)
        if assistant_content:
            metadata["assistant_content"] = assistant_content
        if reasoning_content:
            metadata["reasoning_content"] = reasoning_content
        if raw_tool_call:
            metadata["raw_tool_call"] = dict(raw_tool_call)
        return metadata

    def _build_openai_responses_tool_response_metadata(
        self,
        *,
        data: dict[str, Any],
        raw_tool_call: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract provider-agnostic metadata from a responses tool-call payload."""
        metadata: dict[str, Any] = {}
        assistant_content = self._extract_assistant_content_from_response(data)
        reasoning_content = self._extract_reasoning_content_from_response(data)
        if assistant_content:
            metadata["assistant_content"] = assistant_content
        if reasoning_content:
            metadata["reasoning_content"] = reasoning_content
        if raw_tool_call:
            metadata["raw_tool_call"] = dict(raw_tool_call)
        return metadata

    @staticmethod
    def _extract_assistant_content_from_message(message: dict[str, Any]) -> str:
        """Extract non-reasoning assistant content as plain text."""
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if not isinstance(content, list):
            return ""

        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item.strip())
                continue
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").strip().lower()
            if item_type.startswith("reasoning"):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts).strip()

    @staticmethod
    def _extract_reasoning_content_from_message(message: dict[str, Any]) -> str:
        """Extract reasoning text from DeepSeek/OpenAI-compatible message payloads."""
        parts: list[str] = []

        direct = message.get("reasoning_content")
        if isinstance(direct, str) and direct.strip():
            parts.append(direct.strip())

        reasoning_payload = message.get("reasoning")
        if isinstance(reasoning_payload, dict):
            summary = reasoning_payload.get("summary")
            if isinstance(summary, str) and summary.strip():
                parts.append(summary.strip())
            elif isinstance(summary, list):
                for item in summary:
                    if not isinstance(item, dict):
                        continue
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())

        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = str(item.get("type") or "").strip().lower()
                if not item_type.startswith("reasoning"):
                    continue
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())

        return "\n".join(parts).strip()

    @staticmethod
    def _extract_assistant_content_from_response(data: dict[str, Any]) -> str:
        """Extract assistant content from OpenAI Responses output payload."""
        top_level = data.get("output_text")
        if isinstance(top_level, str) and top_level.strip():
            return top_level.strip()

        output = data.get("output")
        if not isinstance(output, list):
            return ""

        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").strip().lower()
            if item_type != "message":
                continue
            content = item.get("content")
            if isinstance(content, str):
                if content.strip():
                    parts.append(content.strip())
                continue
            if not isinstance(content, list):
                continue
            for chunk in content:
                if isinstance(chunk, str):
                    if chunk.strip():
                        parts.append(chunk.strip())
                    continue
                if not isinstance(chunk, dict):
                    continue
                chunk_type = str(chunk.get("type") or "").strip().lower()
                if chunk_type.startswith("reasoning"):
                    continue
                text = chunk.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()

    @staticmethod
    def _extract_reasoning_content_from_response(data: dict[str, Any]) -> str:
        """Extract reasoning content from OpenAI Responses output payload."""
        parts: list[str] = []

        reasoning = data.get("reasoning")
        if isinstance(reasoning, dict):
            summary = reasoning.get("summary")
            if isinstance(summary, str) and summary.strip():
                parts.append(summary.strip())
            elif isinstance(summary, list):
                for item in summary:
                    if not isinstance(item, dict):
                        continue
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())

        output = data.get("output")
        if not isinstance(output, list):
            return "\n".join(parts).strip()

        for item in output:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").strip().lower()
            if item_type.startswith("reasoning"):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
                summary = item.get("summary")
                if isinstance(summary, str) and summary.strip():
                    parts.append(summary.strip())
                elif isinstance(summary, list):
                    for summary_item in summary:
                        if not isinstance(summary_item, dict):
                            continue
                        summary_text = summary_item.get("text")
                        if isinstance(summary_text, str) and summary_text.strip():
                            parts.append(summary_text.strip())
                continue
            if item_type != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for chunk in content:
                if not isinstance(chunk, dict):
                    continue
                chunk_type = str(chunk.get("type") or "").strip().lower()
                if not chunk_type.startswith("reasoning"):
                    continue
                text = chunk.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())

        return "\n".join(parts).strip()

    def _apply_reasoning_request_tuning(
        self,
        *,
        payload: dict[str, Any],
        model: str,
        provider: str,
    ) -> tuple[dict[str, Any], bool]:
        """Apply provider/model-specific reasoning settings to request payloads."""
        updated = dict(payload)
        applied = False

        if provider == "deepseek":
            thinking_mode = self._deepseek_thinking_mode()
            if thinking_mode:
                updated["thinking"] = {"type": thinking_mode}
                updated.pop("temperature", None)
                applied = True

        if not self._is_reasoning_model(model):
            return updated, applied

        effort = self._reasoning_effort(provider=provider)
        if provider == "deepseek":
            updated["reasoning_effort"] = effort
        else:
            updated["reasoning"] = {"effort": effort}

        # Reasoning models often ignore or reject temperature controls.
        updated.pop("temperature", None)
        return updated, True

    @staticmethod
    def _strip_reasoning_request_tuning(payload: dict[str, Any]) -> dict[str, Any]:
        """Remove reasoning-specific request fields for compatibility fallback."""
        updated = dict(payload)
        updated.pop("reasoning", None)
        updated.pop("reasoning_effort", None)
        updated.pop("thinking", None)
        return updated

    @staticmethod
    def _is_reasoning_model(model: str) -> bool:
        """Return whether model name suggests reasoning-focused behavior."""
        name = str(model or "").strip().lower()
        if not name:
            return False
        markers = (
            "reasoner",
            "reasoning",
            "o1",
            "o3",
            "o4",
            "gpt-5",
        )
        return any(marker in name for marker in markers)

    @staticmethod
    def _reasoning_effort(*, provider: str) -> str:
        """Resolve reasoning effort level with provider-specific env override."""
        default = "medium"
        if provider == "deepseek":
            value = os.getenv("OVERMIND_DEEPSEEK_REASONING_EFFORT", default)
        else:
            value = os.getenv("OVERMIND_OPENAI_REASONING_EFFORT", default)
        normalized = str(value or "").strip().lower()
        if normalized in {"low", "medium", "high"}:
            return normalized
        return default

    @staticmethod
    def _deepseek_thinking_mode() -> str | None:
        """Resolve optional DeepSeek thinking mode request setting."""
        raw = str(os.getenv("OVERMIND_DEEPSEEK_THINKING_MODE", "") or "").strip().lower()
        if raw in {"enabled", "disabled"}:
            return raw
        return None

    def _get_openai_tools(
        self, allowed_tools: list[str]
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """Return OpenAI-compatible tool definitions for allowed tools.

        Args:
            allowed_tools: Allowed tool names for the current agent.

        Returns:
            Tool definitions and alias->internal-name mapping.
        """
        if self.openai_tools_provider is not None:
            payload = self.openai_tools_provider(allowed_tools)
            if isinstance(payload, tuple) and len(payload) == 2:
                tools, aliases = payload
                return list(tools), dict(aliases)
            return list(payload), {}
        return [], {}

    def _infer_with_model(self, *, task: str, model: str) -> dict[str, Any]:
        """Infer a tool call using model-specific logic.

        This implementation uses lightweight pattern matching to produce a tool
        decision. It is primarily meant for deterministic testing and local
        development.

        Args:
            task: Task text.
            model: Model identifier.

        Returns:
            Dict containing `tool_name` and `args`.

        Raises:
            ValueError: If the model is configured to fail.
        """
        raw_task = (task or "").strip()
        normalized_task = self._extract_task_from_policy_prompt(raw_task)
        lowered = normalized_task.lower()
        if model.startswith("fail"):
            raise ValueError(f"model '{model}' is unavailable")

        if self._is_followup_policy_prompt(raw_task) and not self._is_failure_prompt(
            raw_task
        ):
            return {"tool_name": "final_answer", "args": {"message": "done"}}

        curl_url = self._extract_url(normalized_task)
        if "curl" in lowered:
            if not curl_url:
                return {
                    "tool_name": "ask_user",
                    "args": {"message": "Please provide the full URL to fetch."},
                }
            return {
                "tool_name": "run_shell",
                "args": {"command": f"curl -L --max-time 20 {curl_url}"},
            }

        if self._looks_like_shell(lowered):
            command = self._extract_shell_command(normalized_task)
            return {"tool_name": "run_shell", "args": {"command": command}}

        if self._looks_like_write(lowered):
            path, content = self._extract_write_payload(normalized_task)
            return {
                "tool_name": "write_file",
                "args": {"path": path, "content": content},
            }

        if self._looks_like_read(lowered):
            path = self._extract_read_path(normalized_task)
            return {"tool_name": "read_file", "args": {"path": path}}

        if self._looks_like_recall(lowered):
            collection, query = self._extract_memory_query(normalized_task)
            return {
                "tool_name": "search_memory",
                "args": {"collection": collection, "query": query, "top_k": 5},
            }

        if self._looks_like_remember(lowered):
            collection, text = self._extract_memory_store(normalized_task)
            return {
                "tool_name": "store_memory",
                "args": {"collection": collection, "text": text},
            }

        return {
            "tool_name": "store_memory",
            "args": {
                "collection": "runs",
                "text": normalized_task,
                "metadata": {"source": "task"},
            },
        }

    @staticmethod
    def _extract_task_from_policy_prompt(task: str) -> str:
        """Extract original task text from policy-generated prompts when present."""
        text = str(task or "").strip()
        if not text.lower().startswith("task:"):
            return text
        match = re.search(
            (
                r"^\s*Task:\s*(.*?)\n\n(?:"
                r"Last tool:|Recent run history:|Prior failures:|"
                r"Decide the next action\.|Choose the next action\.)"
            ),
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        return re.sub(r"^\s*Task:\s*", "", text, flags=re.IGNORECASE).strip()

    @staticmethod
    def _is_followup_policy_prompt(task: str) -> bool:
        text = str(task or "")
        return "Last tool:" in text and "Last tool result summary:" in text

    @staticmethod
    def _is_failure_prompt(task: str) -> bool:
        text = str(task or "")
        return "Most recent failure:" in text and "Prior failures:" in text

    @staticmethod
    def _extract_url(task: str) -> str | None:
        match = re.search(r"https?://[^\s'\"]+", task or "")
        if not match:
            return None
        return match.group(0)

    @staticmethod
    def _estimate_usage(task: str, response: dict[str, Any]) -> dict[str, int]:
        """Estimate token usage for telemetry.

        Args:
            task: Input task string.
            response: Model response payload.

        Returns:
            Usage dict with `input_tokens`, `output_tokens`, and `total_tokens`.
        """
        input_tokens = max(1, len((task or "").split()))
        output_tokens = max(1, len(str(response).split()))
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    @staticmethod
    def _looks_like_shell(task_lower: str) -> bool:
        """Heuristically detect when a task requests shell execution.

        Args:
            task_lower: Lowercased task string.

        Returns:
            True if the task likely intends to execute a shell command.
        """
        return bool(
            re.search(r"\b(shell|run|execute|cmd|command)\b", task_lower)
            or re.match(r"\s*shell\s*:", task_lower)
        )

    @staticmethod
    def _looks_like_read(task_lower: str) -> bool:
        """Heuristically detect when a task requests reading a file.

        Args:
            task_lower: Lowercased task string.

        Returns:
            True if the task likely intends to read/open a file.
        """
        return bool(
            re.search(r"\b(read|open|cat|show)\b", task_lower)
            or re.match(r"\s*read\s*:", task_lower)
        )

    @staticmethod
    def _looks_like_write(task_lower: str) -> bool:
        """Heuristically detect when a task requests writing a file.

        Args:
            task_lower: Lowercased task string.

        Returns:
            True if the task likely intends to write/save/append a file.
        """
        return bool(
            re.search(r"\b(write|save|append)\b", task_lower)
            or re.match(r"\s*write\s*:", task_lower)
        )

    @staticmethod
    def _looks_like_remember(task_lower: str) -> bool:
        """Heuristically detect when a task requests storing memory.

        Args:
            task_lower: Lowercased task string.

        Returns:
            True if the task likely intends to store a memory item.
        """
        return bool(re.search(r"\b(remember|store|memorize)\b", task_lower))

    @staticmethod
    def _looks_like_recall(task_lower: str) -> bool:
        """Heuristically detect when a task requests searching memory.

        Args:
            task_lower: Lowercased task string.

        Returns:
            True if the task likely intends to search previously stored memory.
        """
        return bool(re.search(r"\b(recall|search|find|lookup)\b", task_lower))

    @staticmethod
    def _extract_shell_command(task: str) -> str:
        """Extract a shell command from a task string.

        Supports the prefix format `shell: <command>`.

        Args:
            task: Raw task string.

        Returns:
            Extracted shell command string.
        """
        prefixed = re.match(r"\s*shell\s*:\s*(.+)$", task, flags=re.IGNORECASE)
        if prefixed:
            return prefixed.group(1).strip()
        return task.strip()

    @staticmethod
    def _extract_read_path(task: str) -> str:
        """Extract a file path from a task string.

        Supports the prefix format `read: <path>`. Otherwise attempts to pull a
        quoted path, falling back to the last whitespace-delimited token.

        Args:
            task: Raw task string.

        Returns:
            Extracted file path string, or an empty string if none is found.
        """
        prefixed = re.match(r"\s*read\s*:\s*(.+)$", task, flags=re.IGNORECASE)
        if prefixed:
            return prefixed.group(1).strip()
        quoted = re.search(r"['\"]([^'\"]+)['\"]", task)
        if quoted:
            return quoted.group(1).strip()
        tokens = task.split()
        if tokens:
            return tokens[-1].strip()
        return ""

    @staticmethod
    def _extract_write_payload(task: str) -> tuple[str, str]:
        """Extract a (path, content) pair from a task string.

        Supports the prefix format `write: <path>: <content>` and a simple
        `... to <path>` heuristic.

        Args:
            task: Raw task string.

        Returns:
            Tuple of `(path, content)`.
        """
        prefixed = re.match(
            r"\s*write\s*:\s*([^:]+)\s*:\s*(.*)$", task, flags=re.IGNORECASE
        )
        if prefixed:
            return prefixed.group(1).strip(), prefixed.group(2)
        parts = task.split(" to ", 1)
        if len(parts) == 2:
            return parts[1].strip(), parts[0].strip()
        return "notes.txt", task.strip()

    @staticmethod
    def _extract_memory_store(task: str) -> tuple[str, str]:
        """Extract a (collection, text) pair for storing memory.

        Supports the prefix format `remember: <collection>: <text>`.

        Args:
            task: Raw task string.

        Returns:
            Tuple of `(collection, text)`.
        """
        prefixed = re.match(
            r"\s*remember\s*:\s*([^:]*)\s*:\s*(.*)$", task, flags=re.IGNORECASE
        )
        if prefixed:
            collection = prefixed.group(1).strip() or "default"
            return collection, prefixed.group(2).strip()
        return "default", task.strip()

    @staticmethod
    def _extract_memory_query(task: str) -> tuple[str, str]:
        """Extract a (collection, query) pair for searching memory.

        Supports the prefix format `recall: <collection>: <query>`.

        Args:
            task: Raw task string.

        Returns:
            Tuple of `(collection, query)`.
        """
        prefixed = re.match(
            r"\s*recall\s*:\s*([^:]*)\s*:\s*(.*)$", task, flags=re.IGNORECASE
        )
        if prefixed:
            collection = prefixed.group(1).strip() or "default"
            return collection, prefixed.group(2).strip()
        return "default", task.strip()
