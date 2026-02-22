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
from typing import Any, Callable

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

        start = time.monotonic()
        try:
            model_name = str(model or "").strip().lower()
            if self._should_use_deepseek(model=model, allowed_tools=allowed_tools):
                response_json = self._infer_with_deepseek(
                    task=task,
                    model=model,
                    allowed_tools=allowed_tools,
                )
            elif model_name.startswith("deepseek"):
                response_json = self._infer_with_model(task=task, model=model)
            elif self._should_use_openai(allowed_tools):
                response_json = self._infer_with_openai(
                    task=task,
                    model=model,
                    allowed_tools=allowed_tools,
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

        latency_ms = int((time.monotonic() - start) * 1000)
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

        start = time.monotonic()
        try:
            model_name = str(model or "").strip().lower()
            if model_name.startswith("deepseek") and os.getenv("DEEPSEEK_API_KEY"):
                raw = self._advise_with_openai_compatible(
                    model=model,
                    api_key=os.getenv("DEEPSEEK_API_KEY") or "",
                    endpoint=os.getenv(
                        "OVERMIND_DEEPSEEK_CHAT_COMPLETIONS_URL",
                        "https://api.deepseek.com/v1/chat/completions",
                    ),
                    task=task,
                    state=state,
                )
            elif os.getenv("OPENAI_API_KEY"):
                raw = self._advise_with_openai_compatible(
                    model=model,
                    api_key=os.getenv("OPENAI_API_KEY") or "",
                    endpoint=os.getenv(
                        "OVERMIND_OPENAI_CHAT_COMPLETIONS_URL",
                        "https://api.openai.com/v1/chat/completions",
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

    def _advise_with_openai_compatible(
        self,
        *,
        model: str,
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
        timeout_s = int(os.getenv("OVERMIND_OPENAI_TIMEOUT_S", "10"))
        data = self._request_openai_chat_completion(
            payload=payload,
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

    def _parse_supervisor_json_response(self, data: dict[str, Any]) -> dict[str, Any]:
        """Parse JSON content from chat completion response."""
        content = self._extract_message_content(data)
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
    def _extract_message_content(data: dict[str, Any]) -> str:
        """Extract chat message content as plain text."""
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            return ""
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
            return "\n".join(parts).strip()
        return ""

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
    ) -> dict[str, Any]:
        """Infer a tool call via OpenAI-compatible chat-completions."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI tool calling")
        endpoint = os.getenv(
            "OVERMIND_OPENAI_CHAT_COMPLETIONS_URL",
            "https://api.openai.com/v1/chat/completions",
        )
        return self._infer_with_openai_compatible(
            task=task,
            model=model,
            allowed_tools=allowed_tools,
            api_key=api_key,
            endpoint=endpoint,
        )

    def _infer_with_deepseek(
        self,
        *,
        task: str,
        model: str,
        allowed_tools: list[str],
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
            allowed_tools=allowed_tools,
            api_key=api_key,
            endpoint=endpoint,
        )

    def _infer_with_openai_compatible(
        self,
        *,
        task: str,
        model: str,
        allowed_tools: list[str],
        api_key: str,
        endpoint: str,
    ) -> dict[str, Any]:
        """Infer one tool call using an OpenAI-compatible chat completions API."""
        tools, alias_map = self._get_openai_tools(allowed_tools)
        if not tools:
            raise RuntimeError("No allowed tools available for OpenAI tool calling")
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": task}],
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0,
        }

        timeout_s = int(os.getenv("OVERMIND_OPENAI_TIMEOUT_S", "10"))
        data = self._request_openai_chat_completion(
            payload=payload,
            api_key=api_key,
            endpoint=endpoint,
            timeout_s=timeout_s,
        )
        try:
            tool_name, args = self._parse_openai_tool_call(
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
                    {"role": "user", "content": task},
                ],
                "tools": tools,
                "tool_choice": "required",
                "temperature": 0,
            }
            retry_data = self._request_openai_chat_completion(
                payload=retry_payload,
                api_key=api_key,
                endpoint=endpoint,
                timeout_s=timeout_s,
            )
            tool_name, args = self._parse_openai_tool_call(
                data=retry_data,
                allowed_tools=allowed_tools,
                alias_map=alias_map,
            )

        return {"tool_name": tool_name, "args": args}

    @staticmethod
    def _request_openai_chat_completion(
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
    def _parse_openai_tool_call(
        *,
        data: dict[str, Any],
        allowed_tools: list[str],
        alias_map: dict[str, str],
    ) -> tuple[str, dict[str, Any]]:
        try:
            message = data["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []
            first_call = tool_calls[0]
            raw_tool_name = str(first_call["function"]["name"])
            tool_name = alias_map.get(raw_tool_name, raw_tool_name)
            args_raw = first_call["function"].get("arguments") or "{}"
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("openai response missing valid tool call") from exc

        if tool_name not in allowed_tools:
            raise RuntimeError(f"openai selected disallowed tool '{tool_name}'")
        if not isinstance(args, dict):
            raise RuntimeError("openai tool call arguments must be an object")
        return tool_name, args

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
