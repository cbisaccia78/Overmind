"""Model inference gateway.

Generates tool-call decisions from task text using an agent's configured model,
and persists model interaction telemetry for auditability.
"""

from __future__ import annotations

import re
import time
from typing import Any

from .repository import Repository


class ModelGateway:
    """Model inference facade that records request/response telemetry.

    The gateway returns a normalized tool-call decision and persists a
    corresponding row in `model_calls` for auditing.
    """

    def __init__(self, repo: Repository):
        """Create a model gateway.

        Args:
            repo: Repository used to persist model call audit records.
        """
        self.repo = repo

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
        request_json = {
            "task": task,
            "agent_id": agent.get("id"),
            "model": model,
            "context": context,
        }

        start = time.monotonic()
        try:
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
        lowered = (task or "").strip().lower()
        if model.startswith("fail"):
            raise ValueError(f"model '{model}' is unavailable")

        if self._looks_like_shell(lowered):
            command = self._extract_shell_command(task)
            return {"tool_name": "run_shell", "args": {"command": command}}

        if self._looks_like_write(lowered):
            path, content = self._extract_write_payload(task)
            return {
                "tool_name": "write_file",
                "args": {"path": path, "content": content},
            }

        if self._looks_like_read(lowered):
            path = self._extract_read_path(task)
            return {"tool_name": "read_file", "args": {"path": path}}

        if self._looks_like_recall(lowered):
            collection, query = self._extract_memory_query(task)
            return {
                "tool_name": "search_memory",
                "args": {"collection": collection, "query": query, "top_k": 5},
            }

        if self._looks_like_remember(lowered):
            collection, text = self._extract_memory_store(task)
            return {
                "tool_name": "store_memory",
                "args": {"collection": collection, "text": text},
            }

        return {
            "tool_name": "store_memory",
            "args": {
                "collection": "runs",
                "text": task,
                "metadata": {"source": "task"},
            },
        }

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
