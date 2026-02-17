"""Deterministic policy implementation.

Uses simple intent classification and structured extraction rules to convert a
free-form task string into planned actions without prefix-only parsing.
"""

from __future__ import annotations

import re
from typing import Any

from .policy import PlanContext, PlannedAction, Policy


class DeterministicPolicy(Policy):
    """Deterministic intent-based policy for tool planning."""

    def plan(
        self,
        task: str,
        *,
        agent: dict[str, Any],
        context: PlanContext,
    ) -> list[PlannedAction]:
        normalized = (task or "").strip()
        actions: list[PlannedAction] = [
            PlannedAction(
                step_type="plan",
                tool_name=None,
                args={
                    "task": normalized,
                    "agent_id": agent.get("id"),
                    "run_id": context.get("run_id"),
                },
            )
        ]

        action = self._action_for_task(normalized)
        actions.append(action)
        actions.append(
            PlannedAction(step_type="eval", tool_name=None, args={"result": "done"})
        )
        return actions

    def _action_for_task(self, task: str) -> PlannedAction:
        lowered = task.lower()

        if self._looks_like_shell(lowered):
            command = self._extract_shell_command(task)
            return PlannedAction("tool", "run_shell", {"command": command})

        if self._looks_like_write(lowered):
            path, content = self._extract_write_payload(task)
            return PlannedAction(
                "tool", "write_file", {"path": path, "content": content}
            )

        if self._looks_like_read(lowered):
            path = self._extract_read_path(task)
            return PlannedAction("tool", "read_file", {"path": path})

        if self._looks_like_recall(lowered):
            collection, query = self._extract_memory_query(task)
            return PlannedAction(
                "tool",
                "search_memory",
                {"collection": collection, "query": query, "top_k": 5},
            )

        if self._looks_like_remember(lowered):
            collection, text = self._extract_memory_store(task)
            return PlannedAction(
                "tool", "store_memory", {"collection": collection, "text": text}
            )

        return PlannedAction(
            "tool",
            "store_memory",
            {
                "collection": "runs",
                "text": task,
                "metadata": {"source": "task"},
            },
        )

    @staticmethod
    def _looks_like_shell(task_lower: str) -> bool:
        return bool(
            re.search(r"\b(shell|run|execute|cmd|command)\b", task_lower)
            or re.match(r"\s*shell\s*:", task_lower)
        )

    @staticmethod
    def _looks_like_read(task_lower: str) -> bool:
        return bool(
            re.search(r"\b(read|open|cat|show)\b", task_lower)
            or re.match(r"\s*read\s*:", task_lower)
        )

    @staticmethod
    def _looks_like_write(task_lower: str) -> bool:
        return bool(
            re.search(r"\b(write|save|append)\b", task_lower)
            or re.match(r"\s*write\s*:", task_lower)
        )

    @staticmethod
    def _looks_like_remember(task_lower: str) -> bool:
        return bool(re.search(r"\b(remember|store|memorize)\b", task_lower))

    @staticmethod
    def _looks_like_recall(task_lower: str) -> bool:
        return bool(re.search(r"\b(recall|search|find|lookup)\b", task_lower))

    @staticmethod
    def _extract_shell_command(task: str) -> str:
        prefixed = re.match(r"\s*shell\s*:\s*(.+)$", task, flags=re.IGNORECASE)
        if prefixed:
            return prefixed.group(1).strip()
        return task.strip()

    @staticmethod
    def _extract_read_path(task: str) -> str:
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
        prefixed = re.match(
            r"\s*remember\s*:\s*([^:]*)\s*:\s*(.*)$", task, flags=re.IGNORECASE
        )
        if prefixed:
            collection = prefixed.group(1).strip() or "default"
            return collection, prefixed.group(2).strip()
        return "default", task.strip()

    @staticmethod
    def _extract_memory_query(task: str) -> tuple[str, str]:
        prefixed = re.match(
            r"\s*recall\s*:\s*([^:]*)\s*:\s*(.*)$", task, flags=re.IGNORECASE
        )
        if prefixed:
            collection = prefixed.group(1).strip() or "default"
            return collection, prefixed.group(2).strip()
        return "default", task.strip()
