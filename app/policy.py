"""Task-to-action policy.

Parses a task string into a deterministic sequence of planned actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PlannedAction:
    """A single planned step emitted by a policy.

    Attributes:
        step_type: Logical step type (e.g. "plan", "tool", "eval").
        tool_name: Tool name to invoke for tool steps, else None.
        args: Arguments payload for the step/tool.
    """

    step_type: str
    tool_name: str | None
    args: dict[str, Any]


class StubPolicy:
    """Deterministic parser from task string to tool actions."""

    def plan(self, task: str) -> list[PlannedAction]:
        """Convert a task string into a deterministic list of planned actions.

        Args:
            task: Task string.

        Returns:
            List of `PlannedAction` items.
        """
        actions: list[PlannedAction] = [
            PlannedAction(step_type="plan", tool_name=None, args={"task": task})
        ]

        if task.startswith("shell:"):
            cmd = task.split("shell:", 1)[1].strip()
            actions.append(
                PlannedAction(
                    step_type="tool", tool_name="run_shell", args={"command": cmd}
                )
            )
        elif task.startswith("read:"):
            path = task.split("read:", 1)[1].strip()
            actions.append(
                PlannedAction(
                    step_type="tool", tool_name="read_file", args={"path": path}
                )
            )
        elif task.startswith("write:"):
            payload = task.split("write:", 1)[1]
            path, _, content = payload.partition(":")
            actions.append(
                PlannedAction(
                    step_type="tool",
                    tool_name="write_file",
                    args={"path": path.strip(), "content": content},
                )
            )
        elif task.startswith("remember:"):
            payload = task.split("remember:", 1)[1]
            collection, _, text = payload.partition(":")
            actions.append(
                PlannedAction(
                    step_type="tool",
                    tool_name="store_memory",
                    args={
                        "collection": collection.strip() or "default",
                        "text": text.strip(),
                    },
                )
            )
        elif task.startswith("recall:"):
            payload = task.split("recall:", 1)[1]
            collection, _, query = payload.partition(":")
            actions.append(
                PlannedAction(
                    step_type="tool",
                    tool_name="search_memory",
                    args={
                        "collection": collection.strip() or "default",
                        "query": query.strip(),
                        "top_k": 5,
                    },
                )
            )
        else:
            actions.append(
                PlannedAction(
                    step_type="tool",
                    tool_name="store_memory",
                    args={
                        "collection": "runs",
                        "text": task,
                        "metadata": {"source": "task"},
                    },
                )
            )

        actions.append(
            PlannedAction(step_type="eval", tool_name=None, args={"result": "done"})
        )
        return actions
