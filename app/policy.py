"""Policy types and interfaces for task planning."""

from __future__ import annotations

from abc import ABC, abstractmethod
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


PlanContext = dict[str, Any]


class Policy(ABC):
    """Interface for converting tasks into executable planned actions."""

    @abstractmethod
    def plan(
        self,
        task: str,
        *,
        agent: dict[str, Any],
        context: PlanContext,
    ) -> list[PlannedAction]:
        """Build a deterministic or model-assisted action plan for a task.

        Args:
            task: User task string.
            agent: Agent configuration and metadata for this run.
            context: Additional planning context (run id, limits, prior state, etc.).

        Returns:
            Ordered list of planned actions.
        """
