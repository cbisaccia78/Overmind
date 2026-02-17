"""Deterministic policy implementation.

Uses simple intent classification and structured extraction rules to convert a
free-form task string into planned actions without prefix-only parsing.
"""

from __future__ import annotations

from typing import Any

from .model_gateway import ModelGateway
from .policy import PlanContext, PlannedAction, Policy


class DeterministicPolicy(Policy):
    """Deterministic intent-based policy for tool planning.

    This policy delegates "model inference" to `ModelGateway` (which may be a
    real model backend or a deterministic heuristic), then converts the
    resulting tool decision into a short, fixed action sequence.
    """

    def __init__(self, model_gateway: ModelGateway):
        """Create a deterministic policy.

        Args:
            model_gateway: Model inference gateway used to decide the next tool.
        """
        self.model_gateway = model_gateway

    def plan(
        self,
        task: str,
        *,
        agent: dict[str, Any],
        context: PlanContext,
    ) -> list[PlannedAction]:
        """Plan actions for a task.

        Args:
            task: Free-form task string.
            agent: Agent configuration and metadata.
            context: Planning context (includes run id, limits, etc.).

        Returns:
            Ordered list of planned actions.
        """
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

        inference = self.model_gateway.infer(
            task=normalized,
            agent=agent,
            context=context,
        )

        if not inference.get("ok"):
            message = inference.get("error", {}).get(
                "message", "model inference failed"
            )
            actions.append(
                PlannedAction(
                    step_type="tool",
                    tool_name="store_memory",
                    args={
                        "collection": "runs",
                        "text": normalized,
                        "metadata": {"source": "task", "model_error": message},
                    },
                )
            )
        else:
            action = PlannedAction(
                step_type="tool",
                tool_name=str(inference.get("tool_name")),
                args=dict(inference.get("args") or {}),
            )
            actions.append(action)
        actions.append(
            PlannedAction(step_type="eval", tool_name=None, args={"result": "done"})
        )
        return actions
