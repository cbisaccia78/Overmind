"""Deterministic fallback policy.

This policy is intentionally simple and finite. It is useful as a fallback
when fully iterative model-driven behavior is not desired.
"""

from __future__ import annotations

from typing import Any

from .model_gateway import ModelGateway
from .model_driven_policy import ModelDrivenPolicy
from .policy import PlanContext


class DeterministicPolicy(ModelDrivenPolicy):
    """Fallback policy with deterministic stop behavior.

    Behavior:
    - First step: ask model gateway for one action using the raw task.
    - After one successful tool call: emit `final_answer` with a summary.
    - After a failed tool call: emit `final_answer` with failure details.

    This keeps runs predictable and bounded while still reusing the same
    action-mapping and output summarization helpers as `ModelDrivenPolicy`.
    """

    def __init__(self, model_gateway: ModelGateway):
        super().__init__(model_gateway=model_gateway)

    def next_action(
        self,
        task: str,
        *,
        agent: dict[str, Any],
        context: PlanContext,
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        tool_steps = [item for item in history if item.get("step_type") == "tool"]
        if not tool_steps:
            return self._infer_next_action(
                prompt=(task or "").strip(),
                agent=agent,
                context=context,
            )

        last_tool = tool_steps[-1]
        tool_input = dict(last_tool.get("input") or {})
        output = dict(last_tool.get("output") or {})

        if not output.get("ok"):
            err = str((output.get("error") or {}).get("message") or "tool failed")
            return {"kind": "final_answer", "message": f"Tool failed: {err}"}

        last_tool_name = str(tool_input.get("tool_name") or "")
        if last_tool_name in {"ask_user", "final_answer"}:
            message = str(output.get("message") or "done")
            if last_tool_name == "ask_user":
                return {"kind": "ask_user", "message": message}
            return {"kind": "final_answer", "message": message}

        return {
            "kind": "final_answer",
            "message": self._summarize_tool_output(output),
        }
