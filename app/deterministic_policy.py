"""Deterministic policy implementation.

Uses simple intent classification and structured extraction rules to convert a
free-form task string into planned actions without prefix-only parsing.
"""

from __future__ import annotations

import re
import os
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

    def next_action(
        self,
        task: str,
        *,
        agent: dict[str, Any],
        context: PlanContext,
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return the next action for iterative agent execution.

        Args:
            task: Current task text (may include appended user input).
            agent: Agent configuration and metadata.
            context: Planning context.
            history: Prior step observations for this run.

        Returns:
            Action dict with `kind` in {tool_call, ask_user, final_answer}.
        """
        allowed_tools = {str(name) for name in (agent.get("tools_allowed") or [])}
        tool_steps = [item for item in history if item.get("step_type") == "tool"]
        lowered = (task or "").lower()

        use_openai = bool(os.getenv("OPENAI_API_KEY") and allowed_tools)

        if not tool_steps:
            if "curl" in lowered and not self._extract_url(task):
                return {
                    "kind": "ask_user",
                    "message": "Please provide the full URL to fetch.",
                }

            if "curl" in lowered and "run_shell" in allowed_tools:
                url = self._extract_url(task)
                if url:
                    return {
                        "kind": "tool_call",
                        "tool_name": "run_shell",
                        "args": {"command": f"curl -L --max-time 20 {url}"},
                    }

            inference = self.model_gateway.infer(
                task=(task or "").strip(),
                agent=agent,
                context=context,
            )
            if not inference.get("ok"):
                message = inference.get("error", {}).get(
                    "message", "model inference failed"
                )
                return {
                    "kind": "final_answer",
                    "message": f"I could not infer a tool action: {message}",
                }
            return {
                "kind": "tool_call",
                "tool_name": str(inference.get("tool_name")),
                "args": dict(inference.get("args") or {}),
            }

        last_tool = tool_steps[-1]
        last_input = dict(last_tool.get("input") or {})
        last_tool_name = str(last_input.get("tool_name") or "")
        output = dict(last_tool.get("output") or {})
        if not output.get("ok"):
            err = output.get("error", {}).get("message", "tool failed")
            return {"kind": "final_answer", "message": f"Tool failed: {err}"}

        if last_tool_name == "final_answer":
            message = str(output.get("message") or "done")
            return {"kind": "final_answer", "message": message}

        # If we're not using OpenAI tool-calling, keep legacy behavior: one tool
        # call then summarize.
        if not use_openai:
            return {
                "kind": "final_answer",
                "message": self._summarize_tool_output(output),
            }

        wants_report = bool(
            re.search(r"\b(analyze|analysis|report|summarize)\b", lowered)
        )
        if wants_report and len(tool_steps) == 1 and "store_memory" in allowed_tools:
            summary = self._summarize_tool_output(output)
            return {
                "kind": "tool_call",
                "tool_name": "store_memory",
                "args": {
                    "collection": "runs",
                    "text": summary,
                    "metadata": {"source": "analysis"},
                },
            }

        # Heuristic: if we just wrote a file in the workspace, consider the task done.
        if last_tool_name == "write_file":
            path = str(output.get("path") or "")
            return {
                "kind": "final_answer",
                "message": f"Wrote file: {path}",
            }

        # Heuristic: if a shell command obviously wrote the requested poem, stop.
        if last_tool_name == "run_shell":
            command = str(output.get("command") or "")
            if "poem.txt" in command:
                return {
                    "kind": "final_answer",
                    "message": "Created poem.txt via run_shell.",
                }

        # Otherwise, ask the model for the next tool call.
        summarized = self._summarize_tool_output(output)
        followup_task = (
            f"Task: {task.strip()}\n\n"
            f"Last tool: {last_tool_name}\n"
            f"Last tool result summary:\n{summarized}\n\n"
            "Decide the next tool call to complete the task, or call final_answer when complete."
        )
        inference = self.model_gateway.infer(
            task=followup_task,
            agent=agent,
            context=context,
        )
        if not inference.get("ok"):
            message = inference.get("error", {}).get("message", "model inference failed")
            return {
                "kind": "final_answer",
                "message": f"I could not infer the next tool action: {message}",
            }
        return {
            "kind": "tool_call",
            "tool_name": str(inference.get("tool_name")),
            "args": dict(inference.get("args") or {}),
        }

    @staticmethod
    def _extract_url(task: str) -> str | None:
        match = re.search(r"https?://[^\s'\"]+", task or "")
        if not match:
            return None
        return match.group(0)

    @staticmethod
    def _summarize_tool_output(output: dict[str, Any]) -> str:
        stdout = str(output.get("stdout") or "").strip()
        stderr = str(output.get("stderr") or "").strip()
        exit_code = output.get("exit_code")
        body = stdout if stdout else stderr
        if len(body) > 800:
            body = body[:800] + "..."
        return f"Tool completed with exit_code={exit_code}.\n\n{body}".strip()
