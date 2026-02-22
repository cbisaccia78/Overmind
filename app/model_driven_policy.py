"""Model-driven policy implementation.

This policy delegates intent parsing and next-step control to the model. It does
not apply task-specific keyword contracts inside the execution loop.
"""

from __future__ import annotations

from typing import Any

from .model_gateway import ModelGateway
from .policy import PlanContext, PlannedAction, Policy


class ModelDrivenPolicy(Policy):
    """Policy that lets the model choose the next action each turn."""

    def __init__(self, model_gateway: ModelGateway):
        """Create a policy instance.

        Args:
            model_gateway: Model inference gateway used for action selection.
        """
        self.model_gateway = model_gateway

    def plan(
        self,
        task: str,
        *,
        agent: dict[str, Any],
        context: PlanContext,
    ) -> list[PlannedAction]:
        """Build a simple plan for legacy orchestrator mode."""
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
            actions.append(
                PlannedAction(
                    step_type="tool",
                    tool_name=str(inference.get("tool_name")),
                    args=dict(inference.get("args") or {}),
                )
            )

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
        """Return the next action for iterative execution.

        Strategy:
        - Always ask the model to choose the next step.
        - On tool failure, ask the model to recover using failure context.
        - Do not use task keyword contracts or phase heuristics.
        """
        if not history:
            prompt = self._build_initial_prompt(task=task)
            return self._infer_next_action(prompt=prompt, agent=agent, context=context)

        last_step = history[-1]
        last_step_type = str(last_step.get("step_type") or "")

        if last_step_type == "tool":
            tool_input = dict(last_step.get("input") or {})
            output = dict(last_step.get("output") or {})
            last_tool_name = str(tool_input.get("tool_name") or "")
            last_args = dict(tool_input.get("args") or {})

            if not output.get("ok"):
                error_message = str(
                    (output.get("error") or {}).get("message") or "tool failed"
                )
                prompt = self._build_failure_replan_prompt(
                    task=task,
                    last_tool_name=last_tool_name,
                    last_args=last_args,
                    error_message=error_message,
                    failure_memory=self._collect_failure_memory(history),
                )
                return self._infer_next_action(
                    prompt=prompt,
                    agent=agent,
                    context=context,
                )

            prompt = self._build_followup_prompt(
                task=task,
                last_tool_name=last_tool_name,
                last_tool_summary=self._summarize_tool_output(output),
                history=history,
            )
            return self._infer_next_action(prompt=prompt, agent=agent, context=context)

        prompt = self._build_initial_prompt(task=task, history=history)
        return self._infer_next_action(prompt=prompt, agent=agent, context=context)

    def _infer_next_action(
        self,
        *,
        prompt: str,
        agent: dict[str, Any],
        context: PlanContext,
    ) -> dict[str, Any]:
        inference = self.model_gateway.infer(task=prompt, agent=agent, context=context)
        if not inference.get("ok"):
            message = inference.get("error", {}).get(
                "message", "model inference failed"
            )
            if self._should_ask_user_on_inference_error(message):
                return {
                    "kind": "ask_user",
                    "message": (
                        "I could not determine the next tool action from the model "
                        "response. Please provide a specific next step."
                    ),
                }
            return {
                "kind": "final_answer",
                "message": f"I could not infer the next tool action: {message}",
            }
        return self._action_from_inference(inference)

    @staticmethod
    def _build_initial_prompt(
        *, task: str, history: list[dict[str, Any]] | None = None
    ) -> str:
        recent = ModelDrivenPolicy._render_recent_history(history or [])
        if recent:
            return (
                f"Task: {task.strip()}\n\n"
                "Recent run history:\n"
                f"{recent}\n\n"
                "Decide the next action. Choose exactly one of: tool call, ask_user, or final_answer."
            )
        return (
            f"Task: {task.strip()}\n\n"
            "Decide the next action. Choose exactly one of: tool call, ask_user, or final_answer."
        )

    @staticmethod
    def _build_followup_prompt(
        *,
        task: str,
        last_tool_name: str,
        last_tool_summary: str,
        history: list[dict[str, Any]],
    ) -> str:
        recent = ModelDrivenPolicy._render_recent_history(history)
        return (
            f"Task: {task.strip()}\n\n"
            f"Last tool: {last_tool_name or 'none'}\n"
            "Last tool result summary:\n"
            f"{last_tool_summary or 'none'}\n\n"
            "Recent run history:\n"
            f"{recent or 'none'}\n\n"
            "Decide the next action. Choose exactly one of: tool call, ask_user, or final_answer."
        )

    @staticmethod
    def _should_ask_user_on_inference_error(message: str) -> bool:
        lowered = str(message or "").lower()
        return "openai response missing valid tool call" in lowered

    @staticmethod
    def _action_from_inference(inference: dict[str, Any]) -> dict[str, Any]:
        tool_name = str(inference.get("tool_name") or "")
        args = dict(inference.get("args") or {})

        if tool_name == "ask_user":
            message = str(
                args.get("message")
                or args.get("prompt")
                or "Additional input is required."
            )
            return {"kind": "ask_user", "message": message}

        if tool_name == "final_answer":
            message = str(args.get("message") or "done")
            return {"kind": "final_answer", "message": message}

        return {
            "kind": "tool_call",
            "tool_name": tool_name,
            "args": args,
        }

    @staticmethod
    def _summarize_tool_output(output: dict[str, Any]) -> str:
        stdout = str(output.get("stdout") or "").strip()
        stderr = str(output.get("stderr") or "").strip()
        exit_code = output.get("exit_code")
        body = stdout if stdout else stderr
        if not body:
            body = ModelDrivenPolicy._extract_mcp_text(output)
        if not body:
            observation = output.get("observation")
            if isinstance(observation, dict):
                body = str(observation.get("text") or "").strip()
        if len(body) > 800:
            body = body[:800] + "..."
        return f"Tool completed with exit_code={exit_code}.\n\n{body}".strip()

    @staticmethod
    def _extract_mcp_text(output: dict[str, Any]) -> str:
        mcp_payload = output.get("mcp")
        if not isinstance(mcp_payload, dict):
            return ""
        result = mcp_payload.get("result")
        if not isinstance(result, dict):
            return ""
        content = result.get("content")
        if not isinstance(content, list):
            return ""
        texts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    texts.append(text)
                continue
            if isinstance(item, dict):
                text = str(item.get("text") or "").strip()
                if text:
                    texts.append(text)
        return "\n".join(texts).strip()

    @staticmethod
    def _collect_failure_memory(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        failures: list[dict[str, Any]] = []
        for item in history:
            if item.get("step_type") != "tool":
                continue
            output = dict(item.get("output") or {})
            if output.get("ok", True):
                continue
            tool_input = dict(item.get("input") or {})
            failures.append(
                {
                    "tool_name": str(tool_input.get("tool_name") or ""),
                    "args": dict(tool_input.get("args") or {}),
                    "error": output.get("error") or item.get("error"),
                }
            )
        return failures[-5:]

    @staticmethod
    def _build_failure_replan_prompt(
        *,
        task: str,
        last_tool_name: str,
        last_args: dict[str, Any],
        error_message: str,
        failure_memory: list[dict[str, Any]],
    ) -> str:
        return (
            f"Task: {task.strip()}\n\n"
            "Prior failures:\n"
            f"{failure_memory}\n\n"
            "Most recent failure:\n"
            f"tool={last_tool_name} args={last_args} error={error_message}\n\n"
            "Choose the next action. Prefer a different approach that can still complete the task. "
            "Choose exactly one of: tool call, ask_user, or final_answer."
        )

    @staticmethod
    def _render_recent_history(history: list[dict[str, Any]], limit: int = 6) -> str:
        lines: list[str] = []
        for item in history[-limit:]:
            step_type = str(item.get("step_type") or "")
            if step_type == "tool":
                tool_input = dict(item.get("input") or {})
                output = dict(item.get("output") or {})
                tool_name = str(tool_input.get("tool_name") or "")
                status = "ok" if output.get("ok") else "failed"
                lines.append(f"tool {tool_name} -> {status}")
                continue
            if step_type == "ask_user":
                lines.append("ask_user")
                continue
            if step_type == "verify":
                output = dict(item.get("output") or {})
                status = "ok" if output.get("ok") else "failed"
                lines.append(f"verify -> {status}")
                continue
            if step_type == "eval":
                lines.append("eval")
        return "\n".join(lines).strip()
