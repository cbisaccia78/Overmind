"""Model-driven policy implementation.

This policy delegates intent parsing and next-step control to the model. It does
not apply task-specific keyword contracts inside the execution loop.
"""

from __future__ import annotations

import json
import re
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

            stall = self._detect_stall_repetition(history)
            if stall is not None:
                prompt = self._build_stall_recovery_prompt(
                    task=task,
                    blocked_tool_names=list(stall.get("blocked_tool_names") or []),
                    repeats=int(stall.get("repeats") or 0),
                    pattern_summary=str(stall.get("pattern_summary") or ""),
                    last_tool_name=last_tool_name,
                    last_tool_summary=self._summarize_tool_output(output),
                    history=history,
                )
                action = self._infer_next_action(
                    prompt=prompt,
                    agent=agent,
                    context=context,
                )
                blocked_tools = set(stall.get("blocked_tool_names") or [])
                if self._violates_stall_constraint(action, blocked_tools):
                    violation_prompt = self._build_stall_violation_prompt(
                        task=task,
                        blocked_tool_names=sorted(blocked_tools),
                        attempted_action=action,
                        history=history,
                    )
                    return self._infer_next_action(
                        prompt=violation_prompt,
                        agent=agent,
                        context=context,
                    )
                return action

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
        task_block = ModelDrivenPolicy._render_task_context(task)
        recent = ModelDrivenPolicy._render_recent_history(history or [])
        contract = ModelDrivenPolicy._decision_contract()
        if recent:
            return (
                f"{task_block}\n\n"
                "Recent run history:\n"
                f"{recent}\n\n"
                f"{contract}"
            )
        return (
            f"{task_block}\n\n"
            f"{contract}"
        )

    @staticmethod
    def _build_followup_prompt(
        *,
        task: str,
        last_tool_name: str,
        last_tool_summary: str,
        history: list[dict[str, Any]],
    ) -> str:
        task_block = ModelDrivenPolicy._render_task_context(task)
        recent = ModelDrivenPolicy._render_recent_history(history)
        contract = ModelDrivenPolicy._decision_contract()
        return (
            f"{task_block}\n\n"
            f"Last tool: {last_tool_name or 'none'}\n"
            "Last tool result summary:\n"
            f"{last_tool_summary or 'none'}\n\n"
            "Recent run history:\n"
            f"{recent or 'none'}\n\n"
            f"{contract}"
        )

    @staticmethod
    def _build_stall_recovery_prompt(
        *,
        task: str,
        blocked_tool_names: list[str],
        repeats: int,
        pattern_summary: str,
        last_tool_name: str,
        last_tool_summary: str,
        history: list[dict[str, Any]],
    ) -> str:
        task_block = ModelDrivenPolicy._render_task_context(task)
        recent = ModelDrivenPolicy._render_recent_history(history)
        contract = ModelDrivenPolicy._decision_contract()
        blocked = ", ".join(f"`{name}`" for name in blocked_tool_names if name) or "none"
        return (
            f"{task_block}\n\n"
            "Stall detected:\n"
            f"- Pattern observed across the last {repeats} successful tool actions.\n"
            f"- Pattern summary: {pattern_summary or 'repeated low-progress actions'}\n\n"
            "Stall recovery constraint (next turn only):\n"
            f"- Do not call any of these tools in your next action: {blocked}.\n"
            "- Choose a different tool/arguments, or ask_user/final_answer if no safe progress is possible.\n\n"
            f"Last tool: {last_tool_name or 'none'}\n"
            "Last tool result summary:\n"
            f"{last_tool_summary or 'none'}\n\n"
            "Recent run history:\n"
            f"{recent or 'none'}\n\n"
            f"{contract}"
        )

    @staticmethod
    def _build_stall_violation_prompt(
        *,
        task: str,
        blocked_tool_names: list[str],
        attempted_action: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> str:
        task_block = ModelDrivenPolicy._render_task_context(task)
        recent = ModelDrivenPolicy._render_recent_history(history)
        contract = ModelDrivenPolicy._decision_contract()
        attempted_kind = str(attempted_action.get("kind") or "")
        attempted_tool = str(attempted_action.get("tool_name") or "")
        attempted_args = ModelDrivenPolicy._compact_args(
            dict(attempted_action.get("args") or {})
        )
        blocked = ", ".join(f"`{name}`" for name in blocked_tool_names if name) or "none"
        return (
            f"{task_block}\n\n"
            "Stall recovery violation:\n"
            "Your previous choice violated the stall recovery constraint.\n"
            f"Attempted action: kind={attempted_kind} tool={attempted_tool} args={attempted_args or '{}'}\n\n"
            "Constraint (this turn):\n"
            f"- Do not call any of these tools: {blocked}.\n"
            "- Choose a different tool/arguments now, or ask_user/final_answer if no safe progress is possible.\n\n"
            "Recent run history:\n"
            f"{recent or 'none'}\n\n"
            f"{contract}"
        )

    @staticmethod
    def _violates_stall_constraint(
        action: dict[str, Any], blocked_tool_names: set[str]
    ) -> bool:
        if str(action.get("kind") or "") != "tool_call":
            return False
        tool_name = str(action.get("tool_name") or "")
        return bool(tool_name and tool_name in blocked_tool_names)

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
    def _detect_stall_repetition(
        history: list[dict[str, Any]], min_repeats: int = 3
    ) -> dict[str, Any] | None:
        """Detect repeated successful tool patterns at the history tail."""
        trailing: list[dict[str, Any]] = []
        for item in reversed(history):
            if str(item.get("step_type") or "") != "tool":
                break
            output = dict(item.get("output") or {})
            if not output.get("ok"):
                break
            trailing.append(item)
        trailing.reverse()
        if len(trailing) < min_repeats:
            return None

        # Case 1: identical tool name repeated, even if args vary.
        last_tool = str(dict(trailing[-1].get("input") or {}).get("tool_name") or "")
        if last_tool:
            same_tool_count = 0
            for item in reversed(trailing):
                tool_name = str(dict(item.get("input") or {}).get("tool_name") or "")
                if tool_name != last_tool:
                    break
                same_tool_count += 1
            if same_tool_count >= min_repeats:
                return {
                    "blocked_tool_names": [last_tool],
                    "repeats": same_tool_count,
                    "pattern_summary": (
                        f"tool `{last_tool}` repeated {same_tool_count} times in a row"
                    ),
                }

        # Case 2: two-tool loop near the tail (e.g., snapshot/navigate oscillation).
        window = trailing[-6:] if len(trailing) >= 6 else trailing
        tool_names = [
            str(dict(item.get("input") or {}).get("tool_name") or "") for item in window
        ]
        unique = sorted({name for name in tool_names if name})
        if len(unique) == 2 and len(tool_names) >= 6:
            counts = {name: tool_names.count(name) for name in unique}
            if all(count >= 2 for count in counts.values()):
                counts_text = ", ".join(f"`{name}` x{counts[name]}" for name in unique)
                return {
                    "blocked_tool_names": unique,
                    "repeats": len(tool_names),
                    "pattern_summary": f"two-tool loop in recent actions ({counts_text})",
                }

        return None

    @staticmethod
    def _build_failure_replan_prompt(
        *,
        task: str,
        last_tool_name: str,
        last_args: dict[str, Any],
        error_message: str,
        failure_memory: list[dict[str, Any]],
    ) -> str:
        task_block = ModelDrivenPolicy._render_task_context(task)
        contract = ModelDrivenPolicy._decision_contract()
        return (
            f"{task_block}\n\n"
            "Prior failures:\n"
            f"{failure_memory}\n\n"
            "Most recent failure:\n"
            f"tool={last_tool_name} args={last_args} error={error_message}\n\n"
            "Choose the next action. Prefer a different approach that can still complete the task.\n\n"
            f"{contract}"
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
                args = dict(tool_input.get("args") or {})
                status = "ok" if output.get("ok") else "failed"
                args_text = ModelDrivenPolicy._compact_args(args)
                if args_text:
                    lines.append(f"tool {tool_name} args={args_text} -> {status}")
                else:
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

    @staticmethod
    def _render_task_context(task: str) -> str:
        original_task, user_inputs = ModelDrivenPolicy._split_task_and_user_inputs(task)
        if not user_inputs:
            return f"Task: {original_task}"

        latest_input = user_inputs[-1]
        older_inputs = user_inputs[:-1]
        sections = [f"Original task:\n{original_task}"]
        if older_inputs:
            recent_older = older_inputs[-2:]
            bullets = "\n".join(f"- {item}" for item in recent_older)
            sections.append(f"Prior user inputs:\n{bullets}")
        sections.append(f"Latest user input (highest priority):\n{latest_input}")
        return "Task context:\n" + "\n\n".join(sections)

    @staticmethod
    def _decision_contract() -> str:
        return (
            "Decide the next action. Choose exactly one of: tool call, ask_user, or final_answer.\n"
            "Prioritize progress toward the latest user input.\n"
            "Avoid repeating the same successful action with identical arguments unless it is necessary."
        )

    @staticmethod
    def _split_task_and_user_inputs(task: str) -> tuple[str, list[str]]:
        raw = str(task or "").strip()
        if not raw:
            return "", []

        markers = list(re.finditer(r"(?:^|\n\n)User input:\s*", raw, flags=re.IGNORECASE))
        if not markers:
            return raw, []

        base_end = markers[0].start()
        base_task = raw[:base_end].strip()
        messages: list[str] = []
        for idx, marker in enumerate(markers):
            start = marker.end()
            end = markers[idx + 1].start() if idx + 1 < len(markers) else len(raw)
            message = raw[start:end].strip()
            if message:
                messages.append(message)
        if not base_task:
            base_task = messages[0] if messages else raw
        return base_task, messages

    @staticmethod
    def _compact_args(args: dict[str, Any], max_len: int = 140) -> str:
        if not args:
            return ""
        rendered = json.dumps(args, sort_keys=True, separators=(",", ":"))
        if len(rendered) <= max_len:
            return rendered
        return rendered[:max_len] + "..."
