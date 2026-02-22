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
        planning_contract = dict(
            context.get("planning_contract") or self._build_task_contract(task)
        )
        progress_state = dict(
            context.get("progress_state")
            or self._derive_progress_state(history, planning_contract)
        )

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

            if (
                planning_contract.get("mode")
                in {
                    "multi_page_research",
                    "web_interaction_feedback",
                }
                and use_openai
            ):
                prompt = self._build_contract_prompt(
                    task=task,
                    planning_contract=planning_contract,
                    progress_state=progress_state,
                    last_tool_name="",
                    last_tool_summary="",
                )
                inference = self.model_gateway.infer(
                    task=prompt,
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
                return self._action_from_inference(inference)

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
            return self._action_from_inference(inference)

        last_tool = tool_steps[-1]
        last_input = dict(last_tool.get("input") or {})
        last_tool_name = str(last_input.get("tool_name") or "")
        last_args = dict(last_input.get("args") or {})
        output = dict(last_tool.get("output") or {})
        if not output.get("ok"):
            err = output.get("error", {}).get("message", "tool failed")
            failure_memory = list(context.get("failure_memory") or [])
            if not failure_memory:
                failure_memory = self._collect_failure_memory(history)

            if not use_openai:
                return {"kind": "final_answer", "message": f"Tool failed: {err}"}

            if len(failure_memory) >= 2:
                return {
                    "kind": "ask_user",
                    "message": "I hit repeated tool failures. Please clarify the expected output and constraints before I continue.",
                }

            prompt = self._build_failure_replan_prompt(
                task=task,
                last_tool_name=last_tool_name,
                last_args=last_args,
                error_message=err,
                failure_memory=failure_memory,
            )
            inference = self.model_gateway.infer(
                task=prompt,
                agent=agent,
                context=context,
            )
            if not inference.get("ok"):
                message = inference.get("error", {}).get(
                    "message", "model inference failed"
                )
                return {
                    "kind": "final_answer",
                    "message": f"I could not infer the next tool action: {message}",
                }

            candidate_tool = str(inference.get("tool_name") or "")
            candidate_args = dict(inference.get("args") or {})
            if self._is_repeated_failure(
                tool_name=candidate_tool,
                args=candidate_args,
                failure_memory=failure_memory,
            ):
                return {
                    "kind": "ask_user",
                    "message": "The next planned action repeats a previously failed tool call. Please provide additional guidance.",
                }

            return self._action_from_inference(
                {"tool_name": candidate_tool, "args": candidate_args}
            )

        if last_tool_name == "ask_user":
            message = str(output.get("message") or "Additional input is required.")
            return {"kind": "ask_user", "message": message}

        if last_tool_name == "final_answer":
            message = str(output.get("message") or "done")
            return {"kind": "final_answer", "message": message}

        if (
            planning_contract.get("mode")
            in {
                "multi_page_research",
                "web_interaction_feedback",
            }
            and use_openai
        ):
            if bool(progress_state.get("contract_satisfied")):
                return {
                    "kind": "final_answer",
                    "message": self._contract_completion_message(progress_state),
                }

            summarized = self._summarize_tool_output(output)
            prompt = self._build_contract_prompt(
                task=task,
                planning_contract=planning_contract,
                progress_state=progress_state,
                last_tool_name=last_tool_name,
                last_tool_summary=summarized,
            )
            inference = self.model_gateway.infer(
                task=prompt,
                agent=agent,
                context=context,
            )
            if not inference.get("ok"):
                message = inference.get("error", {}).get(
                    "message", "model inference failed"
                )
                if self._should_ask_user_on_inference_error(message):
                    return {
                        "kind": "ask_user",
                        "message": (
                            "I could not determine the next tool action from the "
                            "model response. Please provide a specific next action "
                            "to take on the current page."
                        ),
                    }
                return {
                    "kind": "final_answer",
                    "message": f"I could not infer the next tool action: {message}",
                }

            candidate_tool = str(inference.get("tool_name") or "")
            candidate_args = dict(inference.get("args") or {})
            if self._is_repeated_success_candidate(
                candidate_tool=candidate_tool,
                candidate_args=candidate_args,
                last_tool_name=last_tool_name,
                last_args=last_args,
                progress_state=progress_state,
            ):
                return {
                    "kind": "ask_user",
                    "message": "I am repeatedly selecting the same successful action without progress. Please provide a specific page/topic to continue.",
                }
            return self._action_from_inference(
                {"tool_name": candidate_tool, "args": candidate_args}
            )

        verification = self._build_verification_action(
            task=task,
            last_tool_name=last_tool_name,
            output=output,
            history=history,
        )
        if verification is not None:
            return verification

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

        if self._is_repeated_success_loop(history, min_repeats=5):
            return {
                "kind": "ask_user",
                "message": "I am repeating the same successful action without making progress. Please clarify what should happen next.",
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
            "Decide the next action to complete the task. "
            "Choose one of: tool call, ask_user when user input is required, "
            "or final_answer when complete."
        )
        inference = self.model_gateway.infer(
            task=followup_task,
            agent=agent,
            context=context,
        )
        if not inference.get("ok"):
            message = inference.get("error", {}).get(
                "message", "model inference failed"
            )
            if self._should_ask_user_on_inference_error(message):
                return {
                    "kind": "ask_user",
                    "message": (
                        "I could not determine the next tool action from the "
                        "model response. Please provide a specific next step."
                    ),
                }
            return {
                "kind": "final_answer",
                "message": f"I could not infer the next tool action: {message}",
            }
        return self._action_from_inference(inference)

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

    def _build_verification_action(
        self,
        *,
        task: str,
        last_tool_name: str,
        output: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Return a verifier action when tasks produce screenshot artifacts."""
        if any(item.get("step_type") == "verify" for item in history):
            return None

        lowered_task = (task or "").lower()
        screenshot_hint = bool(
            re.search(r"\b(screenshot|png|image|capture)\b", lowered_task)
        )
        tool_hint = "screenshot" in (last_tool_name or "").lower()
        if not screenshot_hint and not tool_hint:
            return None

        raw_path = str(output.get("path") or output.get("filename") or "").strip()
        if not raw_path:
            return None

        checks: list[dict[str, Any]] = [{"type": "file_exists", "path": raw_path}]
        if raw_path.lower().endswith(".png"):
            checks.extend(
                [
                    {"type": "file_min_bytes", "path": raw_path, "min_bytes": 8000},
                    {"type": "png_signature", "path": raw_path},
                    {
                        "type": "png_dimensions_min",
                        "path": raw_path,
                        "min_width": 600,
                        "min_height": 400,
                    },
                    {
                        "type": "file_entropy_min",
                        "path": raw_path,
                        "min_entropy": 3.5,
                    },
                ]
            )

        return {
            "kind": "verify",
            "checks": checks,
            "on_fail_prompt": "The screenshot artifact failed verification. Please provide a valid output path and retry.",
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
            "Choose the next tool call that avoids repeating failed tool+args combinations."
        )

    @staticmethod
    def _is_repeated_failure(
        *, tool_name: str, args: dict[str, Any], failure_memory: list[dict[str, Any]]
    ) -> bool:
        for failure in failure_memory:
            if str(failure.get("tool_name") or "") != tool_name:
                continue
            if dict(failure.get("args") or {}) == args:
                return True
        return False

    @staticmethod
    def _build_task_contract(task: str) -> dict[str, Any]:
        lowered = (task or "").lower()
        is_browser_research = all(
            token in lowered
            for token in ["wikipedia", "navigate", "screenshot", "summar"]
        )
        if is_browser_research:
            return {
                "mode": "multi_page_research",
                "domain": "wikipedia.org",
                "min_unique_pages": 3,
                "require_screenshot_per_page": True,
                "require_summary_per_page": True,
            }

        browser_tokens = [
            "browser",
            "dom",
            "page",
            "website",
            "site",
            "x.com",
            "twitter",
            "interact",
            "click",
            "type",
            "content",
        ]
        goal_tokens = [
            "feedback",
            "engage",
            "engagement",
            "respond",
            "reply",
            "comment",
            "like",
            "quote",
            "post",
            "thread",
        ]
        is_web_interaction_goal = any(
            token in lowered for token in browser_tokens
        ) and any(token in lowered for token in goal_tokens)
        if is_web_interaction_goal:
            return {
                "mode": "web_interaction_feedback",
                "require_dom_exploration": True,
                "require_interaction": True,
                "require_feedback_signal": True,
                "min_exploration_steps": 1,
                "min_interaction_steps": 1,
                "min_feedback_signals": 1,
            }

        return {"mode": "generic"}

    @staticmethod
    def _derive_progress_state(
        history: list[dict[str, Any]], planning_contract: dict[str, Any]
    ) -> dict[str, Any]:
        visited_urls: list[str] = []
        screenshot_count = 0
        summary_count = 0
        exploration_count = 0
        interaction_count = 0
        feedback_signal_count = 0
        consecutive_same_success = 0
        for item in history:
            if item.get("step_type") != "tool":
                continue
            output = dict(item.get("output") or {})
            if not output.get("ok"):
                continue
            tool_input = dict(item.get("input") or {})
            tool_name = str(tool_input.get("tool_name") or "")
            args = dict(tool_input.get("args") or {})
            if tool_name.endswith("browser_navigate"):
                url = str(args.get("url") or "").strip()
                if url:
                    visited_urls.append(url)
            if any(
                marker in tool_name
                for marker in [
                    "snapshot",
                    "evaluate",
                    "network_requests",
                    "console_messages",
                ]
            ):
                exploration_count += 1
            if any(
                marker in tool_name
                for marker in [
                    "_click",
                    "_type",
                    "fill_form",
                    "select_option",
                    "press_key",
                    "_drag",
                ]
            ):
                interaction_count += 1
            if any(
                marker in tool_name
                for marker in [
                    "wait_for",
                    "network_requests",
                    "console_messages",
                    "read_notebook_cell_output",
                ]
            ):
                feedback_signal_count += 1
            if "screenshot" in tool_name:
                screenshot_count += 1
            if tool_name == "store_memory":
                source = str(dict(args.get("metadata") or {}).get("source") or "")
                if source in {"analysis", "summary"}:
                    summary_count += 1

        tail_signature: str | None = None
        for item in reversed(history):
            if item.get("step_type") != "tool":
                break
            output = dict(item.get("output") or {})
            if not output.get("ok"):
                break
            tool_input = dict(item.get("input") or {})
            signature = (
                f"{str(tool_input.get('tool_name') or '')}|"
                f"{repr(sorted(dict(tool_input.get('args') or {}).items()))}"
            )
            if tail_signature is None:
                tail_signature = signature
                consecutive_same_success = 1
                continue
            if signature != tail_signature:
                break
            consecutive_same_success += 1

        unique_urls = sorted(set(visited_urls))
        min_unique = int(planning_contract.get("min_unique_pages") or 0)
        mode = str(planning_contract.get("mode") or "generic")
        if mode == "multi_page_research":
            contract_satisfied = (
                len(unique_urls) >= min_unique
                and screenshot_count >= max(min_unique, 1)
                and summary_count >= max(min_unique, 1)
            )
        elif mode == "web_interaction_feedback":
            contract_satisfied = (
                exploration_count
                >= int(planning_contract.get("min_exploration_steps") or 1)
                and interaction_count
                >= int(planning_contract.get("min_interaction_steps") or 1)
                and feedback_signal_count
                >= int(planning_contract.get("min_feedback_signals") or 1)
            )
        else:
            contract_satisfied = False

        return {
            "unique_urls": unique_urls,
            "unique_pages_done": len(unique_urls),
            "screenshot_count": screenshot_count,
            "summary_count": summary_count,
            "exploration_count": exploration_count,
            "interaction_count": interaction_count,
            "feedback_signal_count": feedback_signal_count,
            "consecutive_same_success": consecutive_same_success,
            "contract_satisfied": contract_satisfied,
        }

    @staticmethod
    def _build_contract_prompt(
        *,
        task: str,
        planning_contract: dict[str, Any],
        progress_state: dict[str, Any],
        last_tool_name: str,
        last_tool_summary: str,
    ) -> str:
        mode = str(planning_contract.get("mode") or "generic")
        if mode == "web_interaction_feedback":
            strategy_hint = (
                "Follow phased execution: (1) explore DOM/state with snapshot/evaluate, "
                "(2) perform one concrete interaction aligned to the goal, "
                "(3) verify user feedback via wait/network/console/snapshot evidence, "
                "then iterate unmet phases."
            )
        else:
            strategy_hint = (
                "Choose ONE next tool call that advances unmet objectives. "
                "Prefer visiting a NEW article URL when unique page count is below target. "
                "Do not repeat the same successful tool+args when progress did not increase. "
                "If all objectives are satisfied, use final_answer."
            )
        return (
            f"Task: {task.strip()}\n\n"
            "Planning contract:\n"
            f"{planning_contract}\n\n"
            "Current progress:\n"
            f"{progress_state}\n\n"
            f"Last successful tool: {last_tool_name or 'none'}\n"
            f"Last tool summary: {last_tool_summary or 'none'}\n\n"
            f"{strategy_hint}"
        )

    @staticmethod
    def _contract_completion_message(progress_state: dict[str, Any]) -> str:
        if int(progress_state.get("interaction_count") or 0) > 0:
            return (
                "Completed the interaction strategy. "
                f"Exploration steps: {int(progress_state.get('exploration_count') or 0)}, "
                f"interactions: {int(progress_state.get('interaction_count') or 0)}, "
                f"feedback signals: {int(progress_state.get('feedback_signal_count') or 0)}."
            )
        pages = int(progress_state.get("unique_pages_done") or 0)
        screenshots = int(progress_state.get("screenshot_count") or 0)
        summaries = int(progress_state.get("summary_count") or 0)
        return (
            "Completed the browsing contract. "
            f"Visited {pages} unique pages, captured {screenshots} screenshots, "
            f"and recorded {summaries} summaries."
        )

    @staticmethod
    def _is_repeated_success_candidate(
        *,
        candidate_tool: str,
        candidate_args: dict[str, Any],
        last_tool_name: str,
        last_args: dict[str, Any],
        progress_state: dict[str, Any],
    ) -> bool:
        if candidate_tool != last_tool_name:
            return False
        if candidate_args != last_args:
            return False
        return int(progress_state.get("consecutive_same_success") or 0) >= 5

    @staticmethod
    def _is_repeated_success_loop(
        history: list[dict[str, Any]], min_repeats: int = 5
    ) -> bool:
        signatures: list[str] = []
        for item in reversed(history):
            if item.get("step_type") != "tool":
                break
            output = dict(item.get("output") or {})
            if not output.get("ok"):
                break
            tool_input = dict(item.get("input") or {})
            tool_name = str(tool_input.get("tool_name") or "")
            if tool_name in {"ask_user", "final_answer"}:
                break
            args = dict(tool_input.get("args") or {})
            signatures.append(f"{tool_name}|{repr(sorted(args.items()))}")

        if len(signatures) < min_repeats:
            return False
        window = signatures[:min_repeats]
        return len(set(window)) == 1
