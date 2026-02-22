"""Deterministic orchestrator for executing runs.

Plans actions via a policy, executes tools via the gateway, and records
steps/events with retry and step-limit handling.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from .policy import Policy
from .repository import Repository
from .tool_gateway import ToolGateway


@dataclass
class RetryConfig:
    """Retry policy configuration for tool calls.

    Attributes:
        max_attempts: Maximum attempts per tool call.
        backoff_seconds: Sleep duration between attempts.
    """

    max_attempts: int = 2
    backoff_seconds: float = 0.1


class Orchestrator:
    """Execute runs deterministically and persist steps, tool calls, and events."""

    def __init__(
        self,
        repo: Repository,
        tool_gateway: ToolGateway,
        policy: Policy,
        retry_config: RetryConfig | None = None,
    ):
        """Create an orchestrator instance.

        Args:
            repo: Persistence layer used to read/write run state.
            tool_gateway: Gateway used to execute allowed tools.
            policy: Policy that expands a task into planned actions.
            retry_config: Retry policy for tool failures.
        """
        self.repo = repo
        self.tool_gateway = tool_gateway
        self.policy = policy
        self.retry_config = retry_config or RetryConfig()

    def launch(self, run_id: str) -> None:
        """Launch processing for a run in a daemon background thread.

        Args:
            run_id: Run ID to process.

        Returns:
            None.
        """
        thread = threading.Thread(target=self.process_run, args=(run_id,), daemon=True)
        thread.start()

    def process_run(self, run_id: str) -> None:
        """Process a queued run end-to-end.

        Loads the run and agent, plans actions using the configured policy,
        executes tools through the gateway with retries, and persists steps and
        events.

        Args:
            run_id: Run ID to process.

        Returns:
            None.
        """
        if callable(getattr(type(self.policy), "next_action", None)):
            self._process_run_iterative(run_id)
            return
        self._process_run_legacy(run_id)

    def _process_run_iterative(self, run_id: str) -> None:
        """Process a run via iterative next-action planning."""
        run = self.repo.get_run(run_id)
        if not run:
            return
        if run["status"] == "canceled":
            return

        agent = self.repo.get_agent(run["agent_id"])
        if not agent:
            self.repo.update_run_status(run_id, "failed")
            self.repo.create_event(run_id, "run.failed", {"error": "agent_not_found"})
            return

        if agent.get("status") != "active":
            self.repo.update_run_status(run_id, "failed")
            self.repo.create_event(run_id, "run.failed", {"error": "agent_disabled"})
            return

        step_limit = int(run["step_limit"])
        self.repo.update_run_status(run_id, "running")
        if not run.get("started_at"):
            self.repo.create_event(
                run_id, "run.started", {"run_id": run_id, "agent_id": agent["id"]}
            )

        history: list[dict[str, Any]] = []
        existing_steps = self.repo.list_steps(run_id)
        for step in existing_steps:
            history.append(
                {
                    "step_type": step.get("type"),
                    "input": step.get("input_json") or {},
                    "output": step.get("output_json") or {},
                    "error": step.get("error"),
                }
            )

        idx = len(existing_steps)
        if idx == 0:
            step = self.repo.create_step(
                run_id=run_id,
                idx=idx,
                step_type="plan",
                input_json={
                    "task": run["task"],
                    "agent_id": agent.get("id"),
                    "run_id": run_id,
                },
            )
            self.repo.finish_step(
                step["id"], output_json={"ok": True, "note": "planned"}
            )
            self.repo.create_event(
                run_id,
                "step.finished",
                {"step_id": step["id"], "idx": idx, "ok": True},
            )
            history.append(
                {
                    "step_type": "plan",
                    "input": {"task": run["task"], "agent_id": agent.get("id")},
                    "output": {"ok": True, "note": "planned"},
                    "error": None,
                }
            )
            idx += 1

        while idx < step_limit:
            latest = self.repo.get_run(run_id)
            if latest and latest["status"] == "canceled":
                self.repo.create_event(run_id, "run.canceled", {"run_id": run_id})
                return

            action = self.policy.next_action(
                run["task"],
                agent=agent,
                context={
                    "run_id": run_id,
                    "step_limit": step_limit,
                    "history": history,
                },
                history=history,
            )
            kind = str(action.get("kind") or "")

            if kind == "tool_call":
                tool_name = str(action.get("tool_name") or "")
                args = dict(action.get("args") or {})
                step = self.repo.create_step(
                    run_id=run_id,
                    idx=idx,
                    step_type="tool",
                    input_json={"tool_name": tool_name, "args": args},
                )
                result = self._execute_tool_with_retries(
                    run_id=run_id,
                    step_id=step["id"],
                    agent=agent,
                    tool_name=tool_name,
                    args=args,
                )
                if result.get("ok"):
                    self.repo.finish_step(step["id"], output_json=result)
                    self.repo.create_event(
                        run_id,
                        "step.finished",
                        {"step_id": step["id"], "idx": idx, "ok": True},
                    )
                    history.append(
                        {
                            "step_type": "tool",
                            "input": {"tool_name": tool_name, "args": args},
                            "output": result,
                            "error": None,
                        }
                    )
                    idx += 1
                    continue

                message = result.get("error", {}).get("message", "tool failed")
                self.repo.finish_step(step["id"], output_json=result, error=message)
                self.repo.create_event(
                    run_id,
                    "step.finished",
                    {"step_id": step["id"], "idx": idx, "ok": False},
                )
                self.repo.update_run_status(run_id, "failed")
                self.repo.create_event(
                    run_id,
                    "run.failed",
                    {"run_id": run_id, "error": result.get("error")},
                )
                return

            if kind == "ask_user":
                prompt = str(action.get("message") or "Additional input is required.")
                step = self.repo.create_step(
                    run_id=run_id,
                    idx=idx,
                    step_type="ask_user",
                    input_json={"prompt": prompt},
                )
                output = {"ok": True, "status": "awaiting_input", "prompt": prompt}
                self.repo.finish_step(step["id"], output_json=output)
                self.repo.create_event(
                    run_id,
                    "run.awaiting_input",
                    {"run_id": run_id, "step_id": step["id"], "prompt": prompt},
                )
                self.repo.update_run_status(run_id, "awaiting_input")
                return

            if kind == "final_answer":
                message = str(action.get("message") or "done")
                step = self.repo.create_step(
                    run_id=run_id,
                    idx=idx,
                    step_type="eval",
                    input_json={"result": "done"},
                )
                output = {"ok": True, "final_answer": message}
                self.repo.finish_step(step["id"], output_json=output)
                self.repo.create_event(
                    run_id,
                    "step.finished",
                    {"step_id": step["id"], "idx": idx, "ok": True},
                )
                self.repo.update_run_status(run_id, "succeeded")
                self.repo.create_event(
                    run_id,
                    "run.succeeded",
                    {"run_id": run_id, "final_answer": message},
                )
                return

            self.repo.update_run_status(run_id, "failed")
            self.repo.create_event(
                run_id,
                "run.failed",
                {
                    "run_id": run_id,
                    "error": {
                        "code": "invalid_action",
                        "message": "policy returned unsupported action",
                    },
                },
            )
            return

        self.repo.create_event(
            run_id, "run.step_limit_reached", {"step_limit": step_limit}
        )
        self.repo.update_run_status(run_id, "failed")

    def _execute_tool_with_retries(
        self,
        *,
        run_id: str,
        step_id: str,
        agent: dict[str, Any],
        tool_name: str,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute one tool call with configured retry behavior."""
        result: dict[str, Any] = {
            "ok": False,
            "error": {"code": "not_executed", "message": "uninitialized"},
        }
        for attempt in range(1, self.retry_config.max_attempts + 1):
            result = self.tool_gateway.call(
                run_id=run_id,
                step_id=step_id,
                agent=agent,
                tool_name=tool_name,
                args=args,
            )
            if result.get("ok"):
                return result
            self.repo.create_event(
                run_id,
                "tool.retry",
                {
                    "step_id": step_id,
                    "tool_name": tool_name,
                    "attempt": attempt,
                    "error": result.get("error"),
                },
            )
            if attempt < self.retry_config.max_attempts:
                time.sleep(self.retry_config.backoff_seconds)
        return result

    def _process_run_legacy(self, run_id: str) -> None:
        """Process a run using the legacy fixed plan->tool->eval flow."""
        run = self.repo.get_run(run_id)
        if not run:
            return
        if run["status"] == "canceled":
            return

        agent = self.repo.get_agent(run["agent_id"])
        if not agent:
            self.repo.update_run_status(run_id, "failed")
            self.repo.create_event(run_id, "run.failed", {"error": "agent_not_found"})
            return

        if agent.get("status") != "active":
            self.repo.update_run_status(run_id, "failed")
            self.repo.create_event(run_id, "run.failed", {"error": "agent_disabled"})
            return

        self.repo.update_run_status(run_id, "running")
        self.repo.create_event(
            run_id, "run.started", {"run_id": run_id, "agent_id": agent["id"]}
        )

        actions = self.policy.plan(
            run["task"],
            agent=agent,
            context={"run_id": run_id, "step_limit": run["step_limit"]},
        )
        step_limit = int(run["step_limit"])

        for idx, action in enumerate(actions):
            if idx >= step_limit:
                self.repo.create_event(
                    run_id, "run.step_limit_reached", {"step_limit": step_limit}
                )
                self.repo.update_run_status(run_id, "failed")
                return

            latest = self.repo.get_run(run_id)
            if latest and latest["status"] == "canceled":
                self.repo.create_event(run_id, "run.canceled", {"run_id": run_id})
                return

            step = self.repo.create_step(
                run_id=run_id,
                idx=idx,
                step_type=action.step_type,
                input_json=action.args,
            )
            self.repo.create_event(
                run_id,
                "step.started",
                {
                    "step_id": step["id"],
                    "idx": idx,
                    "type": action.step_type,
                    "input": action.args,
                },
            )

            if action.tool_name is None:
                self.repo.finish_step(
                    step["id"], output_json={"ok": True, "note": "no tool"}
                )
                self.repo.create_event(
                    run_id,
                    "step.finished",
                    {"step_id": step["id"], "idx": idx, "ok": True},
                )
                continue

            result: dict[str, Any] = {
                "ok": False,
                "error": {"code": "not_executed", "message": "uninitialized"},
            }
            result = self._execute_tool_with_retries(
                run_id=run_id,
                step_id=step["id"],
                agent=agent,
                tool_name=action.tool_name,
                args=action.args,
            )

            if result.get("ok"):
                self.repo.finish_step(step["id"], output_json=result)
                self.repo.create_event(
                    run_id,
                    "step.finished",
                    {"step_id": step["id"], "idx": idx, "ok": True},
                )
            else:
                message = result.get("error", {}).get("message", "tool failed")
                self.repo.finish_step(step["id"], output_json=result, error=message)
                self.repo.create_event(
                    run_id,
                    "step.finished",
                    {"step_id": step["id"], "idx": idx, "ok": False},
                )
                self.repo.update_run_status(run_id, "failed")
                self.repo.create_event(
                    run_id,
                    "run.failed",
                    {"run_id": run_id, "error": result.get("error")},
                )
                return

        self.repo.update_run_status(run_id, "succeeded")
        self.repo.create_event(run_id, "run.succeeded", {"run_id": run_id})
