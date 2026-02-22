"""Deterministic orchestrator for executing runs.

Plans actions via a policy, executes tools via the gateway, and records
steps/events with retry and step-limit handling.
"""

from __future__ import annotations

import threading
import time
import math
from dataclasses import dataclass
from pathlib import Path
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
        keep_resources = False
        try:
            if callable(getattr(type(self.policy), "next_action", None)):
                keep_resources = self._process_run_iterative(run_id)
            else:
                keep_resources = self._process_run_legacy(run_id)
        except Exception as exc:  # noqa: BLE001
            self._mark_unexpected_failure(run_id, exc)
        finally:
            releaser = getattr(self.tool_gateway, "release_run_resources", None)
            if callable(releaser) and not keep_resources:
                releaser(run_id)

    def _mark_unexpected_failure(self, run_id: str, exc: Exception) -> None:
        """Record unexpected orchestrator exceptions and fail active runs."""
        run = self.repo.get_run(run_id)
        if not run:
            return
        status = str(run.get("status") or "")
        if status not in {"succeeded", "failed", "canceled", "awaiting_input"}:
            self.repo.update_run_status(run_id, "failed")
        self.repo.create_event(
            run_id,
            "run.failed",
            {
                "run_id": run_id,
                "error": {
                    "code": "orchestrator_exception",
                    "message": str(exc),
                },
            },
        )

    def _process_run_iterative(self, run_id: str) -> bool:
        """Process a run via iterative next-action planning.

        Returns:
            True when resources should be retained (for awaiting-input runs).
        """
        run = self.repo.get_run(run_id)
        if not run:
            return False
        if run["status"] == "canceled":
            return False

        agent = self.repo.get_agent(run["agent_id"])
        if not agent:
            self.repo.update_run_status(run_id, "failed")
            self.repo.create_event(run_id, "run.failed", {"error": "agent_not_found"})
            return False

        if agent.get("status") != "active":
            self.repo.update_run_status(run_id, "failed")
            self.repo.create_event(run_id, "run.failed", {"error": "agent_disabled"})
            return False

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
        consecutive_failures = 0
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

        while step_limit <= 0 or idx < step_limit:
            latest = self.repo.get_run(run_id)
            if latest and latest["status"] == "canceled":
                self.repo.create_event(run_id, "run.canceled", {"run_id": run_id})
                return False

            action = self.policy.next_action(
                run["task"],
                agent=agent,
                context={
                    "run_id": run_id,
                    "step_limit": step_limit,
                    "history": history,
                    "failure_memory": self._build_failure_memory(history),
                },
                history=history,
            )
            kind = str(action.get("kind") or "")
            self.repo.create_event(
                run_id,
                "plan.action_selected",
                {
                    "run_id": run_id,
                    "idx": idx,
                    "kind": kind,
                    "action": self._action_preview(action),
                },
            )

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
                    consecutive_failures = 0
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
                retryable, category = self._classify_retry(result.get("error"))
                self.repo.finish_step(step["id"], output_json=result, error=message)
                self.repo.create_event(
                    run_id,
                    "step.finished",
                    {"step_id": step["id"], "idx": idx, "ok": False},
                )
                self.repo.create_event(
                    run_id,
                    "tool.failed",
                    {
                        "run_id": run_id,
                        "step_id": step["id"],
                        "tool_name": tool_name,
                        "retryable": retryable,
                        "category": category,
                        "error": result.get("error"),
                    },
                )
                history.append(
                    {
                        "step_type": "tool",
                        "input": {"tool_name": tool_name, "args": args},
                        "output": result,
                        "error": message,
                    }
                )
                consecutive_failures += 1
                idx += 1
                if consecutive_failures >= 2:
                    self.repo.update_run_status(run_id, "failed")
                    self.repo.create_event(
                        run_id,
                        "run.failed",
                        {
                            "run_id": run_id,
                            "error": {
                                "code": "failure_budget_exhausted",
                                "message": "too many consecutive failed actions",
                                "last_error": result.get("error"),
                            },
                        },
                    )
                    return False
                continue

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
                return True

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
                return False

            if kind == "verify":
                checks = list(action.get("checks") or [])
                step = self.repo.create_step(
                    run_id=run_id,
                    idx=idx,
                    step_type="verify",
                    input_json={"checks": checks},
                )
                verification = self._run_verification_checks(checks)
                if verification.get("ok"):
                    self.repo.finish_step(step["id"], output_json=verification)
                    self.repo.create_event(
                        run_id,
                        "step.finished",
                        {"step_id": step["id"], "idx": idx, "ok": True},
                    )
                    history.append(
                        {
                            "step_type": "verify",
                            "input": {"checks": checks},
                            "output": verification,
                            "error": None,
                        }
                    )
                    idx += 1
                    continue

                message = verification.get("error", {}).get(
                    "message", "verification failed"
                )
                self.repo.finish_step(
                    step["id"], output_json=verification, error=message
                )
                self.repo.create_event(
                    run_id,
                    "step.finished",
                    {"step_id": step["id"], "idx": idx, "ok": False},
                )
                prompt = str(action.get("on_fail_prompt") or "")
                if prompt:
                    self.repo.create_event(
                        run_id,
                        "run.awaiting_input",
                        {
                            "run_id": run_id,
                            "step_id": step["id"],
                            "prompt": prompt,
                            "verification_error": verification.get("error"),
                        },
                    )
                    self.repo.update_run_status(run_id, "awaiting_input")
                    return True

                self.repo.update_run_status(run_id, "failed")
                self.repo.create_event(
                    run_id,
                    "run.failed",
                    {"run_id": run_id, "error": verification.get("error")},
                )
                return False

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
            return False

        if step_limit > 0:
            self.repo.create_event(
                run_id, "run.step_limit_reached", {"step_limit": step_limit}
            )
            self.repo.update_run_status(run_id, "failed")
        return False

    @staticmethod
    def _action_preview(action: dict[str, Any]) -> dict[str, Any]:
        """Return a compact action summary suitable for timeline events."""
        kind = str(action.get("kind") or "")
        payload: dict[str, Any] = {"kind": kind}
        if kind == "tool_call":
            payload["tool_name"] = str(action.get("tool_name") or "")
            payload["args"] = dict(action.get("args") or {})
            return payload
        if kind == "ask_user":
            message = str(action.get("message") or "").strip()
            payload["message"] = message[:280] if message else ""
            return payload
        if kind == "final_answer":
            message = str(action.get("message") or "").strip()
            payload["message"] = message[:280] if message else ""
            return payload
        if kind == "verify":
            payload["checks"] = list(action.get("checks") or [])
            return payload
        return payload

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
            retryable, category = self._classify_retry(result.get("error"))
            self.repo.create_event(
                run_id,
                "tool.retry.classified",
                {
                    "step_id": step_id,
                    "tool_name": tool_name,
                    "attempt": attempt,
                    "retryable": retryable,
                    "category": category,
                },
            )
            if not retryable:
                return result
            if attempt < self.retry_config.max_attempts:
                time.sleep(self.retry_config.backoff_seconds)
        return result

    @staticmethod
    def _classify_retry(error: dict[str, Any] | None) -> tuple[bool, str]:
        """Classify tool failures into retryable vs non-retryable buckets."""
        payload = error or {}
        code = str(payload.get("code") or "").lower()
        message = str(payload.get("message") or "").lower()

        transient_codes = {
            "timeout",
            "temporarily_unavailable",
            "rate_limited",
            "network_error",
        }
        permanent_codes = {
            "validation_error",
            "tool_not_allowed",
            "tool_not_found",
            "file_not_found",
            "permission_denied",
            "invalid_path",
            "invalid_arguments",
        }
        if code in transient_codes:
            return True, "transient"
        if code in permanent_codes:
            return False, "permanent"
        if any(token in message for token in ["timeout", "temporar", "rate limit"]):
            return True, "transient"
        if any(
            token in message
            for token in [
                "not allowed",
                "not found",
                "invalid",
                "permission",
                "schema",
                "argument",
            ]
        ):
            return False, "permanent"
        return True, "unknown"

    def _run_verification_checks(self, checks: list[dict[str, Any]]) -> dict[str, Any]:
        """Evaluate postcondition checks for artifacts via typed handlers."""
        handlers = self._verification_handlers()
        results: list[dict[str, Any]] = []
        for check in checks:
            check_type = str(check.get("type") or "")
            handler = handlers.get(check_type)
            if handler is None:
                return {
                    "ok": False,
                    "checks": results,
                    "error": {
                        "code": "verification_failed",
                        "message": f"unsupported verification check: {check_type}",
                    },
                }

            result = handler(check)
            results.append(result)
            if result.get("ok"):
                continue
            return {
                "ok": False,
                "checks": results,
                "error": {
                    "code": "verification_failed",
                    "message": str(
                        result.get("error_message")
                        or f"verification check failed: {check_type}"
                    ),
                },
            }

        return {"ok": True, "checks": results}

    def _verification_handlers(self) -> dict[str, Any]:
        """Return verification handlers keyed by check type."""
        return {
            "file_exists": self._check_file_exists,
            "file_min_bytes": self._check_file_min_bytes,
            "png_signature": self._check_png_signature,
            "png_dimensions_min": self._check_png_dimensions_min,
            "file_entropy_min": self._check_file_entropy_min,
        }

    def _check_file_exists(self, check: dict[str, Any]) -> dict[str, Any]:
        raw_path = str(check.get("path") or "")
        path = self._resolve_check_path(raw_path)
        ok = path is not None and path.exists() and path.is_file()
        result = {"type": "file_exists", "path": raw_path, "ok": ok}
        if ok:
            return result
        result["error_message"] = f"expected file does not exist: {raw_path}"
        return result

    def _check_file_min_bytes(self, check: dict[str, Any]) -> dict[str, Any]:
        raw_path = str(check.get("path") or "")
        path = self._resolve_check_path(raw_path)
        minimum = int(check.get("min_bytes") or 1)
        size = path.stat().st_size if path and path.exists() and path.is_file() else 0
        ok = size >= minimum
        result = {
            "type": "file_min_bytes",
            "path": raw_path,
            "min_bytes": minimum,
            "actual_bytes": size,
            "ok": ok,
        }
        if ok:
            return result
        result["error_message"] = f"file smaller than expected: {raw_path}"
        return result

    def _check_png_signature(self, check: dict[str, Any]) -> dict[str, Any]:
        raw_path = str(check.get("path") or "")
        path = self._resolve_check_path(raw_path)
        signature = b"\x89PNG\r\n\x1a\n"
        data = b""
        if path and path.exists() and path.is_file():
            data = path.read_bytes()
        ok = len(data) >= 8 and data[:8] == signature
        result = {
            "type": "png_signature",
            "path": raw_path,
            "ok": ok,
        }
        if ok:
            return result
        result["error_message"] = f"file is not a valid PNG signature: {raw_path}"
        return result

    def _check_png_dimensions_min(self, check: dict[str, Any]) -> dict[str, Any]:
        raw_path = str(check.get("path") or "")
        path = self._resolve_check_path(raw_path)
        min_width = int(check.get("min_width") or 1)
        min_height = int(check.get("min_height") or 1)
        width = 0
        height = 0
        if path and path.exists() and path.is_file():
            width, height = self._read_png_dimensions(path)
        ok = width >= min_width and height >= min_height
        result = {
            "type": "png_dimensions_min",
            "path": raw_path,
            "min_width": min_width,
            "min_height": min_height,
            "actual_width": width,
            "actual_height": height,
            "ok": ok,
        }
        if ok:
            return result
        result["error_message"] = (
            f"PNG dimensions below minimum for {raw_path}: "
            f"{width}x{height} < {min_width}x{min_height}"
        )
        return result

    def _check_file_entropy_min(self, check: dict[str, Any]) -> dict[str, Any]:
        raw_path = str(check.get("path") or "")
        path = self._resolve_check_path(raw_path)
        min_entropy = float(check.get("min_entropy") or 1.0)
        data = b""
        if path and path.exists() and path.is_file():
            data = path.read_bytes()
        entropy = self._byte_entropy(data)
        ok = entropy >= min_entropy
        result = {
            "type": "file_entropy_min",
            "path": raw_path,
            "min_entropy": min_entropy,
            "actual_entropy": round(entropy, 3),
            "ok": ok,
        }
        if ok:
            return result
        result["error_message"] = (
            f"file entropy below threshold for {raw_path}: "
            f"{entropy:.3f} < {min_entropy:.3f}"
        )
        return result

    @staticmethod
    def _read_png_dimensions(path: Path) -> tuple[int, int]:
        """Read PNG dimensions from IHDR if present; return (0, 0) on failure."""
        try:
            data = path.read_bytes()
        except OSError:
            return 0, 0
        if len(data) < 24:
            return 0, 0
        if data[:8] != b"\x89PNG\r\n\x1a\n":
            return 0, 0
        if data[12:16] != b"IHDR":
            return 0, 0
        width = int.from_bytes(data[16:20], "big")
        height = int.from_bytes(data[20:24], "big")
        return width, height

    @staticmethod
    def _byte_entropy(data: bytes) -> float:
        """Compute Shannon entropy in bits/byte for file bytes."""
        if not data:
            return 0.0
        counts = [0] * 256
        for value in data:
            counts[value] += 1
        total = float(len(data))
        entropy = 0.0
        for count in counts:
            if count == 0:
                continue
            probability = count / total
            entropy -= probability * math.log2(probability)
        return entropy

    @staticmethod
    def _build_failure_memory(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Build compact failure memory from prior tool outputs."""
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

    def _resolve_check_path(self, raw_path: str) -> Path | None:
        """Resolve verification paths relative to workspace root."""
        candidate = Path(raw_path.strip())
        if not raw_path.strip():
            return None
        if not candidate.is_absolute():
            candidate = self.tool_gateway.workspace_root / candidate
        try:
            resolved = candidate.resolve()
        except (OSError, RuntimeError):
            return None
        if (
            self.tool_gateway.workspace_root not in resolved.parents
            and resolved != self.tool_gateway.workspace_root
        ):
            return None
        return resolved

    def _process_run_legacy(self, run_id: str) -> bool:
        """Process a run using the legacy fixed plan->tool->eval flow."""
        run = self.repo.get_run(run_id)
        if not run:
            return False
        if run["status"] == "canceled":
            return False

        agent = self.repo.get_agent(run["agent_id"])
        if not agent:
            self.repo.update_run_status(run_id, "failed")
            self.repo.create_event(run_id, "run.failed", {"error": "agent_not_found"})
            return False

        if agent.get("status") != "active":
            self.repo.update_run_status(run_id, "failed")
            self.repo.create_event(run_id, "run.failed", {"error": "agent_disabled"})
            return False

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
            if step_limit > 0 and idx >= step_limit:
                self.repo.create_event(
                    run_id, "run.step_limit_reached", {"step_limit": step_limit}
                )
                self.repo.update_run_status(run_id, "failed")
                return False

            latest = self.repo.get_run(run_id)
            if latest and latest["status"] == "canceled":
                self.repo.create_event(run_id, "run.canceled", {"run_id": run_id})
                return False

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
                return False

        self.repo.update_run_status(run_id, "succeeded")
        self.repo.create_event(run_id, "run.succeeded", {"run_id": run_id})
        return False
