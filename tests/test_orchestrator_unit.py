from __future__ import annotations

from unittest.mock import MagicMock

from app.orchestrator import Orchestrator, RetryConfig
from app.policy import PlannedAction


def _orchestrator_with(repo: MagicMock, gateway: MagicMock, policy: MagicMock):
    return Orchestrator(
        repo=repo,
        tool_gateway=gateway,
        policy=policy,
        retry_config=RetryConfig(max_attempts=2, backoff_seconds=0),
    )


def _fake_png_bytes(width: int, height: int, payload: bytes) -> bytes:
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_data = (
        width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x02\x00\x00\x00"
    )
    ihdr_chunk = (
        len(ihdr_data).to_bytes(4, "big") + b"IHDR" + ihdr_data + b"\x00\x00\x00\x00"
    )
    return signature + ihdr_chunk + payload


def test_orchestrator_early_exit_when_run_missing():
    repo = MagicMock()
    repo.get_run.return_value = None
    orch = _orchestrator_with(repo, MagicMock(), MagicMock())

    orch.process_run("r1")
    repo.create_event.assert_not_called()


def test_orchestrator_early_exit_when_already_canceled():
    repo = MagicMock()
    repo.get_run.return_value = {"id": "r1", "status": "canceled"}
    orch = _orchestrator_with(repo, MagicMock(), MagicMock())

    orch.process_run("r1")
    repo.create_event.assert_not_called()


def test_orchestrator_keeps_resources_for_awaiting_input_runs():
    class _AskPolicy:
        def next_action(self, *_args, **_kwargs):
            return {"kind": "ask_user", "message": "Need your next instruction."}

    repo = MagicMock()
    repo.get_run.side_effect = [
        {
            "id": "r-await",
            "status": "queued",
            "agent_id": "a-await",
            "task": "navigate and wait",
            "step_limit": 5,
        },
        {"id": "r-await", "status": "running"},
    ]
    repo.get_agent.return_value = {
        "id": "a-await",
        "status": "active",
        "tools_allowed": ["mcp.playwright.browser_navigate"],
    }
    repo.list_steps.return_value = []
    repo.create_step.side_effect = [{"id": "s-plan"}, {"id": "s-ask"}]

    gateway = MagicMock()
    orch = _orchestrator_with(repo, gateway, _AskPolicy())
    orch.process_run("r-await")

    repo.update_run_status.assert_any_call("r-await", "awaiting_input")
    gateway.release_run_resources.assert_not_called()


def test_orchestrator_releases_resources_for_terminal_runs():
    class _DonePolicy:
        def next_action(self, *_args, **_kwargs):
            return {"kind": "final_answer", "message": "done"}

    repo = MagicMock()
    repo.get_run.side_effect = [
        {
            "id": "r-done",
            "status": "queued",
            "agent_id": "a-done",
            "task": "finish",
            "step_limit": 5,
        },
        {"id": "r-done", "status": "running"},
    ]
    repo.get_agent.return_value = {
        "id": "a-done",
        "status": "active",
        "tools_allowed": [],
    }
    repo.list_steps.return_value = []
    repo.create_step.side_effect = [{"id": "s-plan"}, {"id": "s-final"}]

    gateway = MagicMock()
    orch = _orchestrator_with(repo, gateway, _DonePolicy())
    orch.process_run("r-done")

    gateway.release_run_resources.assert_called_once_with("r-done")


def test_orchestrator_marks_run_failed_on_unexpected_exception():
    class _BoomPolicy:
        def next_action(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    repo = MagicMock()
    repo.get_run.side_effect = [
        {
            "id": "r-boom",
            "status": "queued",
            "agent_id": "a-boom",
            "task": "do work",
            "step_limit": 5,
        },
        {"id": "r-boom", "status": "running"},
        {"id": "r-boom", "status": "running"},
    ]
    repo.get_agent.return_value = {
        "id": "a-boom",
        "status": "active",
        "tools_allowed": [],
    }
    repo.list_steps.return_value = []
    repo.create_step.side_effect = [{"id": "s-plan"}]

    gateway = MagicMock()
    orch = _orchestrator_with(repo, gateway, _BoomPolicy())
    orch.process_run("r-boom")

    repo.update_run_status.assert_any_call("r-boom", "failed")
    repo.create_event.assert_any_call(
        "r-boom",
        "run.failed",
        {
            "run_id": "r-boom",
            "error": {"code": "orchestrator_exception", "message": "boom"},
        },
    )
    gateway.release_run_resources.assert_called_once_with("r-boom")


def test_orchestrator_fails_when_agent_missing_or_disabled():
    repo_missing = MagicMock()
    repo_missing.get_run.return_value = {
        "id": "r1",
        "status": "queued",
        "agent_id": "a1",
        "task": "t",
        "step_limit": 2,
    }
    repo_missing.get_agent.return_value = None
    orch_missing = _orchestrator_with(repo_missing, MagicMock(), MagicMock())
    orch_missing.process_run("r1")
    repo_missing.update_run_status.assert_called_with("r1", "failed")
    repo_missing.create_event.assert_any_call(
        "r1", "run.failed", {"error": "agent_not_found"}
    )

    repo_disabled = MagicMock()
    repo_disabled.get_run.return_value = {
        "id": "r2",
        "status": "queued",
        "agent_id": "a2",
        "task": "t",
        "step_limit": 2,
    }
    repo_disabled.get_agent.return_value = {"id": "a2", "status": "disabled"}
    orch_disabled = _orchestrator_with(repo_disabled, MagicMock(), MagicMock())
    orch_disabled.process_run("r2")
    repo_disabled.update_run_status.assert_called_with("r2", "failed")
    repo_disabled.create_event.assert_any_call(
        "r2", "run.failed", {"error": "agent_disabled"}
    )


def test_orchestrator_step_limit_and_mid_run_cancel():
    repo_limit = MagicMock()
    repo_limit.get_run.side_effect = [
        {
            "id": "r1",
            "status": "queued",
            "agent_id": "a1",
            "task": "t",
            "step_limit": 1,
        },
        {"id": "r1", "status": "running"},
    ]
    repo_limit.get_agent.return_value = {"id": "a1", "status": "active"}
    policy = MagicMock()
    policy.plan.return_value = [
        PlannedAction("tool", "store_memory", {"text": "x"}),
        PlannedAction("tool", "store_memory", {"text": "y"}),
    ]
    orch_limit = _orchestrator_with(repo_limit, MagicMock(), policy)
    orch_limit.process_run("r1")
    repo_limit.create_event.assert_any_call(
        "r1", "run.step_limit_reached", {"step_limit": 1}
    )
    repo_limit.update_run_status.assert_any_call("r1", "failed")

    repo_cancel = MagicMock()
    repo_cancel.get_run.side_effect = [
        {
            "id": "r2",
            "status": "queued",
            "agent_id": "a2",
            "task": "t",
            "step_limit": 5,
        },
        {"id": "r2", "status": "canceled"},
    ]
    repo_cancel.get_agent.return_value = {"id": "a2", "status": "active"}
    policy_cancel = MagicMock()
    policy_cancel.plan.return_value = [
        PlannedAction("tool", "store_memory", {"text": "x"})
    ]
    orch_cancel = _orchestrator_with(repo_cancel, MagicMock(), policy_cancel)
    orch_cancel.process_run("r2")
    repo_cancel.create_event.assert_any_call("r2", "run.canceled", {"run_id": "r2"})


def test_orchestrator_retry_then_fail_path(monkeypatch):
    monkeypatch.setattr("app.orchestrator.time.sleep", lambda *_: None)

    repo = MagicMock()
    repo.get_run.side_effect = [
        {
            "id": "r3",
            "status": "queued",
            "agent_id": "a3",
            "task": "t",
            "step_limit": 5,
        },
        {"id": "r3", "status": "running"},
    ]
    repo.get_agent.return_value = {
        "id": "a3",
        "status": "active",
        "tools_allowed": ["run_shell"],
    }
    repo.create_step.return_value = {"id": "s1"}

    policy = MagicMock()
    policy.plan.return_value = [PlannedAction("tool", "run_shell", {"command": "echo"})]

    gateway = MagicMock()
    gateway.call.side_effect = [
        {"ok": False, "error": {"message": "fail1"}},
        {"ok": False, "error": {"message": "fail2"}},
    ]

    orch = _orchestrator_with(repo, gateway, policy)
    orch.process_run("r3")

    policy.plan.assert_called_once_with(
        "t",
        agent=repo.get_agent.return_value,
        context={"run_id": "r3", "step_limit": 5},
    )
    assert gateway.call.call_count == 2
    repo.create_event.assert_any_call(
        "r3",
        "tool.retry",
        {
            "step_id": "s1",
            "tool_name": "run_shell",
            "attempt": 1,
            "error": {"message": "fail1"},
        },
    )
    repo.create_event.assert_any_call(
        "r3",
        "tool.retry",
        {
            "step_id": "s1",
            "tool_name": "run_shell",
            "attempt": 2,
            "error": {"message": "fail2"},
        },
    )
    repo.finish_step.assert_called_with(
        "s1",
        output_json={"ok": False, "error": {"message": "fail2"}},
        error="fail2",
    )
    repo.update_run_status.assert_any_call("r3", "failed")
    repo.create_event.assert_any_call(
        "r3",
        "run.failed",
        {"run_id": "r3", "error": {"message": "fail2"}},
    )


def test_orchestrator_non_retryable_error_stops_early(monkeypatch):
    monkeypatch.setattr("app.orchestrator.time.sleep", lambda *_: None)

    repo = MagicMock()
    repo.get_run.side_effect = [
        {
            "id": "r4",
            "status": "queued",
            "agent_id": "a4",
            "task": "t",
            "step_limit": 5,
        },
        {"id": "r4", "status": "running"},
    ]
    repo.get_agent.return_value = {
        "id": "a4",
        "status": "active",
        "tools_allowed": ["run_shell"],
    }
    repo.create_step.return_value = {"id": "s1"}

    policy = MagicMock()
    policy.plan.return_value = [PlannedAction("tool", "run_shell", {"command": "echo"})]

    gateway = MagicMock()
    gateway.call.return_value = {
        "ok": False,
        "error": {"code": "validation_error", "message": "bad args"},
    }

    orch = _orchestrator_with(repo, gateway, policy)
    orch.process_run("r4")

    assert gateway.call.call_count == 1
    repo.create_event.assert_any_call(
        "r4",
        "tool.retry.classified",
        {
            "step_id": "s1",
            "tool_name": "run_shell",
            "attempt": 1,
            "retryable": False,
            "category": "permanent",
        },
    )


def test_orchestrator_iterative_verify_step_succeeds(tmp_path):
    class _IterativePolicy:
        def __init__(self):
            self._idx = 0

        def next_action(self, *_args, **_kwargs):
            actions = [
                {
                    "kind": "tool_call",
                    "tool_name": "mcp.browser_take_screenshot",
                    "args": {"filename": "wikipedia.png"},
                },
                {
                    "kind": "verify",
                    "checks": [
                        {"type": "file_exists", "path": "wikipedia.png"},
                        {
                            "type": "file_min_bytes",
                            "path": "wikipedia.png",
                            "min_bytes": 8000,
                        },
                    ],
                },
                {"kind": "final_answer", "message": "done"},
            ]
            action = actions[self._idx]
            self._idx += 1
            return action

    screenshot = tmp_path / "wikipedia.png"
    screenshot.write_bytes(b"x" * 9001)

    repo = MagicMock()
    repo.get_run.return_value = {
        "id": "r5",
        "status": "queued",
        "agent_id": "a5",
        "task": "take a screenshot",
        "step_limit": 6,
    }
    repo.get_agent.return_value = {
        "id": "a5",
        "status": "active",
        "tools_allowed": ["mcp.browser_take_screenshot"],
    }
    repo.list_steps.return_value = []
    repo.create_step.side_effect = [
        {"id": "s-plan"},
        {"id": "s-tool"},
        {"id": "s-verify"},
        {"id": "s-eval"},
    ]

    gateway = MagicMock()
    gateway.workspace_root = tmp_path
    gateway.call.return_value = {"ok": True, "filename": "wikipedia.png"}

    orch = _orchestrator_with(repo, gateway, _IterativePolicy())
    orch.process_run("r5")

    repo.update_run_status.assert_any_call("r5", "succeeded")
    repo.create_event.assert_any_call(
        "r5",
        "run.succeeded",
        {"run_id": "r5", "final_answer": "done"},
    )


def test_orchestrator_iterative_can_replan_after_first_failure():
    class _IterativePolicy:
        def __init__(self):
            self._idx = 0
            self.contexts: list[dict] = []

        def next_action(self, _task, *, agent, context, history):
            self.contexts.append(context)
            actions = [
                {
                    "kind": "tool_call",
                    "tool_name": "run_shell",
                    "args": {"command": "bad"},
                },
                {"kind": "final_answer", "message": "replanned successfully"},
            ]
            action = actions[self._idx]
            self._idx += 1
            return action

    repo = MagicMock()
    repo.get_run.return_value = {
        "id": "r6",
        "status": "queued",
        "agent_id": "a6",
        "task": "do a thing",
        "step_limit": 6,
    }
    repo.get_agent.return_value = {
        "id": "a6",
        "status": "active",
        "tools_allowed": ["run_shell"],
    }
    repo.list_steps.return_value = []
    repo.create_step.side_effect = [
        {"id": "s-plan"},
        {"id": "s-tool"},
        {"id": "s-eval"},
    ]

    gateway = MagicMock()
    gateway.workspace_root = None
    gateway.call.return_value = {
        "ok": False,
        "error": {"code": "timeout", "message": "timed out"},
    }

    policy = _IterativePolicy()
    orch = _orchestrator_with(repo, gateway, policy)
    orch.process_run("r6")

    repo.create_event.assert_any_call(
        "r6",
        "tool.failed",
        {
            "run_id": "r6",
            "step_id": "s-tool",
            "tool_name": "run_shell",
            "retryable": True,
            "category": "transient",
            "error": {"code": "timeout", "message": "timed out"},
        },
    )
    assert policy.contexts[-1]["failure_memory"]
    repo.update_run_status.assert_any_call("r6", "succeeded")


def test_orchestrator_verification_rejects_low_entropy_png(tmp_path):
    repo = MagicMock()
    gateway = MagicMock()
    gateway.workspace_root = tmp_path
    orch = _orchestrator_with(repo, gateway, MagicMock())

    path = tmp_path / "blankish.png"
    path.write_bytes(_fake_png_bytes(1280, 720, b"\x00" * 9000))

    result = orch._run_verification_checks(
        [
            {"type": "file_exists", "path": "blankish.png"},
            {"type": "file_min_bytes", "path": "blankish.png", "min_bytes": 8000},
            {"type": "png_signature", "path": "blankish.png"},
            {
                "type": "png_dimensions_min",
                "path": "blankish.png",
                "min_width": 600,
                "min_height": 400,
            },
            {"type": "file_entropy_min", "path": "blankish.png", "min_entropy": 3.5},
        ]
    )

    assert result["ok"] is False
    assert result["error"]["code"] == "verification_failed"
    assert "entropy" in result["error"]["message"]


def test_orchestrator_verification_accepts_rich_png(tmp_path):
    repo = MagicMock()
    gateway = MagicMock()
    gateway.workspace_root = tmp_path
    orch = _orchestrator_with(repo, gateway, MagicMock())

    payload = bytes(range(256)) * 40
    path = tmp_path / "rich.png"
    path.write_bytes(_fake_png_bytes(1280, 720, payload))

    result = orch._run_verification_checks(
        [
            {"type": "file_exists", "path": "rich.png"},
            {"type": "file_min_bytes", "path": "rich.png", "min_bytes": 8000},
            {"type": "png_signature", "path": "rich.png"},
            {
                "type": "png_dimensions_min",
                "path": "rich.png",
                "min_width": 600,
                "min_height": 400,
            },
            {"type": "file_entropy_min", "path": "rich.png", "min_entropy": 3.5},
        ]
    )

    assert result["ok"] is True


def test_orchestrator_emits_plan_action_selected_event():
    class _OneToolPolicy:
        def __init__(self):
            self.calls = 0

        def next_action(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return {
                    "kind": "tool_call",
                    "tool_name": "mcp.playwright.browser_snapshot",
                    "args": {},
                }
            return {"kind": "final_answer", "message": "done"}

    repo = MagicMock()
    repo.get_run.side_effect = [
        {
            "id": "r10",
            "status": "queued",
            "agent_id": "a10",
            "task": "interact with content in a way that generates user feedback",
            "step_limit": 6,
        },
        {"id": "r10", "status": "running"},
        {"id": "r10", "status": "running"},
    ]
    repo.get_agent.return_value = {
        "id": "a10",
        "status": "active",
        "tools_allowed": ["mcp.playwright.browser_snapshot"],
    }
    repo.list_steps.return_value = []
    repo.create_step.side_effect = [
        {"id": "s-plan"},
        {"id": "s-tool"},
        {"id": "s-final"},
    ]

    gateway = MagicMock()
    gateway.call.return_value = {"ok": True, "snapshot": "ok"}

    orch = _orchestrator_with(repo, gateway, _OneToolPolicy())
    orch.process_run("r10")

    assert any(
        call.args[1] == "plan.action_selected"
        and call.args[2].get("action", {}).get("kind") == "tool_call"
        and call.args[2].get("action", {}).get("tool_name")
        == "mcp.playwright.browser_snapshot"
        for call in repo.create_event.call_args_list
    )


def test_orchestrator_final_answer_error_marks_run_failed():
    class _ErrorPolicy:
        def next_action(self, *_args, **_kwargs):
            return {
                "kind": "final_answer",
                "message": "I could not infer the next tool action: timeout",
                "is_error": True,
                "error": {
                    "code": "model_inference_error",
                    "message": "openai request failed: timeout",
                },
            }

    repo = MagicMock()
    repo.get_run.side_effect = [
        {
            "id": "r11",
            "status": "queued",
            "agent_id": "a11",
            "task": "do thing",
            "step_limit": 4,
        },
        {"id": "r11", "status": "running"},
    ]
    repo.get_agent.return_value = {
        "id": "a11",
        "status": "active",
        "tools_allowed": ["run_shell"],
    }
    repo.list_steps.return_value = []
    repo.create_step.side_effect = [{"id": "s-plan"}, {"id": "s-final"}]

    gateway = MagicMock()
    orch = _orchestrator_with(repo, gateway, _ErrorPolicy())
    orch.process_run("r11")

    repo.update_run_status.assert_any_call("r11", "failed")
    repo.create_event.assert_any_call(
        "r11",
        "run.failed",
        {
            "run_id": "r11",
            "error": {
                "code": "model_inference_error",
                "message": "openai request failed: timeout",
            },
        },
    )


def test_orchestrator_emits_supervisor_and_validation_events():
    class _ToolThenDonePolicy:
        def __init__(self):
            self.calls = 0
            self.contexts: list[dict] = []

        def next_action(self, *_args, **kwargs):
            self.contexts.append(kwargs["context"])
            if self.calls == 0:
                self.calls += 1
                return {
                    "kind": "tool_call",
                    "tool_name": "mcp.playwright.browser_snapshot",
                    "args": {},
                }
            return {"kind": "final_answer", "message": "done"}

    repo = MagicMock()
    repo.get_run.side_effect = [
        {
            "id": "r12",
            "status": "queued",
            "agent_id": "a12",
            "task": "inspect page and continue",
            "step_limit": 6,
        },
        {"id": "r12", "status": "running"},
        {"id": "r12", "status": "running"},
    ]
    repo.get_agent.return_value = {
        "id": "a12",
        "status": "active",
        "tools_allowed": ["mcp.playwright.browser_snapshot"],
    }
    repo.list_steps.return_value = []
    repo.create_step.side_effect = [
        {"id": "s-plan"},
        {"id": "s-tool"},
        {"id": "s-final"},
    ]

    gateway = MagicMock()
    gateway.call.return_value = {
        "ok": True,
        "observation": {
            "summary": "MCP tool `browser_snapshot` completed.",
            "action_candidates": ["Home", "Search"],
        },
    }

    policy = _ToolThenDonePolicy()
    orch = _orchestrator_with(repo, gateway, policy)
    orch.process_run("r12")

    assert "supervisor" in policy.contexts[0]
    assert policy.contexts[0]["supervisor"]["mode"] in {
        "exploration",
        "execution",
        "recovery",
    }
    assert any(
        call.args[1] == "supervisor.decided" for call in repo.create_event.call_args_list
    )
    assert any(
        call.args[1] == "worker.validation" for call in repo.create_event.call_args_list
    )

    tool_finish = next(
        call
        for call in repo.finish_step.call_args_list
        if call.args and call.args[0] == "s-tool"
    )
    assert "validation" in tool_finish.kwargs["output_json"]
