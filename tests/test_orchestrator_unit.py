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
            "step_limit": 0,
        }
    ]
    repo_limit.get_agent.return_value = {"id": "a1", "status": "active"}
    policy = MagicMock()
    policy.plan.return_value = [PlannedAction("tool", "store_memory", {"text": "x"})]
    orch_limit = _orchestrator_with(repo_limit, MagicMock(), policy)
    orch_limit.process_run("r1")
    repo_limit.create_event.assert_any_call(
        "r1", "run.step_limit_reached", {"step_limit": 0}
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
