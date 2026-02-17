from __future__ import annotations

from app.main import AppState, create_app
from app.policy import PlannedAction, Policy


class RecordingPolicy(Policy):
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, dict]] = []

    def plan(self, task: str, *, agent: dict, context: dict) -> list[PlannedAction]:
        self.calls.append((task, agent, context))
        return [
            PlannedAction("plan", None, {"task": task}),
            PlannedAction("eval", None, {"result": "done"}),
        ]


def test_app_state_uses_injected_policy(tmp_path):
    policy = RecordingPolicy()
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    state = AppState(
        db_path=str(tmp_path / "state.db"),
        workspace_root=str(workspace),
        policy=policy,
    )

    assert state.policy is policy
    assert state.orchestrator.policy is policy


def test_create_app_uses_injected_policy(tmp_path):
    policy = RecordingPolicy()
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    application = create_app(
        db_path=str(tmp_path / "app.db"),
        workspace_root=str(workspace),
        policy=policy,
    )

    assert application.state.services.policy is policy
    assert application.state.services.orchestrator.policy is policy
