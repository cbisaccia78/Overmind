from __future__ import annotations

from app.db import init_db
from app.model_gateway import ModelGateway
from app.repository import Repository


def test_model_gateway_infers_tool_and_persists_call(tmp_path):
    db_path = str(tmp_path / "model.db")
    init_db(db_path)
    repo = Repository(db_path)
    gateway = ModelGateway(repo)

    agent = repo.create_agent(
        name="planner",
        role="ops",
        model="stub-v1",
        tools_allowed=["run_shell"],
    )
    run = repo.create_run(agent["id"], "shell:echo hi", step_limit=3)

    result = gateway.infer(
        task="shell:echo hi",
        agent=agent,
        context={"run_id": run["id"]},
    )

    assert result["ok"] is True
    assert result["tool_name"] == "run_shell"
    assert result["args"]["command"] == "echo hi"
    assert result["usage"]["total_tokens"] >= 1

    rows = repo.list_model_calls(run["id"])
    assert len(rows) == 1
    assert rows[0]["model"] == "stub-v1"
    assert rows[0]["request_json"]["task"] == "shell:echo hi"
    assert rows[0]["response_json"]["tool_name"] == "run_shell"
    assert rows[0]["usage_json"]["total_tokens"] >= 1
    assert rows[0]["error"] is None


def test_model_gateway_persists_error_when_model_fails(tmp_path):
    db_path = str(tmp_path / "model-fail.db")
    init_db(db_path)
    repo = Repository(db_path)
    gateway = ModelGateway(repo)

    agent = repo.create_agent(
        name="planner-fail",
        role="ops",
        model="fail-v1",
        tools_allowed=["store_memory"],
    )
    run = repo.create_run(agent["id"], "anything", step_limit=3)

    result = gateway.infer(
        task="anything",
        agent=agent,
        context={"run_id": run["id"]},
    )

    assert result["ok"] is False
    assert result["error"]["code"] == "model_error"

    rows = repo.list_model_calls(run["id"])
    assert len(rows) == 1
    assert rows[0]["model"] == "fail-v1"
    assert "unavailable" in rows[0]["error"]
