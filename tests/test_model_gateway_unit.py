from __future__ import annotations

import json

from app.db import init_db
import app.model_gateway as model_gateway_module
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


def test_model_gateway_uses_openai_tool_call_when_configured(tmp_path, monkeypatch):
    db_path = str(tmp_path / "model-openai.db")
    init_db(db_path)
    repo = Repository(db_path)

    def fake_openai_tools_provider(tool_names: list[str]) -> list[dict]:
        assert tool_names == ["read_file"]
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            }
        ]

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            body = {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "read_file",
                                        "arguments": json.dumps({"path": "README.md"}),
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
            return json.dumps(body).encode("utf-8")

    gateway = ModelGateway(repo, openai_tools_provider=fake_openai_tools_provider)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        model_gateway_module.urlrequest,
        "urlopen",
        lambda req, timeout=10: _FakeResp(),
    )

    agent = repo.create_agent(
        name="planner-openai",
        role="ops",
        model="gpt-4o-mini",
        tools_allowed=["read_file"],
    )
    run = repo.create_run(agent["id"], "read README", step_limit=3)

    result = gateway.infer(
        task="read README",
        agent=agent,
        context={"run_id": run["id"]},
    )

    assert result["ok"] is True
    assert result["tool_name"] == "read_file"
    assert result["args"] == {"path": "README.md"}

    rows = repo.list_model_calls(run["id"])
    assert len(rows) == 1
    assert rows[0]["response_json"]["tool_name"] == "read_file"
    assert rows[0]["error"] is None
