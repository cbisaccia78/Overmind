from __future__ import annotations

import json
from urllib import error as urlerror

import pytest

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


def test_infer_with_model_covers_write_recall_remember_paths():
    gateway = ModelGateway(repo=None)

    write_result = gateway._infer_with_model(
        task="write: notes.md: hello world",
        model="stub-v1",
    )
    assert write_result == {
        "tool_name": "write_file",
        "args": {"path": "notes.md", "content": "hello world"},
    }

    recall_result = gateway._infer_with_model(
        task="recall: docs: sqlite tips",
        model="stub-v1",
    )
    assert recall_result == {
        "tool_name": "search_memory",
        "args": {"collection": "docs", "query": "sqlite tips", "top_k": 5},
    }

    remember_result = gateway._infer_with_model(
        task="remember: docs: important note",
        model="stub-v1",
    )
    assert remember_result == {
        "tool_name": "store_memory",
        "args": {"collection": "docs", "text": "important note"},
    }


def test_model_gateway_extractors_cover_fallback_forms():
    assert ModelGateway._extract_shell_command("echo hi") == "echo hi"
    assert ModelGateway._extract_read_path("open README.md") == "README.md"
    assert ModelGateway._extract_read_path("read '/tmp/a.txt'") == "/tmp/a.txt"
    assert ModelGateway._extract_read_path("") == ""
    assert ModelGateway._extract_write_payload("content to out.txt") == (
        "out.txt",
        "content",
    )
    assert ModelGateway._extract_write_payload("just text") == (
        "notes.txt",
        "just text",
    )
    assert ModelGateway._extract_memory_store("remember this") == (
        "default",
        "remember this",
    )
    assert ModelGateway._extract_memory_query("find this") == ("default", "find this")


def test_infer_with_openai_requires_api_key(tmp_path, monkeypatch):
    db_path = str(tmp_path / "model-openai-no-key.db")
    init_db(db_path)
    repo = Repository(db_path)
    gateway = ModelGateway(repo, openai_tools_provider=lambda _: [{"type": "function"}])
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is required"):
        gateway._infer_with_openai(
            task="read", model="gpt-4o-mini", allowed_tools=["read_file"]
        )


def test_infer_with_openai_requires_allowed_tools(tmp_path, monkeypatch):
    db_path = str(tmp_path / "model-openai-no-tools.db")
    init_db(db_path)
    repo = Repository(db_path)
    gateway = ModelGateway(repo, openai_tools_provider=lambda _: [])
    monkeypatch.setenv("OPENAI_API_KEY", "key")

    with pytest.raises(RuntimeError, match="No allowed tools available"):
        gateway._infer_with_openai(
            task="read", model="gpt-4o-mini", allowed_tools=["read_file"]
        )


def test_infer_with_openai_wraps_transport_errors(tmp_path, monkeypatch):
    db_path = str(tmp_path / "model-openai-error.db")
    init_db(db_path)
    repo = Repository(db_path)
    gateway = ModelGateway(
        repo,
        openai_tools_provider=lambda _: [
            {"type": "function", "function": {"name": "read_file"}}
        ],
    )
    monkeypatch.setenv("OPENAI_API_KEY", "key")

    def _fail(req, timeout=10):
        del req, timeout
        raise urlerror.URLError("network")

    monkeypatch.setattr(model_gateway_module.urlrequest, "urlopen", _fail)

    with pytest.raises(RuntimeError, match="openai request failed"):
        gateway._infer_with_openai(
            task="read", model="gpt-4o-mini", allowed_tools=["read_file"]
        )


def test_infer_with_openai_rejects_malformed_or_disallowed_calls(tmp_path, monkeypatch):
    db_path = str(tmp_path / "model-openai-malformed.db")
    init_db(db_path)
    repo = Repository(db_path)
    gateway = ModelGateway(
        repo,
        openai_tools_provider=lambda _: [
            {"type": "function", "function": {"name": "read_file"}}
        ],
    )
    monkeypatch.setenv("OPENAI_API_KEY", "key")

    class _Resp:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

    monkeypatch.setattr(
        model_gateway_module.urlrequest,
        "urlopen",
        lambda req, timeout=10: _Resp({"choices": [{"message": {}}]}),
    )
    with pytest.raises(RuntimeError, match="response missing valid tool call"):
        gateway._infer_with_openai(
            task="read", model="gpt-4o-mini", allowed_tools=["read_file"]
        )

    monkeypatch.setattr(
        model_gateway_module.urlrequest,
        "urlopen",
        lambda req, timeout=10: _Resp(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"function": {"name": "run_shell", "arguments": "{}"}}
                            ]
                        }
                    }
                ]
            }
        ),
    )
    with pytest.raises(RuntimeError, match="selected disallowed tool"):
        gateway._infer_with_openai(
            task="read", model="gpt-4o-mini", allowed_tools=["read_file"]
        )

    monkeypatch.setattr(
        model_gateway_module.urlrequest,
        "urlopen",
        lambda req, timeout=10: _Resp(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "read_file",
                                        "arguments": json.dumps(
                                            ["not", "an", "object"]
                                        ),
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        ),
    )
    with pytest.raises(RuntimeError, match="arguments must be an object"):
        gateway._infer_with_openai(
            task="read", model="gpt-4o-mini", allowed_tools=["read_file"]
        )
