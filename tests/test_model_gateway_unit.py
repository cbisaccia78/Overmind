from __future__ import annotations

import json

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


def test_model_gateway_emits_model_output_events_for_live_stream(tmp_path):
    db_path = str(tmp_path / "model-events.db")
    init_db(db_path)
    repo = Repository(db_path)
    gateway = ModelGateway(repo)

    agent = repo.create_agent(
        name="planner-events",
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
    events = repo.list_events(run["id"])
    event_types = [str(event.get("type") or "") for event in events]
    assert "model.infer.started" in event_types
    assert "model.output.started" in event_types
    assert "model.output.delta" in event_types
    assert "model.output.completed" in event_types


def test_model_gateway_uses_openai_tool_call_when_configured(tmp_path, monkeypatch):
    db_path = str(tmp_path / "model-openai.db")
    init_db(db_path)
    repo = Repository(db_path)

    def fake_openai_tools_provider(tool_names: list[str]) -> list[dict]:
        assert tool_names == ["read_file", "ask_user", "final_answer"]
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

    gateway = ModelGateway(repo, openai_tools_provider=fake_openai_tools_provider)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        gateway,
        "_request_openai_responses",
        lambda **kwargs: {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_read_file_1",
                    "name": "read_file",
                    "arguments": json.dumps({"path": "README.md"}),
                }
            ]
        },
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


def test_model_gateway_uses_deepseek_tool_call_when_model_is_deepseek(
    tmp_path, monkeypatch
):
    db_path = str(tmp_path / "model-deepseek.db")
    init_db(db_path)
    repo = Repository(db_path)

    def fake_openai_tools_provider(tool_names: list[str]) -> list[dict]:
        assert tool_names == ["read_file", "ask_user", "final_answer"]
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
            del exc_type, exc, tb
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

    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setattr(
        model_gateway_module.urlrequest,
        "urlopen",
        lambda req, timeout=10: _FakeResp(),
    )

    agent = repo.create_agent(
        name="planner-deepseek",
        role="ops",
        model="deepseek-chat",
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


def test_model_gateway_resolves_openai_tool_aliases(tmp_path, monkeypatch):
    db_path = str(tmp_path / "model-openai-alias.db")
    init_db(db_path)
    repo = Repository(db_path)

    def fake_openai_tools_provider(
        tool_names: list[str],
    ) -> tuple[list[dict], dict[str, str]]:
        assert tool_names == ["mcp.playwright.navigate"]
        return (
            [
                {
                    "type": "function",
                    "function": {
                        "name": "mcp_playwright_navigate",
                        "description": "Navigate",
                        "parameters": {
                            "type": "object",
                            "properties": {"url": {"type": "string"}},
                            "required": ["url"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            {"mcp_playwright_navigate": "mcp.playwright.navigate"},
        )

    gateway = ModelGateway(repo, openai_tools_provider=fake_openai_tools_provider)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        gateway,
        "_request_openai_responses",
        lambda **kwargs: {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_nav_1",
                    "name": "mcp_playwright_navigate",
                    "arguments": json.dumps({"url": "https://wikipedia.org"}),
                }
            ]
        },
    )

    result = gateway._infer_with_openai(
        task="navigate",
        model="gpt-4o-mini",
        allowed_tools=["mcp.playwright.navigate"],
    )
    assert result["tool_name"] == "mcp.playwright.navigate"
    assert result["args"] == {"url": "https://wikipedia.org"}


def test_infer_with_openai_retries_required_tool_choice_on_missing_tool_call(
    tmp_path, monkeypatch
):
    db_path = str(tmp_path / "model-openai-retry.db")
    init_db(db_path)
    repo = Repository(db_path)
    gateway = ModelGateway(
        repo,
        openai_tools_provider=lambda _: [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
    )
    monkeypatch.setenv("OPENAI_API_KEY", "key")

    calls: list[dict] = []

    def _request_openai_responses(*, payload: dict, **kwargs):
        del kwargs
        calls.append(payload)
        if payload.get("tool_choice") == "auto":
            return {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "No tool yet"}
                        ],
                    }
                ]
            }
        assert payload.get("tool_choice") == "required"
        return {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_read_file_2",
                    "name": "read_file",
                    "arguments": json.dumps({"path": "README.md"}),
                }
            ]
        }

    monkeypatch.setattr(gateway, "_request_openai_responses", _request_openai_responses)

    result = gateway._infer_with_openai(
        task="read me",
        model="gpt-4o-mini",
        allowed_tools=["read_file"],
    )
    assert result == {"tool_name": "read_file", "args": {"path": "README.md"}}
    assert len(calls) == 2
    assert calls[0]["tool_choice"] == "auto"
    assert calls[1]["tool_choice"] == "required"
    assert calls[1]["input"][0]["role"] == "system"


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

    monkeypatch.setattr(
        gateway,
        "_request_openai_responses",
        lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("openai request failed: network")
        ),
    )

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

    monkeypatch.setattr(
        gateway,
        "_request_openai_responses",
        lambda **kwargs: {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "No function call"}],
                }
            ]
        },
    )
    with pytest.raises(RuntimeError, match="response missing valid tool call"):
        gateway._infer_with_openai(
            task="read", model="gpt-4o-mini", allowed_tools=["read_file"]
        )

    monkeypatch.setattr(
        gateway,
        "_request_openai_responses",
        lambda **kwargs: {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_disallowed",
                    "name": "run_shell",
                    "arguments": "{}",
                }
            ]
        },
    )
    with pytest.raises(RuntimeError, match="selected disallowed tool"):
        gateway._infer_with_openai(
            task="read", model="gpt-4o-mini", allowed_tools=["read_file"]
        )

    monkeypatch.setattr(
        gateway,
        "_request_openai_responses",
        lambda **kwargs: {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_bad_args",
                    "name": "read_file",
                    "arguments": json.dumps(["not", "an", "object"]),
                }
            ]
        },
    )
    with pytest.raises(RuntimeError, match="arguments must be an object"):
        gateway._infer_with_openai(
            task="read", model="gpt-4o-mini", allowed_tools=["read_file"]
        )


def test_infer_with_openai_applies_reasoning_tuning_for_reasoning_models(
    tmp_path, monkeypatch
):
    db_path = str(tmp_path / "model-openai-reasoning.db")
    init_db(db_path)
    repo = Repository(db_path)
    gateway = ModelGateway(
        repo,
        openai_tools_provider=lambda _: [
            {"type": "function", "function": {"name": "read_file"}}
        ],
    )
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("OVERMIND_OPENAI_REASONING_EFFORT", "high")

    calls: list[dict] = []

    def _request_openai_responses(*, payload: dict, **kwargs):
        del kwargs
        calls.append(payload)
        return {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_reasoning_tune",
                    "name": "read_file",
                    "arguments": json.dumps({"path": "README.md"}),
                }
            ]
        }

    monkeypatch.setattr(gateway, "_request_openai_responses", _request_openai_responses)

    result = gateway._infer_with_openai(
        task="read", model="gpt-5.2", allowed_tools=["read_file"]
    )
    assert result == {"tool_name": "read_file", "args": {"path": "README.md"}}
    assert len(calls) == 1
    assert calls[0].get("reasoning") == {"effort": "high"}
    assert "temperature" not in calls[0]


def test_infer_with_openai_retries_without_reasoning_fields_when_needed(
    tmp_path, monkeypatch
):
    db_path = str(tmp_path / "model-openai-reasoning-retry.db")
    init_db(db_path)
    repo = Repository(db_path)
    gateway = ModelGateway(
        repo,
        openai_tools_provider=lambda _: [
            {"type": "function", "function": {"name": "read_file"}}
        ],
    )
    monkeypatch.setenv("OPENAI_API_KEY", "key")

    calls: list[dict] = []

    def _request_openai_responses(*, payload: dict, **kwargs):
        del kwargs
        calls.append(payload)
        if "reasoning" in payload:
            raise RuntimeError("openai request failed: unsupported reasoning field")
        return {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_reasoning_retry",
                    "name": "read_file",
                    "arguments": json.dumps({"path": "README.md"}),
                }
            ]
        }

    monkeypatch.setattr(gateway, "_request_openai_responses", _request_openai_responses)

    result = gateway._infer_with_openai(
        task="read", model="gpt-5.2", allowed_tools=["read_file"]
    )
    assert result == {"tool_name": "read_file", "args": {"path": "README.md"}}
    assert len(calls) == 2
    assert "reasoning" in calls[0]
    assert "reasoning" not in calls[1]


def test_model_gateway_persists_reasoning_and_tool_call_metadata(tmp_path, monkeypatch):
    db_path = str(tmp_path / "model-openai-reasoning-metadata.db")
    init_db(db_path)
    repo = Repository(db_path)
    gateway = ModelGateway(
        repo,
        openai_tools_provider=lambda _: [
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
        ],
    )
    monkeypatch.setenv("OPENAI_API_KEY", "key")

    monkeypatch.setattr(
        gateway,
        "_request_openai_responses",
        lambda **kwargs: {
            "output": [
                {
                    "type": "reasoning",
                    "summary": [
                        {
                            "type": "summary_text",
                            "text": "Need file contents before deciding.",
                        }
                    ],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Reading file now."}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_reasoning_1",
                    "name": "read_file",
                    "arguments": json.dumps({"path": "README.md"}),
                },
            ]
        },
    )

    agent = repo.create_agent(
        name="planner-openai-reasoning-metadata",
        role="ops",
        model="gpt-5.2",
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
    assert result["response"]["reasoning_content"] == "Need file contents before deciding."
    assert result["response"]["assistant_content"] == "Reading file now."
    assert result["response"]["raw_tool_call"]["id"] == "call_reasoning_1"

    rows = repo.list_model_calls(run["id"])
    assert len(rows) == 1
    assert rows[0]["response_json"]["reasoning_content"] == (
        "Need file contents before deciding."
    )
    assert rows[0]["response_json"]["raw_tool_call"]["id"] == "call_reasoning_1"


def test_deepseek_followup_replays_reasoning_content_for_tool_calls(
    tmp_path, monkeypatch
):
    db_path = str(tmp_path / "model-deepseek-reasoning-followup.db")
    init_db(db_path)
    repo = Repository(db_path)
    gateway = ModelGateway(
        repo,
        openai_tools_provider=lambda _: [
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
        ],
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")

    agent = repo.create_agent(
        name="planner-deepseek-followup",
        role="ops",
        model="deepseek-chat",
        tools_allowed=["read_file"],
    )
    run = repo.create_run(agent["id"], "read README", step_limit=5)

    repo.create_model_call(
        run_id=run["id"],
        agent_id=agent["id"],
        model="deepseek-chat",
        request_json={"task": "read README"},
        response_json={
            "tool_name": "read_file",
            "args": {"path": "README.md"},
            "assistant_content": "Calling read_file.",
            "reasoning_content": "Need to inspect README before planning next step.",
            "raw_tool_call": {
                "id": "call_deepseek_1",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": json.dumps({"path": "README.md"}),
                },
            },
        },
        usage_json={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        error=None,
        latency_ms=10,
    )

    calls: list[dict] = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "read_file",
                                            "arguments": json.dumps(
                                                {"path": "README.md"}
                                            ),
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            ).encode("utf-8")

    def _urlopen(req, timeout=10):
        del timeout
        calls.append(json.loads(req.data.decode("utf-8")))
        return _Resp()

    monkeypatch.setattr(model_gateway_module.urlrequest, "urlopen", _urlopen)

    result = gateway.infer(
        task="decide the next action",
        agent=agent,
        context={
            "run_id": run["id"],
            "history": [
                {
                    "step_type": "tool",
                    "input": {"tool_name": "read_file", "args": {"path": "README.md"}},
                    "output": {"ok": True, "stdout": "hello"},
                    "error": None,
                }
            ],
        },
    )
    assert result["ok"] is True
    assert result["tool_name"] == "read_file"
    assert len(calls) == 1

    messages = calls[0]["messages"]
    assert messages[0]["role"] == "assistant"
    assert "Need to inspect README before planning next step." in messages[0]["content"]
    assert messages[0]["tool_calls"][0]["id"] == "call_deepseek_1"
    assert messages[1]["role"] == "tool"
    assert messages[1]["tool_call_id"] == "call_deepseek_1"
    assert messages[2] == {"role": "user", "content": "decide the next action"}


def test_infer_with_deepseek_applies_thinking_mode_when_enabled(
    tmp_path, monkeypatch
):
    db_path = str(tmp_path / "model-deepseek-thinking.db")
    init_db(db_path)
    repo = Repository(db_path)
    gateway = ModelGateway(
        repo,
        openai_tools_provider=lambda _: [
            {"type": "function", "function": {"name": "read_file"}}
        ],
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "key")
    monkeypatch.setenv("OVERMIND_DEEPSEEK_THINKING_MODE", "enabled")

    calls: list[dict] = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "read_file",
                                            "arguments": json.dumps(
                                                {"path": "README.md"}
                                            ),
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            ).encode("utf-8")

    def _urlopen(req, timeout=10):
        del timeout
        calls.append(json.loads(req.data.decode("utf-8")))
        return _Resp()

    monkeypatch.setattr(model_gateway_module.urlrequest, "urlopen", _urlopen)

    result = gateway._infer_with_deepseek(
        task="read",
        model="deepseek-chat",
        allowed_tools=["read_file"],
    )
    assert result == {"tool_name": "read_file", "args": {"path": "README.md"}}
    assert len(calls) == 1
    assert calls[0]["thinking"] == {"type": "enabled"}
    assert "temperature" not in calls[0]
