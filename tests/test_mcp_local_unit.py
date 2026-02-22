from __future__ import annotations

import asyncio
import json

import pytest

import app.mcp_local as mcp_local


@pytest.fixture(autouse=True)
def _clear_sessions() -> None:
    with mcp_local._SESSIONS_LOCK:
        sessions = list(mcp_local._SESSIONS.values())
        mcp_local._SESSIONS.clear()
    for session in sessions:
        if hasattr(session, "close"):
            session.close()
    yield
    with mcp_local._SESSIONS_LOCK:
        sessions = list(mcp_local._SESSIONS.values())
        mcp_local._SESSIONS.clear()
    for session in sessions:
        if hasattr(session, "close"):
            session.close()


def test_parse_and_dump_local_mcp_servers() -> None:
    assert mcp_local.parse_local_mcp_servers(None) == []
    assert mcp_local.parse_local_mcp_servers("") == []
    assert mcp_local.parse_local_mcp_servers("not-json") == []
    assert mcp_local.parse_local_mcp_servers(json.dumps({"id": "x"})) == []

    raw = json.dumps(
        [
            {"id": "", "command": "python"},
            {"id": "server-a", "command": "", "enabled": True},
            {
                "id": "server-a",
                "command": "python",
                "args": ["-m", "server"],
                "env": {"PORT": 8000},
                "enabled": False,
            },
            {
                "id": "server-b",
                "command": "node",
            },
            "bad-entry",
        ]
    )

    configs = mcp_local.parse_local_mcp_servers(raw)
    assert len(configs) == 2
    assert configs[0].id == "server-a"
    assert configs[0].command == "python"
    assert configs[0].args == ["-m", "server"]
    assert configs[0].env == {"PORT": "8000"}
    assert configs[0].enabled is False

    assert configs[1].id == "server-b"
    assert configs[1].args == []
    assert configs[1].env == {}
    assert configs[1].enabled is True

    dumped = mcp_local.dump_local_mcp_servers(configs)
    payload = json.loads(dumped)
    assert payload[0]["id"] == "server-a"
    assert payload[1]["id"] == "server-b"


def test_discover_tools_normalizes_and_handles_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = mcp_local.LocalMcpServerConfig(
        id="demo",
        command="python",
        args=[],
        env={},
        enabled=True,
    )

    tools_payload = [
        {
            "name": "echo",
            "description": "Echo tool",
            "inputSchema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
            },
        },
        {
            "name": "no-schema",
            "description": "No schema",
            "inputSchema": "invalid",
        },
        {"description": "missing name"},
    ]

    def _fake_run_coro(coro):
        coro.close()
        return tools_payload

    monkeypatch.setattr(mcp_local, "_run_coro", _fake_run_coro)
    discovered = mcp_local.discover_tools(config)
    assert [tool.remote_name for tool in discovered] == ["echo", "no-schema"]
    assert discovered[0].local_name == "mcp.demo.echo"
    assert discovered[1].input_schema == {
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    }

    def _raise_run_coro(coro):
        coro.close()
        raise RuntimeError("boom")

    monkeypatch.setattr(mcp_local, "_run_coro", _raise_run_coro)
    assert mcp_local.discover_tools(config) == []


def test_call_tool_success_error_and_session_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = mcp_local.LocalMcpServerConfig(
        id="demo",
        command="python",
        args=[],
        env={},
        enabled=True,
    )

    def _run_ok(coro):
        coro.close()
        return {"isError": False, "content": [{"type": "text", "text": "ok"}]}

    monkeypatch.setattr(mcp_local, "_run_coro", _run_ok)
    ok = mcp_local.call_tool(config, "echo", {"text": "x"})
    assert ok["ok"] is True
    assert ok["mcp"]["server_id"] == "demo"

    def _run_is_error(coro):
        coro.close()
        return {"isError": True, "content": []}

    monkeypatch.setattr(mcp_local, "_run_coro", _run_is_error)
    is_error = mcp_local.call_tool(config, "echo", {"text": "x"})
    assert is_error["ok"] is False
    assert is_error["error"]["code"] == "mcp_tool_error"

    def _run_textual_error(coro):
        coro.close()
        return {
            "isError": False,
            "content": [
                {
                    "type": "text",
                    "text": "### Error\nError: browserType.launchPersistentContext failed",
                }
            ],
        }

    monkeypatch.setattr(mcp_local, "_run_coro", _run_textual_error)
    textual_error = mcp_local.call_tool(config, "echo", {"text": "x"})
    assert textual_error["ok"] is False
    assert textual_error["error"]["code"] == "mcp_tool_error"
    assert "### Error" in textual_error["error"]["message"]

    def _run_raises(coro):
        coro.close()
        raise RuntimeError("transport down")

    monkeypatch.setattr(mcp_local, "_run_coro", _run_raises)
    transport_error = mcp_local.call_tool(config, "echo", {"text": "x"})
    assert transport_error["ok"] is False
    assert transport_error["error"]["code"] == "mcp_transport_error"

    class _FakeSession:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, str], int]] = []

        def call_tool(self, *, remote_name: str, args: dict[str, str], timeout_s: int):
            self.calls.append((remote_name, args, timeout_s))
            return {"isError": False, "content": [{"type": "text", "text": "session"}]}

    session = _FakeSession()
    monkeypatch.setattr(mcp_local, "_get_or_create_session", lambda **_: session)
    via_session = mcp_local.call_tool(
        config,
        "echo",
        {"text": "x"},
        timeout_s=5,
        session_key="run-1",
    )
    assert via_session["ok"] is True
    assert session.calls == [("echo", {"text": "x"}, 5)]


def test_close_sessions_and_get_or_create_session_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Session:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    s1 = _Session()
    s2 = _Session()
    s3 = _Session()
    with mcp_local._SESSIONS_LOCK:
        mcp_local._SESSIONS[("a", "run-1")] = s1
        mcp_local._SESSIONS[("b", "run-1")] = s2
        mcp_local._SESSIONS[("c", "run-2")] = s3

    closed = mcp_local.close_sessions("run-1")
    assert closed == 2
    assert s1.closed is True
    assert s2.closed is True
    assert s3.closed is False

    existing = _Session()
    with mcp_local._SESSIONS_LOCK:
        mcp_local._SESSIONS[("srv", "reuse")] = existing

    config = mcp_local.LocalMcpServerConfig(
        id="srv",
        command="python",
        args=[],
        env={},
        enabled=True,
    )
    reused = mcp_local._get_or_create_session(
        config=config,
        session_key="reuse",
        timeout_s=1,
    )
    assert reused is existing

    preexisting = _Session()

    class _CreatedSession(_Session):
        def __init__(self, config: mcp_local.LocalMcpServerConfig, timeout_s: int):
            super().__init__()
            del timeout_s
            with mcp_local._SESSIONS_LOCK:
                mcp_local._SESSIONS[(config.id, "race")] = preexisting

    monkeypatch.setattr(mcp_local, "_PersistentMcpSession", _CreatedSession)
    raced = mcp_local._get_or_create_session(
        config=config,
        session_key="race",
        timeout_s=1,
    )
    assert raced is preexisting


def test_run_coro_and_normalize_result_branches() -> None:
    async def _value() -> str:
        return "ok"

    assert mcp_local._run_coro(_value()) == "ok"

    async def _inside_loop_value() -> str:
        async def _inner() -> str:
            return "inside"

        return mcp_local._run_coro(_inner())

    assert asyncio.run(_inside_loop_value()) == "inside"

    async def _inside_loop_error() -> None:
        async def _boom() -> None:
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            mcp_local._run_coro(_boom())

    asyncio.run(_inside_loop_error())

    class _Dumpable:
        def model_dump(self, by_alias: bool, exclude_none: bool) -> dict[str, object]:
            assert by_alias is True
            assert exclude_none is True
            return {"isError": False, "content": ["dumped"]}

    dumped = mcp_local._normalize_tool_result(_Dumpable())
    assert dumped["content"] == ["dumped"]

    from_dict = mcp_local._normalize_tool_result(
        {"isError": False, "content": ["dict"]}
    )
    assert from_dict == {"isError": False, "content": ["dict"]}

    fallback = mcp_local._normalize_tool_result(123)
    assert fallback == {"content": ["123"], "isError": False}


def test_async_helpers_and_persistent_session(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Tool:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def model_dump(self, by_alias: bool, exclude_none: bool) -> dict[str, object]:
            assert by_alias is True
            assert exclude_none is True
            return self._payload

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

        async def list_tools(self):
            return [
                _Tool(
                    {
                        "name": "echo",
                        "description": "Echo",
                        "inputSchema": {"type": "object"},
                    }
                )
            ]

        async def call_tool(
            self,
            *,
            name: str,
            arguments: dict[str, str],
            timeout: int,
            raise_on_error: bool,
        ):
            assert name == "echo"
            assert arguments == {"text": "hello"}
            assert timeout in {1, 3}
            assert raise_on_error is False
            return {"isError": False, "content": [{"type": "text", "text": "hello"}]}

    config = mcp_local.LocalMcpServerConfig(
        id="demo",
        command="python",
        args=[],
        env={},
        enabled=True,
    )

    monkeypatch.setattr(
        mcp_local, "_build_client", lambda *_args, **_kwargs: _FakeClient()
    )

    tools = asyncio.run(mcp_local._discover_tools_async(config, timeout_s=3))
    assert tools[0]["name"] == "echo"

    called = asyncio.run(
        mcp_local._call_tool_async(
            config,
            remote_name="echo",
            args={"text": "hello"},
            timeout_s=3,
        )
    )
    assert called["isError"] is False

    session = mcp_local._PersistentMcpSession(config=config, timeout_s=1)
    listed = session.list_tools(timeout_s=1)
    assert listed[0]["name"] == "echo"
    invoked = session.call_tool(remote_name="echo", args={"text": "hello"}, timeout_s=1)
    assert invoked["isError"] is False
    session.close()


def test_persistent_session_startup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FailingClient:
        async def __aenter__(self):
            raise RuntimeError("startup failed")

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

    config = mcp_local.LocalMcpServerConfig(
        id="demo",
        command="python",
        args=[],
        env={},
        enabled=True,
    )

    monkeypatch.setattr(
        mcp_local, "_build_client", lambda *_args, **_kwargs: _FailingClient()
    )

    with pytest.raises(RuntimeError, match="startup failed"):
        mcp_local._PersistentMcpSession(config=config, timeout_s=1)
