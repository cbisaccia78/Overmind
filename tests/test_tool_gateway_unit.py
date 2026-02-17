from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from app.tool_gateway import ToolGateway


def _gateway(tmp_path: Path) -> ToolGateway:
    repo = MagicMock()
    memory = MagicMock()
    docker = MagicMock()
    return ToolGateway(
        repo=repo,
        memory=memory,
        docker_runner=docker,
        workspace_root=str(tmp_path),
        max_file_bytes=10,
    )


def test_safe_path_rejects_escape(tmp_path: Path):
    gateway = _gateway(tmp_path)
    try:
        gateway._safe_path("../escape.txt")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "escapes workspace" in str(exc)


def test_call_wraps_dispatch_exception_and_denied_tool(tmp_path: Path):
    gateway = _gateway(tmp_path)
    agent = {"id": "a1", "tools_allowed": ["read_file"]}

    denied = gateway.call(
        run_id="r1",
        step_id="s1",
        agent=agent,
        tool_name="write_file",
        args={"path": "x", "content": "y"},
    )
    assert denied["ok"] is False
    assert denied["error"]["code"] == "tool_not_allowed"

    gateway._dispatch = MagicMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]
    err = gateway.call(
        run_id="r1",
        step_id="s2",
        agent={"id": "a1", "tools_allowed": ["read_file"]},
        tool_name="read_file",
        args={"path": "x"},
    )
    assert err["ok"] is False
    assert err["error"]["code"] == "tool_error"
    assert "boom" in err["error"]["message"]

    unregistered = gateway.call(
        run_id="r1",
        step_id="s3",
        agent={"id": "a1", "tools_allowed": ["ghost_tool"]},
        tool_name="ghost_tool",
        args={},
    )
    assert unregistered["ok"] is False
    assert unregistered["error"]["code"] == "tool_not_allowed"
    assert "not registered" in unregistered["error"]["message"]


def test_dispatch_all_branches(tmp_path: Path):
    gateway = _gateway(tmp_path)

    # run_shell bad args
    bad = gateway._dispatch("run_shell", {"command": "   "})
    assert bad["ok"] is False
    assert bad["error"]["code"] == "bad_args"

    # run_shell success path (delegation)
    gateway.docker_runner.run_shell.return_value = {"ok": True, "exit_code": 0}
    shell = gateway._dispatch(
        "run_shell",
        {
            "command": "echo hi",
            "timeout_s": 9,
            "allow_write": True,
            "write_subdir": "tmp",
        },
    )
    assert shell["ok"] is True
    gateway.docker_runner.run_shell.assert_called_once()

    # read_file not found
    missing = gateway._dispatch("read_file", {"path": "missing.txt"})
    assert missing["ok"] is False
    assert missing["error"]["code"] == "not_found"

    # read_file too large
    large = tmp_path / "large.txt"
    large.write_text("x" * 20, encoding="utf-8")
    too_large = gateway._dispatch("read_file", {"path": "large.txt"})
    assert too_large["ok"] is False
    assert too_large["error"]["code"] == "size_limit"

    # read_file success
    small = tmp_path / "small.txt"
    small.write_text("hello", encoding="utf-8")
    read_ok = gateway._dispatch("read_file", {"path": "small.txt"})
    assert read_ok["ok"] is True
    assert read_ok["path"] == "small.txt"

    # write_file too large
    write_big = gateway._dispatch(
        "write_file", {"path": "out.txt", "content": "y" * 20}
    )
    assert write_big["ok"] is False
    assert write_big["error"]["code"] == "size_limit"

    # write_file success
    write_ok = gateway._dispatch(
        "write_file", {"path": "dir/out.txt", "content": "abc"}
    )
    assert write_ok["ok"] is True
    assert write_ok["bytes"] == 3
    assert (tmp_path / "dir" / "out.txt").read_text(encoding="utf-8") == "abc"

    # store/search memory
    gateway.memory.store.return_value = {"id": "m1"}
    store_ok = gateway._dispatch(
        "store_memory", {"collection": "docs", "text": "abc", "metadata": {}}
    )
    assert store_ok == {"ok": True, "item": {"id": "m1"}}

    gateway.memory.search.return_value = [{"id": "m1", "score": 0.9}]
    search_ok = gateway._dispatch(
        "search_memory", {"collection": "docs", "query": "a", "top_k": 1}
    )
    assert search_ok == {"ok": True, "results": [{"id": "m1", "score": 0.9}]}

    unknown = gateway._dispatch("nope", {})
    assert unknown["ok"] is False
    assert unknown["error"]["code"] == "unknown_tool"


def test_dispatch_strict_schema_validation(tmp_path: Path):
    gateway = _gateway(tmp_path)

    missing = gateway._dispatch("run_shell", {})
    assert missing["ok"] is False
    assert missing["error"]["code"] == "bad_args"

    bad_type = gateway._dispatch("run_shell", {"command": "echo hi", "timeout_s": "10"})
    assert bad_type["ok"] is False
    assert bad_type["error"]["code"] == "bad_args"

    out_of_bounds = gateway._dispatch("search_memory", {"query": "q", "top_k": 1000})
    assert out_of_bounds["ok"] is False
    assert out_of_bounds["error"]["code"] == "bad_args"

    unknown_arg = gateway._dispatch("read_file", {"path": "seed.txt", "extra": True})
    assert unknown_arg["ok"] is False
    assert unknown_arg["error"]["code"] == "bad_args"
