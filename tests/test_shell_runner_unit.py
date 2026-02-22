from __future__ import annotations

import subprocess

import pytest

from app.shell_runner import ShellRunner


def test_safe_subdir_rejects_workspace_escape(tmp_path):
    runner = ShellRunner(workspace_root=str(tmp_path))

    with pytest.raises(ValueError, match="escapes workspace"):
        runner._safe_subdir("../outside", run_id=None)


def test_run_shell_rejects_invalid_timeout(tmp_path):
    runner = ShellRunner(workspace_root=str(tmp_path), default_timeout_s=20)

    result = runner.run_shell("echo hi", timeout_s=-1)

    assert result["ok"] is False
    assert result["error"]["code"] == "bad_config"
    assert result["error"]["details"]["timeout_s"] == -1


def test_run_shell_rejects_bad_write_subdir(tmp_path):
    runner = ShellRunner(workspace_root=str(tmp_path), default_timeout_s=20)

    result = runner.run_shell("echo hi", allow_write=True, write_subdir="../escape")

    assert result["ok"] is False
    assert result["error"]["code"] == "bad_config"
    assert "escapes workspace" in result["error"]["message"]
    assert result["command"] == "echo hi"


def test_run_shell_success_and_writable_subdir(tmp_path):
    runner = ShellRunner(workspace_root=str(tmp_path), default_timeout_s=20)

    result = runner.run_shell("printf hello", allow_write=True, run_id="r1")

    assert result["ok"] is True
    assert result["exit_code"] == 0
    assert result["stdout"] == "hello"
    assert result["working_dir"] == ".overmind_runs/r1"
    assert (tmp_path / ".overmind_runs" / "r1").exists()


def test_run_shell_timeout_returns_structured_error(tmp_path, monkeypatch):
    runner = ShellRunner(workspace_root=str(tmp_path), default_timeout_s=20)

    def _timeout(*args, **kwargs):
        del args, kwargs
        raise subprocess.TimeoutExpired(cmd="sh -lc sleep 5", timeout=1)

    monkeypatch.setattr(subprocess, "run", _timeout)

    result = runner.run_shell("sleep 5", timeout_s=1)

    assert result["ok"] is False
    assert result["error"]["code"] == "timeout"
    assert "timed out" in result["error"]["message"]
    assert result["error"]["details"]["timeout_s"] == 1
    assert result["working_dir"] == "."
