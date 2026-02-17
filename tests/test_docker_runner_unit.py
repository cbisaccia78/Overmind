from __future__ import annotations

import subprocess
from types import SimpleNamespace

from app.docker_runner import DockerRunner


def test_is_available_branches(monkeypatch, tmp_path):
    runner = DockerRunner(workspace_root=str(tmp_path))

    monkeypatch.setattr("app.docker_runner.shutil.which", lambda _: None)
    assert runner.is_available() is False

    monkeypatch.setattr("app.docker_runner.shutil.which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(
        "app.docker_runner.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr=""),
    )
    assert runner.is_available() is False

    monkeypatch.setattr(
        "app.docker_runner.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    assert runner.is_available() is True

    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["docker", "info"], timeout=3)

    monkeypatch.setattr("app.docker_runner.subprocess.run", _raise_timeout)
    assert runner.is_available() is False


def test_run_shell_unavailable(monkeypatch, tmp_path):
    runner = DockerRunner(workspace_root=str(tmp_path))
    monkeypatch.setattr(runner, "is_available", lambda: False)

    result = runner.run_shell("echo hi")
    assert result["ok"] is False
    assert result["error"]["code"] == "docker_unavailable"


def test_run_shell_success_and_writable_mount(monkeypatch, tmp_path):
    runner = DockerRunner(workspace_root=str(tmp_path))
    monkeypatch.setattr(runner, "is_available", lambda: True)

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("app.docker_runner.subprocess.run", fake_run)

    result = runner.run_shell(
        command="echo hi",
        timeout_s=3,
        allow_write=True,
        write_subdir="work",
    )
    assert result["ok"] is True
    assert result["exit_code"] == 0
    assert result["command"] == "echo hi"
    assert "<redacted-command>" in result["docker_cmd"]

    cmd = captured["cmd"]
    assert "--workdir" in cmd
    assert "/workspace_writable" in cmd
    assert any(":/workspace_writable:rw" in part for part in cmd)


def test_run_shell_blocks_escape_and_handles_timeout(monkeypatch, tmp_path):
    runner = DockerRunner(workspace_root=str(tmp_path))
    monkeypatch.setattr(runner, "is_available", lambda: True)

    captured = {}

    def fake_run_escape(cmd, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=1, stdout="", stderr="err")

    monkeypatch.setattr("app.docker_runner.subprocess.run", fake_run_escape)
    result = runner.run_shell(
        command="echo hi",
        allow_write=True,
        write_subdir="../../outside",
    )
    assert result["ok"] is False
    cmd = captured["cmd"]
    assert "/workspace_writable" not in cmd

    def fake_run_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["docker"], timeout=1)

    monkeypatch.setattr("app.docker_runner.subprocess.run", fake_run_timeout)
    timeout_result = runner.run_shell("sleep 3", timeout_s=1)
    assert timeout_result["ok"] is False
    assert timeout_result["error"]["code"] == "timeout"
    assert timeout_result["command"] == "sleep 3"
