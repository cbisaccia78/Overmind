from __future__ import annotations

import os
import shutil
import subprocess
import time

import pytest


RUN_DOCKER_TESTS = os.getenv("OVERMIND_RUN_DOCKER_TESTS") == "1"
DOCKER_AVAILABLE = shutil.which("docker") is not None


def _docker_usable() -> bool:
    if not DOCKER_AVAILABLE:
        return False
    try:
        probe = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


DOCKER_USABLE = _docker_usable() if RUN_DOCKER_TESTS else False

pytestmark = pytest.mark.skipif(
    not RUN_DOCKER_TESTS or not DOCKER_USABLE,
    reason="docker tests disabled or docker daemon unavailable/inaccessible",
)


def _wait_for_run(client, run_id: str, timeout_s: float = 15.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] in {"succeeded", "failed", "canceled"}:
            return run
        time.sleep(0.1)
    return client.get(f"/api/runs/{run_id}").json()


def test_end_to_end_docker_tool_call_and_logs(client):
    agent = client.post(
        "/api/agents",
        json={
            "name": "shell-agent",
            "role": "ops",
            "model": "stub-v1",
            "tools_allowed": ["run_shell"],
        },
    ).json()

    run = client.post(
        "/api/runs",
        json={
            "agent_id": agent["id"],
            "task": "shell:echo overmind",
            "step_limit": 6,
        },
    ).json()

    done = _wait_for_run(client, run["id"])
    assert done["status"] == "succeeded"

    tool_calls = client.get(f"/api/runs/{run['id']}/tool-calls").json()
    assert len(tool_calls) >= 1
    assert tool_calls[0]["tool_name"] == "run_shell"
    assert tool_calls[0]["allowed"] is True

    events = client.get(f"/api/runs/{run['id']}/events").json()
    event_types = {e["type"] for e in events}
    assert "tool.called" in event_types
    assert "run.succeeded" in event_types
