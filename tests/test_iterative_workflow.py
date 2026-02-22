from __future__ import annotations

import time


def _wait_for_run(client, run_id: str, timeout_s: float = 5.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] in {"awaiting_input", "succeeded", "failed", "canceled"}:
            return run
        time.sleep(0.05)
    return client.get(f"/api/runs/{run_id}").json()


def test_iterative_pause_and_resume_with_user_input(client):
    agent = client.post(
        "/api/agents",
        json={
            "name": "iterative-agent",
            "role": "operator",
            "model": "stub-v1",
            "tools_allowed": ["run_shell", "store_memory"],
        },
    ).json()

    run = client.post(
        "/api/runs",
        json={
            "agent_id": agent["id"],
            "task": "curl a website, analyze results, and report",
            "step_limit": 8,
        },
    ).json()

    paused = _wait_for_run(client, run["id"])
    assert paused["status"] == "awaiting_input"

    resumed = client.post(
        f"/api/runs/{run['id']}/input",
        json={"message": "https://example.com"},
    )
    assert resumed.status_code == 200

    done = _wait_for_run(client, run["id"], timeout_s=10.0)
    assert done["status"] in {"succeeded", "failed"}

    events = client.get(f"/api/runs/{run['id']}/events").json()
    event_types = {event["type"] for event in events}
    assert "run.awaiting_input" in event_types
    assert "run.input_received" in event_types
