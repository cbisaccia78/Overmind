from __future__ import annotations

import time


def _wait_for_run(client, run_id: str, timeout_s: float = 5.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] in {"succeeded", "failed", "canceled"}:
            return run
        time.sleep(0.05)
    return client.get(f"/api/runs/{run_id}").json()


def test_orchestrator_step_accounting(client):
    agent = client.post(
        "/api/agents",
        json={
            "name": "mem-agent",
            "role": "memory",
            "model": "stub-v1",
            "tools_allowed": ["store_memory", "search_memory"],
        },
    ).json()

    run = client.post(
        "/api/runs",
        json={
            "agent_id": agent["id"],
            "task": "remember:notes:deterministic state machine",
            "step_limit": 5,
        },
    ).json()

    done = _wait_for_run(client, run["id"])
    assert done["status"] == "succeeded"

    steps = client.get(f"/api/runs/{run['id']}/steps").json()
    assert len(steps) == 3  # plan -> tool -> eval
    assert [s["idx"] for s in steps] == [0, 1, 2]

    model_calls = client.app.state.services.repo.list_model_calls(run["id"])
    assert len(model_calls) >= 1
    assert model_calls[0]["model"] == agent["model"]
