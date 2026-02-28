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


def test_replay_stream_returns_timeline(client):
    agent = client.post(
        "/api/agents",
        json={
            "name": "mem-agent-2",
            "role": "memory",
            "model": "stub-v1",
            "tools_allowed": ["store_memory"],
        },
    ).json()

    run = client.post(
        "/api/runs",
        json={
            "agent_id": agent["id"],
            "task": "remember:notes:timeline data",
            "step_limit": 5,
        },
    ).json()
    _wait_for_run(client, run["id"])

    replay = client.get(f"/api/runs/{run['id']}/replay")
    assert replay.status_code == 200
    assert "data:" in replay.text


def test_replay_follow_stream_emits_done_event(client):
    agent = client.post(
        "/api/agents",
        json={
            "name": "mem-agent-follow",
            "role": "memory",
            "model": "stub-v1",
            "tools_allowed": ["store_memory"],
        },
    ).json()

    run = client.post(
        "/api/runs",
        json={
            "agent_id": agent["id"],
            "task": "remember:notes:follow stream",
            "step_limit": 5,
        },
    ).json()
    _wait_for_run(client, run["id"])

    replay = client.get(f"/api/runs/{run['id']}/replay?follow=1&poll_ms=10")
    assert replay.status_code == 200
    assert "data:" in replay.text
    assert "event: done" in replay.text
