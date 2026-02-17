from __future__ import annotations


def test_tool_allowlist_is_enforced(client):
    agent = client.post(
        "/api/agents",
        json={
            "name": "read-only",
            "role": "analyst",
            "model": "stub-v1",
            "tools_allowed": ["read_file"],
        },
    ).json()

    run = client.post(
        "/api/runs",
        json={"agent_id": agent["id"], "task": "read:seed.txt", "step_limit": 3},
    ).json()

    denied = client.post(
        f"/api/runs/{run['id']}/tools/call",
        json={"tool_name": "write_file", "args": {"path": "x.txt", "content": "bad"}},
    )
    assert denied.status_code == 200
    body = denied.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "tool_not_allowed"


def test_tool_args_validation_is_enforced(client):
    agent = client.post(
        "/api/agents",
        json={
            "name": "runner",
            "role": "ops",
            "model": "stub-v1",
            "tools_allowed": ["run_shell"],
        },
    ).json()

    run = client.post(
        "/api/runs",
        json={"agent_id": agent["id"], "task": "shell:echo hi", "step_limit": 3},
    ).json()

    bad = client.post(
        f"/api/runs/{run['id']}/tools/call",
        json={
            "tool_name": "run_shell",
            "args": {"command": "echo hi", "timeout_s": "9"},
        },
    )
    assert bad.status_code == 200
    payload = bad.json()
    assert payload["ok"] is False
    assert payload["error"]["code"] == "bad_args"
