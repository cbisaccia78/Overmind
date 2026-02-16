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
