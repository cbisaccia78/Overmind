from __future__ import annotations


def test_agent_registry_crud(client):
    create = client.post(
        "/api/agents",
        json={
            "name": "coder",
            "role": "dev",
            "model": "stub-v1",
            "tools_allowed": ["read_file"],
        },
    )
    assert create.status_code == 200
    agent = create.json()

    listed = client.get("/api/agents").json()
    assert any(a["id"] == agent["id"] for a in listed)

    got = client.get(f"/api/agents/{agent['id']}")
    assert got.status_code == 200
    assert got.json()["name"] == "coder"

    updated = client.patch(
        f"/api/agents/{agent['id']}",
        json={"role": "engineer", "tools_allowed": ["read_file", "write_file"]},
    )
    assert updated.status_code == 200
    assert updated.json()["role"] == "engineer"
    assert updated.json()["version"] == 2

    disabled = client.post(f"/api/agents/{agent['id']}/disable")
    assert disabled.status_code == 200
    assert disabled.json()["status"] == "disabled"
