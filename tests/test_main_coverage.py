from __future__ import annotations

from app.main import AppState


def test_health_and_not_found_branches(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"ok": True}

    missing_agent = client.get("/api/agents/missing")
    assert missing_agent.status_code == 404
    assert missing_agent.json()["error"] == "Agent not found"

    missing_run = client.get("/api/runs/missing")
    assert missing_run.status_code == 404
    assert missing_run.json()["error"] == "Run not found"

    missing_replay = client.get("/api/runs/missing/replay")
    assert missing_replay.status_code == 404

    missing_call = client.post(
        "/api/runs/missing/tools/call",
        json={"tool_name": "read_file", "args": {"path": "seed.txt"}},
    )
    assert missing_call.status_code == 404


def test_update_disable_and_create_run_missing_agent(client):
    update_missing = client.patch("/api/agents/missing", json={"role": "x"})
    assert update_missing.status_code == 404

    disable_missing = client.post("/api/agents/missing/disable")
    assert disable_missing.status_code == 404

    create_missing = client.post(
        "/api/runs",
        json={"agent_id": "missing", "task": "shell:echo hi", "step_limit": 3},
    )
    assert create_missing.status_code == 404


def test_create_run_accepts_unlimited_step_limit_zero(client):
    agent = client.post(
        "/api/agents",
        json={
            "name": "zero-limit-agent",
            "role": "ops",
            "model": "stub-v1",
            "tools_allowed": ["store_memory"],
        },
    ).json()

    created = client.post(
        "/api/runs",
        json={
            "agent_id": agent["id"],
            "task": "remember:notes:allow-zero-limit",
            "step_limit": 0,
        },
    )
    assert created.status_code == 200
    assert created.json()["step_limit"] == 0


def test_run_collection_endpoints_and_cancel_flow(client):
    agent = client.post(
        "/api/agents",
        json={
            "name": "collector",
            "role": "ops",
            "model": "stub-v1",
            "tools_allowed": ["store_memory"],
        },
    ).json()

    run = client.post(
        "/api/runs",
        json={
            "agent_id": agent["id"],
            "task": "remember:notes:hello",
            "step_limit": 4,
        },
    ).json()

    runs = client.get("/api/runs")
    assert runs.status_code == 200
    assert any(item["id"] == run["id"] for item in runs.json())

    steps = client.get(f"/api/runs/{run['id']}/steps")
    tool_calls = client.get(f"/api/runs/{run['id']}/tool-calls")
    model_calls = client.get(f"/api/runs/{run['id']}/model-calls")
    events = client.get(f"/api/runs/{run['id']}/events")
    assert steps.status_code == 200
    assert tool_calls.status_code == 200
    assert model_calls.status_code == 200
    assert events.status_code == 200

    canceled = client.post(f"/api/runs/{run['id']}/cancel")
    assert canceled.status_code == 200
    assert canceled.json()["status"] == "canceled"

    cancel_missing = client.post("/api/runs/missing/cancel")
    assert cancel_missing.status_code == 404


def test_run_input_endpoint_branches(client):
    missing = client.post("/api/runs/missing/input", json={"message": "hello"})
    assert missing.status_code == 404

    agent = client.post(
        "/api/agents",
        json={
            "name": "input-agent",
            "role": "ops",
            "model": "stub-v1",
            "tools_allowed": ["store_memory"],
        },
    ).json()
    run = client.post(
        "/api/runs",
        json={"agent_id": agent["id"], "task": "remember:notes:hello", "step_limit": 4},
    ).json()

    not_waiting = client.post(
        f"/api/runs/{run['id']}/input",
        json={"message": "extra context"},
    )
    assert not_waiting.status_code == 409

    paused = client.app.state.services.repo.create_run(
        agent_id=agent["id"],
        task="curl and analyze",
        step_limit=8,
    )
    client.app.state.services.repo.update_run_status(paused["id"], "awaiting_input")
    accepted = client.post(
        f"/api/runs/{paused['id']}/input",
        json={"message": "https://example.com"},
    )
    assert accepted.status_code == 200
    assert accepted.json()["status"] in {"running", "succeeded", "failed"}


def test_call_tool_agent_missing_branch(client, monkeypatch):
    agent = client.post(
        "/api/agents",
        json={
            "name": "runner",
            "role": "ops",
            "model": "stub-v1",
            "tools_allowed": ["read_file"],
        },
    ).json()
    run = client.post(
        "/api/runs",
        json={"agent_id": agent["id"], "task": "read:seed.txt", "step_limit": 3},
    ).json()

    original_get_agent = client.app.state.services.repo.get_agent

    def fake_get_agent(agent_id: str):
        if agent_id == agent["id"]:
            return None
        return original_get_agent(agent_id)

    monkeypatch.setattr(client.app.state.services.repo, "get_agent", fake_get_agent)

    called = client.post(
        f"/api/runs/{run['id']}/tools/call",
        json={"tool_name": "read_file", "args": {"path": "seed.txt"}},
    )
    assert called.status_code == 404
    assert called.json()["error"] == "Agent not found"


def test_ui_form_and_run_routes_full_branches(client):
    missing_model = client.post(
        "/agents",
        data={
            "name": "missing-model-agent",
            "role": "operator",
            "model": "",
            "tools": ["read_file"],
        },
        follow_redirects=False,
    )
    assert missing_model.status_code == 422
    assert "Name, role, and model are required" in missing_model.text

    created = client.post(
        "/agents",
        data={
            "name": "form-agent",
            "role": "operator",
            "model": "example-model-v1",
            "tools": ["read_file", "write_file"],
        },
        follow_redirects=False,
    )
    assert created.status_code == 303
    assert created.headers["location"] == "/agents"

    agents_page = client.get("/agents")
    assert agents_page.status_code == 200
    assert "form-agent" in agents_page.text

    no_task = client.post(
        "/runs",
        data={"agent_id": "", "task": "", "step_limit": "2"},
        follow_redirects=False,
    )
    assert no_task.status_code == 422
    assert "Agent is required" in no_task.text

    missing_id_with_agents = client.post(
        "/runs",
        data={"agent_id": "", "task": "shell:echo hi", "step_limit": "2"},
        follow_redirects=False,
    )
    assert missing_id_with_agents.status_code == 422
    assert "Agent is required" in missing_id_with_agents.text

    missing_task = client.post(
        "/runs",
        data={
            "agent_id": client.get("/api/agents").json()[0]["id"],
            "task": "",
            "step_limit": "2",
        },
        follow_redirects=False,
    )
    assert missing_task.status_code == 422
    assert "Task is required" in missing_task.text

    invalid_id = client.post(
        "/runs",
        data={"agent_id": "not-real", "task": "shell:echo hi", "step_limit": "2"},
        follow_redirects=False,
    )
    assert invalid_id.status_code == 422
    assert "Selected agent does not exist" in invalid_id.text

    valid_id = client.get("/api/agents").json()[0]["id"]
    ok = client.post(
        "/runs",
        data={"agent_id": valid_id, "task": "remember:notes:ui", "step_limit": "3"},
        follow_redirects=False,
    )
    assert ok.status_code == 303
    assert ok.headers["location"].startswith("/runs/")

    run_id = ok.headers["location"].rsplit("/", 1)[-1]
    detail = client.get(f"/runs/{run_id}")
    assert detail.status_code == 200
    assert "Model Calls" in detail.text

    missing_detail = client.get("/runs/does-not-exist")
    assert missing_detail.status_code == 404
    assert "Run not found" in missing_detail.text


def test_appstate_recovers_stale_running_runs_on_restart(tmp_path):
    db_path = tmp_path / "recover.db"
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True, exist_ok=True)

    first = AppState(db_path=str(db_path), workspace_root=str(workspace))
    agent = first.repo.create_agent(
        name="recover-agent",
        role="ops",
        model="stub-v1",
        tools_allowed=["store_memory"],
    )
    run = first.repo.create_run(
        agent_id=agent["id"], task="remember:notes:test", step_limit=4
    )
    first.repo.update_run_status(run["id"], "running")

    second = AppState(db_path=str(db_path), workspace_root=str(workspace))
    recovered = second.repo.get_run(run["id"])
    assert recovered is not None
    assert recovered["status"] == "failed"

    events = second.repo.list_events(run["id"])
    failed_events = [event for event in events if event.get("type") == "run.failed"]
    assert failed_events
    payload = failed_events[-1].get("payload_json") or {}
    assert payload.get("error", {}).get("code") == "interrupted_restart"
