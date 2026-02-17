from __future__ import annotations


def test_dashboard_pages_render(client):
    home = client.get("/")
    assert home.status_code == 200

    agents = client.get("/agents")
    assert agents.status_code == 200

    runs = client.get("/runs")
    assert runs.status_code == 200


def test_runs_requires_agent_before_start(client):
    runs = client.get("/runs")
    assert runs.status_code == 200
    assert "Create at least one agent" in runs.text

    post = client.post(
        "/runs",
        data={"task": "shell:echo hi", "step_limit": "8"},
        follow_redirects=False,
    )
    assert post.status_code == 422
    assert "Create at least one agent before starting a run" in post.text
