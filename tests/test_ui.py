from __future__ import annotations

import time
import json
from urllib import error as urlerror

import app.main as main_module


def test_dashboard_pages_render(client):
    home = client.get("/")
    assert home.status_code == 200

    agents = client.get("/agents")
    assert agents.status_code == 200
    assert 'type="checkbox" name="tools"' in agents.text

    runs = client.get("/runs")
    assert runs.status_code == 200


def test_agents_page_model_dropdown_uses_active_provider_models(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")

    class _Resp:
        def __init__(self, payload: dict):
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

        def read(self):
            return json.dumps(self._payload).encode("utf-8")

    def _urlopen(req, timeout=10):
        del timeout
        url = req.full_url
        if "api.openai.com" in url:
            return _Resp({"data": [{"id": "gpt-4.1-mini"}]})
        if "api.deepseek.com" in url:
            return _Resp({"data": [{"id": "deepseek-chat"}]})
        return _Resp({"data": []})

    monkeypatch.setattr(main_module.urlrequest, "urlopen", _urlopen)

    agents = client.get("/agents")
    assert agents.status_code == 200
    assert 'name="model"' in agents.text
    assert "OpenAI · gpt-4.1-mini" in agents.text
    assert "DeepSeek · deepseek-chat" in agents.text


def test_agents_page_model_dropdown_uses_deepseek_models_fallback_endpoint(
    client, monkeypatch
):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")

    class _Resp:
        def __init__(self, payload: dict):
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

        def read(self):
            return json.dumps(self._payload).encode("utf-8")

    def _urlopen(req, timeout=10):
        del timeout
        url = req.full_url
        if url.rstrip("/") == "https://api.deepseek.com/v1/models":
            raise urlerror.HTTPError(url=url, code=404, msg="not found", hdrs=None, fp=None)
        if url.rstrip("/") == "https://api.deepseek.com/models":
            return _Resp({"data": [{"id": "deepseek-chat"}]})
        return _Resp({"data": []})

    monkeypatch.setattr(main_module.urlrequest, "urlopen", _urlopen)

    agents = client.get("/agents")
    assert agents.status_code == 200
    assert "DeepSeek · deepseek-chat" in agents.text


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


def _wait_for_run(client, run_id: str, timeout_s: float = 5.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] in {"awaiting_input", "succeeded", "failed", "canceled"}:
            return run
        time.sleep(0.05)
    return client.get(f"/api/runs/{run_id}").json()


def test_run_detail_shows_awaiting_prompt_and_accepts_followup_input(client):
    agent = client.post(
        "/api/agents",
        json={
            "name": "ui-iterative-agent",
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

    detail = client.get(f"/runs/{run['id']}")
    assert detail.status_code == 200
    assert "Agent Needs Input" in detail.text
    assert "Continue Run" in detail.text
    assert "Please provide the full URL to fetch." in detail.text
    assert "Live Model Output" in detail.text
    assert "live-model-output" in detail.text

    empty = client.post(
        f"/runs/{run['id']}/input",
        data={"message": ""},
        follow_redirects=False,
    )
    assert empty.status_code == 422
    assert "Input is required." in empty.text

    resumed = client.post(
        f"/runs/{run['id']}/input",
        data={"message": "https://example.com"},
        follow_redirects=False,
    )
    assert resumed.status_code == 303
    assert resumed.headers["location"] == f"/runs/{run['id']}"


def test_run_detail_followup_input_rejected_when_not_awaiting(client):
    agent = client.post(
        "/api/agents",
        json={
            "name": "ui-basic-agent",
            "role": "operator",
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

    rejected = client.post(
        f"/runs/{run['id']}/input",
        data={"message": "continue"},
        follow_redirects=False,
    )
    assert rejected.status_code == 409
    assert "Run is not awaiting input." in rejected.text


def test_runs_page_shows_provide_input_action_for_awaiting_runs(client):
    repo = client.app.state.services.repo
    agent = repo.create_agent(
        name="ui-awaiting-agent",
        role="operator",
        model="stub-v1",
        tools_allowed=["store_memory"],
    )
    run = repo.create_run(
        agent_id=agent["id"],
        task="wait for user",
        step_limit=8,
    )
    repo.update_run_status(run["id"], "awaiting_input")

    page = client.get("/runs")
    assert page.status_code == 200
    assert "Provide Input" in page.text
    assert f"/runs/{run['id']}" in page.text


def test_runs_form_accepts_arbitrary_and_infinite_step_limit(client):
    agent = client.post(
        "/api/agents",
        json={
            "name": "ui-step-limit-agent",
            "role": "operator",
            "model": "stub-v1",
            "tools_allowed": ["store_memory"],
        },
    ).json()

    large_limit = client.post(
        "/runs",
        data={
            "agent_id": agent["id"],
            "task": "remember:notes:large-step-limit",
            "step_limit": "100000",
        },
        follow_redirects=False,
    )
    assert large_limit.status_code == 303
    large_run_id = large_limit.headers["location"].rsplit("/", 1)[-1]
    large_run = client.get(f"/api/runs/{large_run_id}").json()
    assert large_run["step_limit"] == 100000

    infinite_limit = client.post(
        "/runs",
        data={
            "agent_id": agent["id"],
            "task": "remember:notes:infinite-step-limit",
            "step_limit": "7",
            "step_limit_infinite": "1",
        },
        follow_redirects=False,
    )
    assert infinite_limit.status_code == 303
    infinite_run_id = infinite_limit.headers["location"].rsplit("/", 1)[-1]
    infinite_run = client.get(f"/api/runs/{infinite_run_id}").json()
    assert infinite_run["step_limit"] == 0
