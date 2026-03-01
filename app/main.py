"""FastAPI entrypoint for Overmind.

Defines API routes (agents, runs, memory, tool calls) and minimal HTML UI
pages. Wires together services via `AppState`.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import parse_qs

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .db import init_db, resolve_db_path
from .model_driven_policy import ModelDrivenPolicy
from .mcp_local import LocalMcpServerConfig
from .memory import LocalVectorMemory
from .model_gateway import ModelGateway
from .orchestrator import Orchestrator
from .policy import Policy
from .repository import Repository
from .shell_runner import ShellRunner
from .schemas import (
    AgentCreate,
    AgentUpdate,
    McpLocalServerRequest,
    MemorySearchRequest,
    MemoryStoreRequest,
    RunCreate,
    RunInputRequest,
    ToolCallRequest,
)
from .supervisor import Supervisor
from .tool_gateway import ToolGateway


WORKSPACE_ROOT = str(
    Path(os.getenv("OVERMIND_WORKSPACE", Path(__file__).resolve().parents[1])).resolve()
)


class AppState:
    """Container for application service singletons.

    Attributes:
        repo: SQLite-backed persistence layer.
        memory: Local memory subsystem (FTS-backed retrieval + optional embeddings).
        shell_runner: Host shell runner.
        model_gateway: Model inference facade that records audit telemetry.
        supervisor: Strategic controller that emits per-turn directives.
        policy: Tactical planner (model-driven by default).
        gateway: Tool execution gateway.
        orchestrator: Background run orchestrator.
    """

    def __init__(
        self,
        db_path: str | None = None,
        workspace_root: str | None = None,
        policy: Policy | None = None,
    ) -> None:
        """Initialize application services.

        Creates/initializes the database, then wires together the repository,
        memory, shell runner, tool gateway, and orchestrator.

        Args:
            db_path: Optional path to the SQLite database file.
            workspace_root: Optional workspace root used to constrain tools.
        """
        db_path = resolve_db_path(db_path)
        init_db(db_path)
        workspace = str(Path(workspace_root or WORKSPACE_ROOT).resolve())

        self.repo = Repository(db_path)
        self._recover_interrupted_runs()

        stored_openai_key = self.repo.get_setting("openai_api_key")
        if stored_openai_key and not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = stored_openai_key
        stored_deepseek_key = self.repo.get_setting("deepseek_api_key")
        if stored_deepseek_key and not os.getenv("DEEPSEEK_API_KEY"):
            os.environ["DEEPSEEK_API_KEY"] = stored_deepseek_key

        self.memory = LocalVectorMemory(self.repo)
        self.shell_runner = ShellRunner(workspace_root=workspace)
        self.gateway = ToolGateway(
            repo=self.repo,
            memory=self.memory,
            shell_runner=self.shell_runner,
            workspace_root=workspace,
        )
        self.model_gateway = ModelGateway(
            self.repo,
            openai_tools_provider=self.gateway.list_openai_tools_with_aliases,
        )
        self.supervisor = Supervisor(model_gateway=self.model_gateway)
        self.policy = policy or ModelDrivenPolicy(model_gateway=self.model_gateway)
        self.orchestrator = Orchestrator(
            repo=self.repo,
            tool_gateway=self.gateway,
            policy=self.policy,
            supervisor=self.supervisor,
        )

    def _recover_interrupted_runs(self) -> None:
        """Mark stale in-progress runs as failed after process restart."""
        for run in self.repo.list_runs():
            if run.get("status") != "running":
                continue
            run_id = str(run.get("id") or "")
            if not run_id:
                continue
            self.repo.update_run_status(run_id, "failed")
            self.repo.create_event(
                run_id,
                "run.failed",
                {
                    "run_id": run_id,
                    "error": {
                        "code": "interrupted_restart",
                        "message": "run was in progress when the app restarted",
                    },
                },
            )


def create_app(
    db_path: str | None = None,
    workspace_root: str | None = None,
    policy: Policy | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        db_path: Optional path to the SQLite database file.
        workspace_root: Optional workspace root used to constrain tools.

    Returns:
        Configured `FastAPI` application.
    """
    application = FastAPI(title="Overmind", version="0.1.0")
    application.state.services = AppState(
        db_path=db_path,
        workspace_root=workspace_root,
        policy=policy,
    )
    application.mount(
        "/static",
        StaticFiles(directory=str(Path(__file__).parent / "static")),
        name="static",
    )
    return application


app = create_app()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _services() -> AppState:
    """Return the singleton `AppState` stored on the FastAPI app."""
    return app.state.services


def _available_tools() -> list[str]:
    """Return currently available tools, including discovered MCP tools."""
    return _services().gateway.list_tool_names()


def _mask_api_key(key: str | None) -> str:
    """Return a masked display value for API keys."""
    if not key:
        return ""
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"


def _mask_openai_key(key: str | None) -> str:
    """Backward-compatible wrapper for OpenAI key masking."""
    return _mask_api_key(key)


def _test_api_key(
    *, api_key: str, endpoint: str, provider_name: str
) -> tuple[bool, str]:
    """Validate provider API key by calling the models endpoint."""
    req = urlrequest.Request(
        endpoint,
        headers={
            "Authorization": f"Bearer {api_key}",
        },
        method="GET",
    )
    try:
        with urlrequest.urlopen(req, timeout=10):
            return True, f"{provider_name} key is valid."
    except urlerror.HTTPError as exc:
        return False, f"{provider_name} key test failed: HTTP {exc.code}"
    except (urlerror.URLError, TimeoutError) as exc:
        return False, f"{provider_name} key test failed: {exc}"


def _test_api_key_endpoints(
    *, api_key: str, endpoints: list[str], provider_name: str
) -> tuple[bool, str]:
    """Validate API key against one or more endpoints."""
    last_message = f"{provider_name} key test failed: no endpoint configured"
    for endpoint in endpoints:
        ok, message = _test_api_key(
            api_key=api_key,
            endpoint=endpoint,
            provider_name=provider_name,
        )
        if ok:
            return ok, message
        last_message = message
    return False, last_message


def _test_openai_key(api_key: str) -> tuple[bool, str]:
    """Validate OpenAI key by calling the models endpoint."""
    return _test_api_key(
        api_key=api_key,
        endpoint=os.getenv(
            "OVERMIND_OPENAI_MODELS_URL", "https://api.openai.com/v1/models"
        ),
        provider_name="OpenAI",
    )


def _test_deepseek_key(api_key: str) -> tuple[bool, str]:
    """Validate DeepSeek key by calling the models endpoint."""
    return _test_api_key_endpoints(
        api_key=api_key,
        endpoints=_deepseek_model_endpoints(),
        provider_name="DeepSeek",
    )


def _deepseek_model_endpoints() -> list[str]:
    """Return DeepSeek model-discovery endpoints in preference order."""
    configured = str(os.getenv("OVERMIND_DEEPSEEK_MODELS_URL") or "").strip()
    if configured:
        return [configured]
    return [
        "https://api.deepseek.com/v1/models",
        "https://api.deepseek.com/models",
    ]


def _extract_model_ids(payload: Any) -> list[str]:
    """Extract model identifiers from provider model list payloads."""
    data: Any = payload
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            data = payload.get("data")
        elif isinstance(payload.get("models"), list):
            data = payload.get("models")
        else:
            data = []

    if not isinstance(data, list):
        return []

    models: list[str] = []
    for item in data:
        if isinstance(item, str):
            model_id = item.strip()
        elif isinstance(item, dict):
            model_id = str(
                item.get("id") or item.get("model") or item.get("name") or ""
            ).strip()
        else:
            model_id = ""
        if model_id:
            models.append(model_id)
    return sorted(set(models))


def _fetch_provider_models_from_endpoints(*, api_key: str, endpoints: list[str]) -> list[str]:
    """Fetch model IDs from one or more provider endpoints."""
    discovered: list[str] = []
    for endpoint in endpoints:
        discovered.extend(_fetch_provider_models(api_key=api_key, endpoint=endpoint))
    return sorted(set(discovered))


def _fetch_provider_models(*, api_key: str, endpoint: str) -> list[str]:
    """Fetch model IDs from an OpenAI-compatible /models endpoint."""
    req = urlrequest.Request(
        endpoint,
        headers={
            "Authorization": f"Bearer {api_key}",
        },
        method="GET",
    )
    try:
        with urlrequest.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urlerror.HTTPError, urlerror.URLError, TimeoutError, json.JSONDecodeError):
        return []

    return _extract_model_ids(payload)


def _available_remote_models() -> list[dict[str, str]]:
    """Return available model options from all configured remote providers."""
    options: list[dict[str, str]] = [
        {
            "provider": "local",
            "id": "stub-v1",
            "label": "Local · stub-v1",
        }
    ]

    openai_key = os.getenv("OPENAI_API_KEY") or _services().repo.get_setting(
        "openai_api_key"
    )
    if openai_key:
        for model_id in _fetch_provider_models(
            api_key=openai_key,
            endpoint=os.getenv(
                "OVERMIND_OPENAI_MODELS_URL", "https://api.openai.com/v1/models"
            ),
        ):
            options.append(
                {"provider": "openai", "id": model_id, "label": f"OpenAI · {model_id}"}
            )

    deepseek_key = os.getenv("DEEPSEEK_API_KEY") or _services().repo.get_setting(
        "deepseek_api_key"
    )
    if deepseek_key:
        for model_id in _fetch_provider_models_from_endpoints(
            api_key=deepseek_key,
            endpoints=_deepseek_model_endpoints(),
        ):
            options.append(
                {
                    "provider": "deepseek",
                    "id": model_id,
                    "label": f"DeepSeek · {model_id}",
                }
            )

    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, str]] = []
    for item in options:
        key = (item["provider"], item["id"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _provider_key_status(
    *, setting_key: str, env_key: str
) -> tuple[bool, str | None]:
    """Return (configured, masked) status for a provider API key."""
    stored = _services().repo.get_setting(setting_key)
    active = os.getenv(env_key)
    key = active or stored
    return bool(key), _mask_api_key(key) if key else None


async def _parse_urlencoded_form(request: Request) -> dict[str, str]:
    """Parse a URL-encoded form body into a string dictionary.

    Args:
        request: Incoming request.

    Returns:
        Mapping of form field name to last value.
    """
    body = (await request.body()).decode("utf-8")
    parsed = parse_qs(body, keep_blank_values=True)
    return {key: values[-1] if values else "" for key, values in parsed.items()}


async def _parse_urlencoded_form_multi(request: Request) -> dict[str, list[str]]:
    """Parse a URL-encoded form body into a multi-value dictionary.

    Args:
        request: Incoming request.

    Returns:
        Mapping of field name to all submitted values for that field.
    """
    body = (await request.body()).decode("utf-8")
    return parse_qs(body, keep_blank_values=True)


def _latest_awaiting_prompt(
    events: list[dict[str, Any]],
    steps: list[dict[str, Any]],
) -> str:
    """Return the most recent user-facing prompt for an awaiting-input run."""
    for event in reversed(events):
        if str(event.get("type") or "") != "run.awaiting_input":
            continue
        payload = dict(event.get("payload_json") or {})
        prompt = str(payload.get("prompt") or "").strip()
        if prompt:
            return prompt

    for step in reversed(steps):
        if str(step.get("type") or "") != "ask_user":
            continue
        step_input = dict(step.get("input_json") or {})
        prompt = str(step_input.get("prompt") or "").strip()
        if prompt:
            return prompt

    return "The agent is waiting for your input."


def _run_detail_context(
    run_id: str,
    *,
    run_input_error: str | None = None,
    run_input_value: str = "",
) -> dict[str, Any] | None:
    """Build the template context for the run detail page."""
    repo = _services().repo
    run = repo.get_run(run_id)
    if not run:
        return None

    steps = repo.list_steps(run_id)
    tool_calls = repo.list_tool_calls(run_id)
    model_calls = repo.list_model_calls(run_id)
    events = repo.list_events(run_id)
    awaiting_prompt = (
        _latest_awaiting_prompt(events, steps)
        if run.get("status") == "awaiting_input"
        else ""
    )

    return {
        "run": run,
        "steps": steps,
        "tool_calls": tool_calls,
        "model_calls": model_calls,
        "events": events,
        "awaiting_prompt": awaiting_prompt,
        "run_input_error": run_input_error,
        "run_input_value": run_input_value,
    }


def _build_run_timeline(repo: Repository, run_id: str) -> list[dict[str, Any]]:
    """Build a merged run timeline across steps/tool calls/model calls/events."""
    steps = repo.list_steps(run_id)
    tool_calls = repo.list_tool_calls(run_id)
    model_calls = repo.list_model_calls(run_id)
    events = repo.list_events(run_id)

    timeline: list[dict[str, Any]] = []
    for item in steps:
        timeline.append({"kind": "step", "ts": item.get("started_at"), "data": item})
    for item in tool_calls:
        timeline.append(
            {"kind": "tool_call", "ts": item.get("created_at"), "data": item}
        )
    for item in model_calls:
        timeline.append(
            {"kind": "model_call", "ts": item.get("created_at"), "data": item}
        )
    for item in events:
        timeline.append({"kind": "event", "ts": item.get("ts"), "data": item})

    timeline.sort(
        key=lambda x: (
            x.get("ts") or "",
            str(x.get("kind") or ""),
            str((x.get("data") or {}).get("id") or ""),
        )
    )
    return timeline


def _timeline_item_key(item: dict[str, Any]) -> str:
    """Return a stable dedupe key for a timeline item."""
    kind = str(item.get("kind") or "")
    data = item.get("data")
    if isinstance(data, dict):
        item_id = str(data.get("id") or "").strip()
        if item_id:
            return f"{kind}:{item_id}"
    encoded = json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"{kind}:{encoded}"


def _settings_context(
    *,
    settings_error: str | None = None,
    settings_message: str | None = None,
) -> dict[str, Any]:
    """Build settings page context with provider key statuses and MCP servers."""
    openai_status = get_openai_key_status()
    deepseek_status = get_deepseek_key_status()
    mcp_servers = list_mcp_local_servers()["servers"]
    return {
        "openai_configured": openai_status["configured"],
        "openai_masked": openai_status["masked"],
        "deepseek_configured": deepseek_status["configured"],
        "deepseek_masked": deepseek_status["masked"],
        "mcp_servers": mcp_servers,
        "settings_error": settings_error,
        "settings_message": settings_message,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    """Health check endpoint.

    Returns:
        Simple JSON payload indicating the service is up.
    """
    return {"ok": True}


# Agent Registry API
@app.post("/api/agents")
def create_agent(payload: AgentCreate) -> dict[str, Any]:
    """Create an agent.

    Args:
        payload: Agent creation payload.

    Returns:
        Created agent row.
    """
    row = _services().repo.create_agent(
        name=payload.name,
        role=payload.role,
        model=payload.model,
        tools_allowed=payload.tools_allowed,
    )
    return row


@app.get("/api/agents")
def list_agents() -> list[dict[str, Any]]:
    """List all agents.

    Returns:
        List of agent rows.
    """
    return _services().repo.list_agents()


@app.get("/api/agents/{agent_id}")
def get_agent(agent_id: str) -> dict[str, Any]:
    """Fetch a single agent.

    Args:
        agent_id: Agent ID.

    Returns:
        Agent row.

    Raises:
        HTTPException: If the agent does not exist.
    """
    row = _services().repo.get_agent(agent_id)
    if not row:
        raise HTTPException(status_code=404, detail="Agent not found")
    return row


@app.patch("/api/agents/{agent_id}")
def update_agent(agent_id: str, payload: AgentUpdate) -> dict[str, Any]:
    """Update an agent.

    Args:
        agent_id: Agent ID.
        payload: Partial update payload.

    Returns:
        Updated agent row.

    Raises:
        HTTPException: If the agent does not exist.
    """
    updates = payload.model_dump(exclude_none=True)
    row = _services().repo.update_agent(agent_id, updates)
    if not row:
        raise HTTPException(status_code=404, detail="Agent not found")
    return row


@app.post("/api/agents/{agent_id}/disable")
def disable_agent(agent_id: str) -> dict[str, Any]:
    """Disable an agent.

    Args:
        agent_id: Agent ID.

    Returns:
        Updated agent row.

    Raises:
        HTTPException: If the agent does not exist.
    """
    row = _services().repo.disable_agent(agent_id)
    if not row:
        raise HTTPException(status_code=404, detail="Agent not found")
    return row


# Runs API
@app.post("/api/runs")
def create_run(payload: RunCreate) -> dict[str, Any]:
    """Create a run and launch execution.

    Args:
        payload: Run creation payload.

    Returns:
        Created run row.

    Raises:
        HTTPException: If the referenced agent does not exist.
    """
    repo = _services().repo
    agent = repo.get_agent(payload.agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    run = repo.create_run(
        agent_id=payload.agent_id, task=payload.task, step_limit=payload.step_limit
    )
    repo.create_event(
        run["id"], "run.queued", {"run_id": run["id"], "task": payload.task}
    )
    _services().orchestrator.launch(run["id"])
    return run


@app.get("/api/runs")
def list_runs() -> list[dict[str, Any]]:
    """List all runs.

    Returns:
        List of run rows.
    """
    return _services().repo.list_runs()


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    """Fetch a single run.

    Args:
        run_id: Run ID.

    Returns:
        Run row.

    Raises:
        HTTPException: If the run does not exist.
    """
    run = _services().repo.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> dict[str, Any]:
    """Cancel a run.

    Args:
        run_id: Run ID.

    Returns:
        Updated run row.

    Raises:
        HTTPException: If the run does not exist.
    """
    repo = _services().repo
    run = repo.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    repo.update_run_status(run_id, "canceled")
    repo.create_event(run_id, "run.canceled", {"run_id": run_id})
    return repo.get_run(run_id)  # type: ignore[return-value]


@app.post("/api/runs/{run_id}/input")
def provide_run_input(run_id: str, payload: RunInputRequest) -> dict[str, Any]:
    """Provide user input for a run paused in awaiting_input state.

    Args:
        run_id: Run ID.
        payload: User input payload.

    Returns:
        Updated run row.

    Raises:
        HTTPException: If the run does not exist or is not awaiting input.
    """
    repo = _services().repo
    run = repo.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.get("status") != "awaiting_input":
        raise HTTPException(status_code=409, detail="Run is not awaiting input")

    repo.append_run_task(run_id, payload.message)
    repo.create_event(run_id, "run.input_received", {"run_id": run_id})
    repo.update_run_status(run_id, "running")
    _services().orchestrator.launch(run_id)
    updated = repo.get_run(run_id)
    if not updated:
        raise HTTPException(status_code=404, detail="Run not found")
    return updated


@app.get("/api/runs/{run_id}/steps")
def get_run_steps(run_id: str) -> list[dict[str, Any]]:
    """List steps for a run.

    Args:
        run_id: Run ID.

    Returns:
        List of step rows.
    """
    return _services().repo.list_steps(run_id)


@app.get("/api/runs/{run_id}/tool-calls")
def get_run_tool_calls(run_id: str) -> list[dict[str, Any]]:
    """List tool calls for a run.

    Args:
        run_id: Run ID.

    Returns:
        List of tool call rows.
    """
    return _services().repo.list_tool_calls(run_id)


@app.get("/api/runs/{run_id}/model-calls")
def get_run_model_calls(run_id: str) -> list[dict[str, Any]]:
    """List model calls for a run.

    Args:
        run_id: Run ID.

    Returns:
        List of model call rows.
    """
    return _services().repo.list_model_calls(run_id)


@app.get("/api/runs/{run_id}/events")
def get_run_events(run_id: str) -> list[dict[str, Any]]:
    """List events for a run.

    Args:
        run_id: Run ID.

    Returns:
        List of event rows.
    """
    return _services().repo.list_events(run_id)


@app.get("/api/runs/{run_id}/replay")
def replay_run(run_id: str, follow: bool = False, poll_ms: int = 300) -> StreamingResponse:
    """Stream a run timeline as Server-Sent Events (SSE).

    Args:
        run_id: Run ID.
        follow: When true, keep streaming new timeline items until the run reaches
            a terminal status.
        poll_ms: Poll interval for follow mode.

    Returns:
        Streaming SSE response where each event contains a JSON timeline item.

    Raises:
        HTTPException: If the run does not exist.
    """
    repo = _services().repo
    run = repo.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    poll_seconds = max(0.05, min(5.0, float(poll_ms) / 1000.0))
    terminal_statuses = {"succeeded", "failed", "canceled"}

    def _stream() -> Any:
        """Yield SSE lines for timeline items.

        Yields:
            SSE-formatted `data:` lines.
        """
        if not follow:
            for item in _build_run_timeline(repo, run_id):
                yield f"data: {json.dumps(item)}\n\n"
            return

        seen_keys: set[str] = set()
        while True:
            emitted = False
            for item in _build_run_timeline(repo, run_id):
                key = _timeline_item_key(item)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                emitted = True
                yield f"data: {json.dumps(item)}\n\n"

            latest = repo.get_run(run_id)
            status = str((latest or {}).get("status") or "")
            if status in terminal_statuses and not emitted:
                yield (
                    "event: done\n"
                    f"data: {json.dumps({'run_id': run_id, 'status': status})}\n\n"
                )
                break

            if not emitted:
                yield ": keepalive\n\n"
            time.sleep(poll_seconds)

    return StreamingResponse(_stream(), media_type="text/event-stream")


@app.post("/api/runs/{run_id}/tools/call")
def call_tool(run_id: str, payload: ToolCallRequest) -> dict[str, Any]:
    """Invoke an allowed tool for the run's agent.

    Args:
        run_id: Run ID.
        payload: Tool call request payload.

    Returns:
        Tool result dict.

    Raises:
        HTTPException: If the run or agent does not exist.
    """
    repo = _services().repo
    run = repo.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    agent = repo.get_agent(run["agent_id"])
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return _services().gateway.call(
        run_id=run_id,
        step_id=None,
        agent=agent,
        tool_name=payload.tool_name,
        args=payload.args,
    )


# Memory API
@app.post("/api/memory/store")
def store_memory(payload: MemoryStoreRequest) -> dict[str, Any]:
    """Store a memory item.

    Args:
        payload: Memory store request.

    Returns:
        JSON payload containing the stored item.
    """
    item = _services().memory.store(payload.text, payload.collection, payload.metadata)
    return {"ok": True, "item": item}


@app.post("/api/memory/search")
def search_memory(payload: MemorySearchRequest) -> dict[str, Any]:
    """Search memory items.

    Args:
        payload: Memory search request.

    Returns:
        JSON payload containing enriched results.
    """
    results = _services().memory.search(
        payload.query, payload.collection, payload.top_k
    )
    return {"ok": True, "results": results}


@app.get("/api/settings/openai-key")
def get_openai_key_status() -> dict[str, Any]:
    """Return whether an OpenAI API key is configured."""
    configured, masked = _provider_key_status(
        setting_key="openai_api_key", env_key="OPENAI_API_KEY"
    )
    return {
        "ok": True,
        "configured": configured,
        "masked": masked or "",
    }


@app.get("/api/settings/deepseek-key")
def get_deepseek_key_status() -> dict[str, Any]:
    """Return whether a DeepSeek API key is configured."""
    configured, masked = _provider_key_status(
        setting_key="deepseek_api_key", env_key="DEEPSEEK_API_KEY"
    )
    return {
        "ok": True,
        "configured": configured,
        "masked": masked or "",
    }


@app.get("/api/settings/mcp/servers")
def list_mcp_local_servers() -> dict[str, Any]:
    """List configured local MCP servers."""
    servers = _services().gateway.list_local_mcp_servers()
    return {
        "ok": True,
        "servers": [
            {
                "id": server.id,
                "command": server.command,
                "args": server.args,
                "env": server.env,
                "enabled": server.enabled,
            }
            for server in servers
        ],
    }


@app.post("/api/settings/mcp/servers")
def upsert_mcp_local_server(payload: McpLocalServerRequest) -> dict[str, Any]:
    """Create or replace a local MCP server config."""
    _services().gateway.upsert_local_mcp_server(
        config=LocalMcpServerConfig(
            id=payload.id.strip(),
            command=payload.command.strip(),
            args=[arg.strip() for arg in payload.args if arg.strip()],
            env={str(k): str(v) for k, v in payload.env.items()},
            enabled=payload.enabled,
        )
    )
    return list_mcp_local_servers()


@app.delete("/api/settings/mcp/servers/{server_id}")
def delete_mcp_local_server(server_id: str) -> dict[str, Any]:
    """Delete one local MCP server config."""
    removed = _services().gateway.remove_local_mcp_server(server_id.strip())
    if not removed:
        raise HTTPException(status_code=404, detail="MCP server not found")
    return list_mcp_local_servers()


# Basic UI
@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    """Render the home page.

    Args:
        request: Incoming request.

    Returns:
        HTML response.
    """
    repo = _services().repo
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "agents_count": len(repo.list_agents()),
            "runs_count": len(repo.list_runs()),
            "shell_available": _services().shell_runner.is_available(),
        },
    )


@app.get("/agents", response_class=HTMLResponse)
def agents_page(request: Request) -> HTMLResponse:
    """Render the agents page.

    Args:
        request: Incoming request.

    Returns:
        HTML response.
    """
    available_tools = _available_tools()
    available_models = _available_remote_models()
    return templates.TemplateResponse(
        request,
        "agents.html",
        {
            "agents": _services().repo.list_agents(),
            "available_tools": available_tools,
            "available_models": available_models,
            "form_error": None,
            "form_values": {
                "name": "",
                "role": "operator",
                "model": "",
                "tools": available_tools,
            },
        },
    )


@app.post("/agents", response_class=HTMLResponse)
async def create_agent_form(request: Request) -> Response:
    """Handle agent creation from the HTML form.

    Args:
        request: Incoming request.

    Returns:
        Redirect response to the agents page.
    """
    form = await _parse_urlencoded_form_multi(request)
    name = (form.get("name", [""])[-1]).strip()
    role = (form.get("role", [""])[-1]).strip()
    model = (form.get("model", [""])[-1]).strip()
    tools = [tool.strip() for tool in form.get("tools", []) if tool.strip()]

    available_tools = _available_tools()
    available_models = _available_remote_models()

    if not name or not role or not model:
        return templates.TemplateResponse(
            request,
            "agents.html",
            {
                "agents": _services().repo.list_agents(),
                "available_tools": available_tools,
                "available_models": available_models,
                "form_error": "Name, role, and model are required.",
                "form_values": {
                    "name": name,
                    "role": role,
                    "model": model,
                    "tools": tools,
                },
            },
            status_code=422,
        )

    _services().repo.create_agent(
        name=name,
        role=role,
        model=model,
        tools_allowed=tools,
    )
    return RedirectResponse(url="/agents", status_code=303)


@app.get("/runs", response_class=HTMLResponse)
def runs_page(request: Request) -> HTMLResponse:
    """Render the runs list page.

    Args:
        request: Incoming request.

    Returns:
        HTML response.
    """
    repo = _services().repo
    return templates.TemplateResponse(
        request,
        "runs.html",
        {
            "runs": repo.list_runs(),
            "agents": repo.list_agents(),
            "run_error": None,
            "run_form_values": {
                "agent_id": "",
                "task": "shell:echo hello overmind",
                "step_limit": "8",
                "step_limit_infinite": False,
            },
        },
    )


@app.post("/runs", response_class=HTMLResponse)
async def create_run_form(request: Request) -> Response:
    """Handle run creation from the HTML form.

    Args:
        request: Incoming request.

    Returns:
        Redirect response to the run detail page (or other UI page).
    """
    form = await _parse_urlencoded_form(request)
    repo = _services().repo
    agent_id = form.get("agent_id", "").strip()
    task = form.get("task", "").strip()
    step_limit_raw = form.get("step_limit", "8").strip() or "8"
    step_limit_infinite = form.get("step_limit_infinite", "").strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }
    run_form_values = {
        "agent_id": agent_id,
        "task": task,
        "step_limit": step_limit_raw,
        "step_limit_infinite": step_limit_infinite,
    }

    agents = repo.list_agents()
    if not agents:
        return templates.TemplateResponse(
            request,
            "runs.html",
            {
                "runs": repo.list_runs(),
                "agents": agents,
                "run_error": "Create at least one agent before starting a run.",
                "run_form_values": run_form_values,
            },
            status_code=422,
        )

    if not agent_id:
        return templates.TemplateResponse(
            request,
            "runs.html",
            {
                "runs": repo.list_runs(),
                "agents": agents,
                "run_error": "Agent is required.",
                "run_form_values": run_form_values,
            },
            status_code=422,
        )

    if not task:
        return templates.TemplateResponse(
            request,
            "runs.html",
            {
                "runs": repo.list_runs(),
                "agents": agents,
                "run_error": "Task is required.",
                "run_form_values": run_form_values,
            },
            status_code=422,
        )

    agent = repo.get_agent(agent_id)
    if not agent:
        return templates.TemplateResponse(
            request,
            "runs.html",
            {
                "runs": repo.list_runs(),
                "agents": agents,
                "run_error": "Selected agent does not exist.",
                "run_form_values": run_form_values,
            },
            status_code=422,
        )

    if step_limit_infinite:
        step_limit = 0
    else:
        try:
            step_limit = int(step_limit_raw)
        except ValueError:
            return templates.TemplateResponse(
                request,
                "runs.html",
                {
                    "runs": repo.list_runs(),
                    "agents": agents,
                    "run_error": "Step limit must be a positive integer, or enable infinity.",
                    "run_form_values": run_form_values,
                },
                status_code=422,
            )

        if step_limit < 1:
            return templates.TemplateResponse(
                request,
                "runs.html",
                {
                    "runs": repo.list_runs(),
                    "agents": agents,
                    "run_error": "Step limit must be at least 1, or enable infinity.",
                    "run_form_values": run_form_values,
                },
                status_code=422,
            )

    run = repo.create_run(
        agent_id=agent_id,
        task=task,
        step_limit=step_limit,
    )
    repo.create_event(run["id"], "run.queued", {"run_id": run["id"], "task": task})
    _services().orchestrator.launch(run["id"])
    return RedirectResponse(url=f"/runs/{run['id']}", status_code=303)


@app.get("/runs/{run_id}", response_class=HTMLResponse)
def run_detail_page(run_id: str, request: Request) -> HTMLResponse:
    """Render the run detail page.

    Args:
        run_id: Run ID.
        request: Incoming request.

    Returns:
        HTML response.
    """
    context = _run_detail_context(run_id)
    if context is None:
        return HTMLResponse("Run not found", status_code=404)
    return templates.TemplateResponse(request, "run_detail.html", context)


@app.post("/runs/{run_id}/input", response_class=HTMLResponse)
async def provide_run_input_form(run_id: str, request: Request) -> Response:
    """Handle run follow-up input from the run detail HTML page."""
    form = await _parse_urlencoded_form(request)
    message = form.get("message", "").strip()

    context = _run_detail_context(
        run_id,
        run_input_value=message,
    )
    if context is None:
        return HTMLResponse("Run not found", status_code=404)

    run = context["run"]
    if run.get("status") != "awaiting_input":
        context["run_input_error"] = "Run is not awaiting input."
        return templates.TemplateResponse(
            request,
            "run_detail.html",
            context,
            status_code=409,
        )

    if not message:
        context["run_input_error"] = "Input is required."
        return templates.TemplateResponse(
            request,
            "run_detail.html",
            context,
            status_code=422,
        )

    repo = _services().repo
    repo.append_run_task(run_id, message)
    repo.create_event(run_id, "run.input_received", {"run_id": run_id})
    repo.update_run_status(run_id, "running")
    _services().orchestrator.launch(run_id)
    return RedirectResponse(url=f"/runs/{run_id}", status_code=303)


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request) -> HTMLResponse:
    """Render application settings page."""
    return templates.TemplateResponse(
        request,
        "settings.html",
        _settings_context(),
    )


@app.post("/settings/openai", response_class=HTMLResponse)
async def settings_openai_form(request: Request) -> HTMLResponse:
    """Handle OpenAI key save/clear/test actions from settings page."""
    form = await _parse_urlencoded_form(request)
    action = (form.get("action") or "save").strip().lower()
    key_input = (form.get("openai_api_key") or "").strip()
    repo = _services().repo

    stored = repo.get_setting("openai_api_key")
    active = os.getenv("OPENAI_API_KEY")
    effective_key = key_input or active or stored

    settings_error: str | None = None
    settings_message: str | None = None

    if action == "test":
        if not effective_key:
            settings_error = "No OpenAI API key is configured."
        else:
            ok, message = _test_openai_key(effective_key)
            if ok:
                settings_message = message
            else:
                settings_error = message
    elif action == "clear":
        repo.set_setting("openai_api_key", None)
        os.environ.pop("OPENAI_API_KEY", None)
        _services().memory.reload_backend()
        settings_message = "OpenAI API key removed."
    else:
        if not key_input:
            settings_error = "OpenAI API key is required to save."
        else:
            repo.set_setting("openai_api_key", key_input)
            os.environ["OPENAI_API_KEY"] = key_input
            _services().memory.reload_backend()
            settings_message = "OpenAI API key saved."

    return templates.TemplateResponse(
        request,
        "settings.html",
        _settings_context(
            settings_error=settings_error,
            settings_message=settings_message,
        ),
        status_code=422 if settings_error else 200,
    )


@app.post("/settings/deepseek", response_class=HTMLResponse)
async def settings_deepseek_form(request: Request) -> HTMLResponse:
    """Handle DeepSeek key save/clear/test actions from settings page."""
    form = await _parse_urlencoded_form(request)
    action = (form.get("action") or "save").strip().lower()
    key_input = (form.get("deepseek_api_key") or "").strip()
    repo = _services().repo

    stored = repo.get_setting("deepseek_api_key")
    active = os.getenv("DEEPSEEK_API_KEY")
    effective_key = key_input or active or stored

    settings_error: str | None = None
    settings_message: str | None = None

    if action == "test":
        if not effective_key:
            settings_error = "No DeepSeek API key is configured."
        else:
            ok, message = _test_deepseek_key(effective_key)
            if ok:
                settings_message = message
            else:
                settings_error = message
    elif action == "clear":
        repo.set_setting("deepseek_api_key", None)
        os.environ.pop("DEEPSEEK_API_KEY", None)
        settings_message = "DeepSeek API key removed."
    else:
        if not key_input:
            settings_error = "DeepSeek API key is required to save."
        else:
            repo.set_setting("deepseek_api_key", key_input)
            os.environ["DEEPSEEK_API_KEY"] = key_input
            settings_message = "DeepSeek API key saved."

    return templates.TemplateResponse(
        request,
        "settings.html",
        _settings_context(
            settings_error=settings_error,
            settings_message=settings_message,
        ),
        status_code=422 if settings_error else 200,
    )


@app.post("/settings/mcp", response_class=HTMLResponse)
async def settings_mcp_form(request: Request) -> HTMLResponse:
    """Handle local MCP server add/remove actions from settings page."""
    form = await _parse_urlencoded_form(request)
    action = (form.get("action") or "add").strip().lower()
    server_id = (form.get("mcp_id") or "").strip()
    command = (form.get("mcp_command") or "").strip()
    args_raw = (form.get("mcp_args") or "").strip()

    settings_error: str | None = None
    settings_message: str | None = None

    if action == "remove":
        if not server_id:
            settings_error = "MCP server id is required to remove."
        else:
            removed = _services().gateway.remove_local_mcp_server(server_id)
            if removed:
                settings_message = f"Removed MCP server '{server_id}'."
            else:
                settings_error = "MCP server not found."
    else:
        if not server_id or not command:
            settings_error = "MCP server id and command are required."
        else:
            args = (
                [part for part in args_raw.split(" ") if part.strip()]
                if args_raw
                else []
            )
            payload = McpLocalServerRequest(
                id=server_id,
                command=command,
                args=args,
                env={},
                enabled=True,
            )
            upsert_mcp_local_server(payload)
            tool_prefix = f"mcp.{server_id}."
            discovered = [
                name
                for name in _services().gateway.list_tool_names()
                if name.startswith(tool_prefix)
            ]
            if discovered:
                settings_message = (
                    f"Saved MCP server '{server_id}' ({len(discovered)} tool(s) discovered)."
                )
            else:
                settings_message = (
                    f"Saved MCP server '{server_id}', but no tools were discovered. "
                    "Use an absolute command path and verify the server can start quickly."
                )

    return templates.TemplateResponse(
        request,
        "settings.html",
        _settings_context(
            settings_error=settings_error,
            settings_message=settings_message,
        ),
        status_code=422 if settings_error else 200,
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    """Return JSON errors for HTTPException.

    Args:
        _: Incoming request (unused).
        exc: Raised HTTP exception.

    Returns:
        JSON response with a consistent `{ok: false, error: ...}` shape.
    """
    return JSONResponse(
        status_code=exc.status_code, content={"ok": False, "error": exc.detail}
    )
