# Overmind (Local-First Agent Control Plane)

Repo contains a runnable single-process app with:
- Agent Registry (CRUD + version increments)
- Deterministic Orchestrator (step loop, retries, step limit)
- Tool Gateway (allowlist + audit log)
- Docker Sandbox Runner (real container execution)
- Vector Memory (deterministic local embeddings + search)
- Structured Logs/Telemetry (events + replay timeline)
- Minimal Web Dashboard (agents, runs, run detail)

## Architecture

Single-process Python FastAPI app with:
- HTTP API + dashboard pages
- In-process worker threads for orchestrator run execution
- SQLite for local persistence
- Docker CLI sandbox runner

Key files:
- [app/main.py](app/main.py)
- [app/db.py](app/db.py)
- [app/repository.py](app/repository.py)
- [app/orchestrator.py](app/orchestrator.py)
- [app/tool_gateway.py](app/tool_gateway.py)
- [app/docker_runner.py](app/docker_runner.py)
- [app/memory.py](app/memory.py)

## Data Model

SQLite tables implemented in [app/db.py](app/db.py):
- `agents(id, name, role, model, tools_allowed, status, created_at, updated_at, version)`
- `runs(id, agent_id, task, status, step_limit, created_at, started_at, finished_at)`
- `steps(id, run_id, idx, type, input_json, output_json, started_at, finished_at, error)`
- `tool_calls(id, run_id, step_id, tool_name, args_json, result_json, allowed, latency_ms, created_at)`
- `memory_items(id, collection, text, embedding, metadata_json, created_at)`
- `events(id, run_id, type, payload_json, ts)`

## Setup and Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
make dev
```
 
Environment variables:

- `OVERMIND_DB`: SQLite DB path (default: `data/overmind.db`).
- `OVERMIND_WORKSPACE`: Workspace root used to constrain file tools (defaults to the repo root).

Tests:

```bash
make test
```

Docker integration test (opt-in):

```bash
make test-docker
```

Full test run (normal + Docker):

```bash
make test-full
```

## API Coverage

Agents:
- `POST /api/agents`
- `GET /api/agents`
- `GET /api/agents/{id}`
- `PATCH /api/agents/{id}`
- `POST /api/agents/{id}/disable`

Runs:
- `POST /api/runs`
- `GET /api/runs`
- `GET /api/runs/{id}`
- `POST /api/runs/{id}/cancel`
- `GET /api/runs/{id}/steps`
- `GET /api/runs/{id}/tool-calls`
- `GET /api/runs/{id}/events`
- `GET /api/runs/{id}/replay`

Memory:
- `POST /api/memory/store`
- `POST /api/memory/search`

Tool Gateway:
- `POST /api/runs/{id}/tools/call`

## Docker Sandbox Notes

`run_shell` executes in Docker (`alpine:3.20`) with:
- `--network none`
- `--cpus 0.5`
- `--memory 256m`
- read-only workspace mount by default; explicit writable mount optional
- subprocess-level timeout enforcement