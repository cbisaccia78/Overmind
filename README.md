# Overmind (Local-First Agent Control Plane)

[![CI](https://github.com/cbisaccia78/Overmind/actions/workflows/ci.yml/badge.svg)](https://github.com/cbisaccia78/Overmind/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/cbisaccia78/overmind/branch/main/graph/badge.svg)](https://app.codecov.io/github/cbisaccia78/overmind)

Overmind is a local desktop app for running and monitoring AI agents on your own machine.
It’s designed to feel like a control panel: create agents, start runs, see tool/model calls, and replay what happened.

## Install

The easiest way to use overmind is to download the latest release from the GitHub Releases page.

- AppImage: `Overmind-<version>.AppImage`
  

  ```bash
  chmod +x Overmind-<version>.AppImage
  ./Overmind-<version>.AppImage
  ```

- Debian/Ubuntu package: `overmind-desktop_<version>_amd64.deb`


  ```bash
  sudo apt install ./overmind-desktop_<version>_amd64.deb
  ```

We currently only support Linux, so if you're on mac/windows you need to set up a linux VM.

## Configure OpenAI Key (Desktop App)

You can set your OpenAI key directly in the app:

1. Open Overmind.
2. Go to **Settings**.
3. Paste your key into **OpenAI API Key**.
4. Click **Save Key**.
5. (Optional) Click **Test Key** to verify it works.

This key is saved in Overmind app settings and is loaded automatically when you launch the app from the desktop icon.

## Run from Source (Development)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
npm install
make dev
```

`make dev` launches the Electron desktop app and starts the Python backend automatically.

If you only want the API server during development:

```bash
make api-dev
```

Tests:

```bash
make test
```

Full test run:

```bash
make test-full
```

## Architecture

Electron desktop shell + single-process Python FastAPI backend:
- Native desktop window for dashboard pages
- Local HTTP API consumed by the desktop app
- In-process worker threads for orchestrator run execution
- SQLite for local persistence
- Host shell runner
- Agent Registry (CRUD + version increments)
- Deterministic Orchestrator (step loop, retries, step limit)
- Tool Gateway (allowlist + strict arg validation + audit log)
- Host Shell Runner (direct host OS execution)
- Memory (FTS-backed retrieval + optional embeddings)
- Structured Logs/Telemetry (events + replay timeline + tool/model call audit)
- Desktop Dashboard via Electron (agents, runs, run detail)

Key files:
- [app/main.py](app/main.py)
- [app/db.py](app/db.py)
- [app/repository.py](app/repository.py)
- [app/orchestrator.py](app/orchestrator.py)
- [app/tool_gateway.py](app/tool_gateway.py)
- [app/shell_runner.py](app/shell_runner.py)
- [app/memory.py](app/memory.py)

## Data Model

SQLite tables implemented in [app/db.py](app/db.py):
- `agents(id, name, role, model, tools_allowed, status, created_at, updated_at, version)`
- `runs(id, agent_id, task, status, step_limit, created_at, started_at, finished_at)`
- `steps(id, run_id, idx, type, input_json, output_json, started_at, finished_at, error)`
- `tool_calls(id, run_id, step_id, tool_name, args_json, result_json, allowed, latency_ms, created_at)`
- `model_calls(id, run_id, agent_id, model, request_json, response_json, usage_json, error, latency_ms, created_at)`
- `memory_items(id, collection, text, embedding, embedding_model, dims, metadata_json, created_at)`
- `events(id, run_id, type, payload_json, ts)`

## Packaging (Linux)

Build desktop distributables:

```bash
make package
```

This produces Linux artifacts in `dist/`:
- AppImage (`*.AppImage`)
- Debian package (`*.deb`)

To produce an unpacked build directory only:

```bash
make package-unpacked
```

Packaging note: release artifacts bundle the backend as a standalone executable via PyInstaller. End users do not need Python installed.

## Release Build (GitHub)

To produce a release build and publish installers through GitHub Actions:

1. Bump desktop version in `package.json` to match your next tag (for example: `0.1.3` with tag `v0.1.3`).
2. Commit and push to `main`.
3. Build locally to verify packaging succeeds:

```bash
make package
```

4. Create and push a version tag:

```bash
git tag v0.1.3
git push origin v0.1.3
```

The `Release` workflow builds and uploads these assets to the GitHub Release:
- `Overmind-<version>.AppImage`
- `overmind-desktop_<version>_amd64.deb`

To install those artifacts, see **Install (GitHub Releases)** above.
 
Environment variables:

- `OVERMIND_DB`: SQLite DB path (default: `data/overmind.db`).
- `OVERMIND_WORKSPACE`: Workspace root used to constrain file tools (defaults to the repo root).
- `OVERMIND_EMBEDDING_PROVIDER`: `auto` (default), `openai` (requires `OPENAI_API_KEY`), or any other value to force local FTS/BM25 mode.
- `OPENAI_API_KEY`: Optional; if set (and provider is `auto`/`openai`), memory embeddings are generated via OpenAI.
- `OVERMIND_EMBEDDING_MODEL`: OpenAI embedding model name (default: `text-embedding-3-small`).
- `OVERMIND_OPENAI_EMBEDDINGS_URL`: Optional override for the embeddings endpoint.

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
- `POST /api/runs/{id}/input`
- `GET /api/runs/{id}/steps`
- `GET /api/runs/{id}/tool-calls`
- `GET /api/runs/{id}/events`
- `GET /api/runs/{id}/replay`

Memory:
- `POST /api/memory/store`
- `POST /api/memory/search`

Tool Gateway:
- `POST /api/runs/{id}/tools/call`

## Host Shell Notes

`run_shell` executes directly on the host OS with:
- working directory set to the workspace root by default
- optional writable subdirectory selection when `allow_write=true`
- workspace path validation for writable subdirectories
- subprocess-level timeout enforcement