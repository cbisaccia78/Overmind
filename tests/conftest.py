from __future__ import annotations

from pathlib import Path
import os

import pytest
from fastapi.testclient import TestClient

from app.main import AppState, app


@pytest.fixture(autouse=True)
def _isolate_openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests deterministic by default.

    Many modules switch behavior when OPENAI_API_KEY is set (model planning,
    embeddings). For unit/integration tests we default to local/deterministic
    behavior unless explicitly opted in.
    """
    if os.getenv("OVERMIND_TEST_KEEP_OPENAI") == "1":
        return
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OVERMIND_EMBEDDING_PROVIDER", raising=False)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "seed.txt").write_text("hello overmind", encoding="utf-8")
    return ws


@pytest.fixture
def client(tmp_path: Path, workspace: Path) -> TestClient:
    db_path = tmp_path / "test.db"
    app.state.services = AppState(db_path=str(db_path), workspace_root=str(workspace))
    return TestClient(app)
