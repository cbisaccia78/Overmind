from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import AppState, app


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
