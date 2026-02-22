from __future__ import annotations

from urllib import error as urlerror

import pytest

import app.memory as memory_module
from app.memory import (
    LocalBm25Backend,
    LocalVectorMemory,
    OpenAIEmbeddingBackend,
    _select_backend,
)


class _RepoStub:
    def __init__(self, candidates, listed):
        self._candidates = candidates
        self._listed = listed

    def search_memory_candidates(self, collection, query, limit):
        del collection, query, limit
        return list(self._candidates)

    def list_memory_items(self, collection):
        del collection
        return list(self._listed)


def test_memory_search_ranking(client):
    client.post(
        "/api/memory/store",
        json={"collection": "docs", "text": "python sqlite fastapi", "metadata": {}},
    )
    client.post(
        "/api/memory/store",
        json={"collection": "docs", "text": "gardening flowers roses", "metadata": {}},
    )

    resp = client.post(
        "/api/memory/search",
        json={"collection": "docs", "query": "fastapi sqlite", "top_k": 2},
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 2
    assert "fastapi" in results[0]["text"]
    assert results[0]["score"] >= results[1]["score"]


def test_memory_stores_embedding_metadata_and_candidates(client):
    client.post(
        "/api/memory/store",
        json={
            "collection": "docs",
            "text": "vector search sqlite",
            "metadata": {"k": "v"},
        },
    )

    repo = client.app.state.services.repo
    rows = repo.list_memory_items("docs")
    assert len(rows) >= 1
    assert "embedding_model" in rows[0]
    assert "dims" in rows[0]

    candidates = repo.search_memory_candidates("docs", "sqlite", 5)
    assert len(candidates) >= 1
    assert any("sqlite" in row["text"] for row in candidates)


def test_local_vector_memory_search_backfills_and_deduplicates():
    candidates = [
        {
            "id": 1,
            "text": "alpha",
            "embedding": [1.0, 0.0],
            "dims": 2,
            "embedding_model": "stub",
        }
    ]
    listed = [
        {
            "id": 1,
            "text": "alpha",
            "embedding": [1.0, 0.0],
            "dims": 2,
            "embedding_model": "stub",
        },
        {
            "id": 2,
            "text": "beta",
            "embedding": [0.0, 1.0],
            "dims": 2,
            "embedding_model": "stub",
        },
    ]
    memory = LocalVectorMemory.__new__(LocalVectorMemory)
    memory.repo = _RepoStub(candidates=candidates, listed=listed)
    memory.backend = LocalBm25Backend()
    memory.embed = lambda text: [1.0, 0.0]

    results = memory.search("query", "docs", top_k=2)

    assert len(results) == 2
    assert [r["id"] for r in results] == [1, 2]


def test_local_vector_memory_search_returns_empty_when_no_items():
    memory = LocalVectorMemory.__new__(LocalVectorMemory)
    memory.repo = _RepoStub(candidates=[], listed=[])
    memory.backend = LocalBm25Backend()
    memory.embed = lambda text: []

    assert memory.search("query", "docs", top_k=3) == []


def test_select_backend_openai_requires_api_key(monkeypatch):
    monkeypatch.setenv("OVERMIND_EMBEDDING_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is required"):
        _select_backend()


def test_openai_embedding_backend_wraps_transport_errors(monkeypatch):
    backend = OpenAIEmbeddingBackend(api_key="k", model_name="m")

    def _fail(req, timeout=10):
        del req, timeout
        raise urlerror.URLError("boom")

    monkeypatch.setattr(memory_module.urlrequest, "urlopen", _fail)

    with pytest.raises(RuntimeError, match="embedding request failed"):
        backend.embed("hello")
