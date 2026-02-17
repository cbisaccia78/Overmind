from __future__ import annotations


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
