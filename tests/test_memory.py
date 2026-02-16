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
