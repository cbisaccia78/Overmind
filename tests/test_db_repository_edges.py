from __future__ import annotations

from app.db import get_conn, init_db
from app.memory import LocalVectorMemory
from app.repository import Repository


def test_db_dict_factory_handles_invalid_json(tmp_path):
    db_path = str(tmp_path / "edge.db")
    init_db(db_path)

    with get_conn(db_path) as conn:
        conn.execute(
            """
            INSERT INTO agents(id, name, role, model, tools_allowed, status, created_at, updated_at, version)
            VALUES('a1','n','r','m','{not-json','active','t','t',1)
            """
        )

    repo = Repository(db_path)
    row = repo.get_agent("a1")
    assert row is not None
    assert row["tools_allowed"] == "{not-json"


def test_repository_noop_paths_and_memory_collection_filter(tmp_path):
    db_path = str(tmp_path / "repo.db")
    init_db(db_path)
    repo = Repository(db_path)

    assert repo.update_agent("missing", {"role": "x"}) is None
    repo.update_run_status("missing", "running")

    a = repo.create_agent("n", "r", "m", ["store_memory"])
    run = repo.create_run(a["id"], "remember:notes:test", step_limit=2)
    repo.update_run_status(run["id"], "running")
    repo.update_run_status(run["id"], "failed")

    mem = LocalVectorMemory(repo=repo)
    vec = mem.embed("")
    assert vec == []

    repo.add_memory_item("docs", "abc", [0.1] * 8, "test-embed", 8, {"k": "v"})
    repo.add_memory_item("other", "xyz", [0.2] * 8, "test-embed", 8, None)
    docs = repo.list_memory_items("docs")
    assert len(docs) == 1
    assert docs[0]["collection"] == "docs"


def test_repository_model_call_roundtrip(tmp_path):
    db_path = str(tmp_path / "repo-model.db")
    init_db(db_path)
    repo = Repository(db_path)

    agent = repo.create_agent("n", "r", "stub-v1", ["store_memory"])
    run = repo.create_run(agent["id"], "remember:notes:test", step_limit=2)

    created = repo.create_model_call(
        run_id=run["id"],
        agent_id=agent["id"],
        model="stub-v1",
        request_json={"task": "remember:notes:test"},
        response_json={"tool_name": "store_memory", "args": {"collection": "notes"}},
        usage_json={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        error=None,
        latency_ms=12,
    )

    assert created["model"] == "stub-v1"
    rows = repo.list_model_calls(run["id"])
    assert len(rows) == 1
    assert rows[0]["request_json"]["task"] == "remember:notes:test"
    assert rows[0]["response_json"]["tool_name"] == "store_memory"
    assert rows[0]["usage_json"]["total_tokens"] == 5
