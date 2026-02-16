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

    mem = LocalVectorMemory(repo=repo, dims=8)
    vec = mem.embed("")
    assert vec == [0.0] * 8

    repo.add_memory_item("docs", "abc", [0.1] * 8, {"k": "v"})
    repo.add_memory_item("other", "xyz", [0.2] * 8, None)
    docs = repo.list_memory_items("docs")
    assert len(docs) == 1
    assert docs[0]["collection"] == "docs"
