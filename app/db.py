"""SQLite database utilities and schema initialization.

Provides helpers for resolving the DB path, creating connections with a
dict row factory that decodes JSON-ish columns, and initializing tables.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


DEFAULT_DB_PATH = "data/overmind.db"


def utc_now() -> str:
    """Return the current UTC timestamp as an ISO-8601 string.

    Returns:
      ISO-8601 timestamp string with timezone info.
    """
    return datetime.now(timezone.utc).isoformat()


def resolve_db_path(db_path: str | None = None) -> str:
    """Resolve the SQLite DB path and ensure its parent directory exists.

    Resolution order is:
    1) Explicit `db_path` argument
    2) `OVERMIND_DB` environment variable
    3) `DEFAULT_DB_PATH`

    Args:
      db_path: Optional path to the SQLite database.

    Returns:
      A filesystem path (as a string) to the SQLite database.
    """
    raw = db_path or os.getenv("OVERMIND_DB", DEFAULT_DB_PATH)
    path = Path(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def _dict_factory(cursor: sqlite3.Cursor, row: tuple[Any, ...]) -> dict[str, Any]:
    """Convert a SQLite row tuple into a dict keyed by column names.

    For a small set of known JSON-ish columns, attempt to decode JSON strings
    into Python values.

    Args:
      cursor: SQLite cursor with a populated `description`.
      row: Raw row values as returned by SQLite.

    Returns:
      Mapping of column name to decoded value.
    """
    result: dict[str, Any] = {}
    for idx, col in enumerate(cursor.description):
        key = col[0]
        value = row[idx]
        if isinstance(value, str) and (
            key.endswith("_json") or key in {"tools_allowed", "embedding"}
        ):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        result[key] = value
    return result


@contextmanager
def get_conn(db_path: str) -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection configured for this app.

    The connection uses WAL mode, enforces foreign keys, and uses a row factory
    that returns dictionaries.

    Args:
      db_path: Filesystem path to the SQLite database.

    Yields:
      An open `sqlite3.Connection`. The connection is committed and closed on
      exit.
    """
    conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
    conn.row_factory = _dict_factory
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str) -> None:
    """Create all tables and indexes if they do not exist.

    Args:
      db_path: Filesystem path to the SQLite database.

    Returns:
      None.
    """
    schema = """
    CREATE TABLE IF NOT EXISTS agents (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      role TEXT NOT NULL,
      model TEXT NOT NULL,
      tools_allowed TEXT NOT NULL,
      status TEXT NOT NULL,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      version INTEGER NOT NULL DEFAULT 1
    );

    CREATE TABLE IF NOT EXISTS runs (
      id TEXT PRIMARY KEY,
      agent_id TEXT NOT NULL,
      task TEXT NOT NULL,
      status TEXT NOT NULL,
      step_limit INTEGER NOT NULL,
      created_at TEXT NOT NULL,
      started_at TEXT,
      finished_at TEXT,
      FOREIGN KEY(agent_id) REFERENCES agents(id)
    );

    CREATE TABLE IF NOT EXISTS steps (
      id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL,
      idx INTEGER NOT NULL,
      type TEXT NOT NULL,
      input_json TEXT,
      output_json TEXT,
      started_at TEXT NOT NULL,
      finished_at TEXT,
      error TEXT,
      FOREIGN KEY(run_id) REFERENCES runs(id)
    );

    CREATE TABLE IF NOT EXISTS tool_calls (
      id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL,
      step_id TEXT,
      tool_name TEXT NOT NULL,
      args_json TEXT,
      result_json TEXT,
      allowed INTEGER NOT NULL,
      latency_ms INTEGER NOT NULL,
      created_at TEXT NOT NULL,
      FOREIGN KEY(run_id) REFERENCES runs(id),
      FOREIGN KEY(step_id) REFERENCES steps(id)
    );

    CREATE TABLE IF NOT EXISTS model_calls (
      id TEXT PRIMARY KEY,
      run_id TEXT,
      agent_id TEXT,
      model TEXT NOT NULL,
      request_json TEXT,
      response_json TEXT,
      usage_json TEXT,
      error TEXT,
      latency_ms INTEGER NOT NULL,
      created_at TEXT NOT NULL,
      FOREIGN KEY(run_id) REFERENCES runs(id),
      FOREIGN KEY(agent_id) REFERENCES agents(id)
    );

    CREATE TABLE IF NOT EXISTS memory_items (
      id TEXT PRIMARY KEY,
      collection TEXT NOT NULL,
      text TEXT NOT NULL,
      embedding TEXT NOT NULL,
      embedding_model TEXT NOT NULL DEFAULT 'fts5-bm25',
      dims INTEGER NOT NULL DEFAULT 0,
      metadata_json TEXT,
      created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS events (
      id TEXT PRIMARY KEY,
      run_id TEXT,
      type TEXT NOT NULL,
      payload_json TEXT,
      ts TEXT NOT NULL,
      FOREIGN KEY(run_id) REFERENCES runs(id)
    );

    CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
    CREATE INDEX IF NOT EXISTS idx_steps_run_idx ON steps(run_id, idx);
    CREATE INDEX IF NOT EXISTS idx_tool_calls_run ON tool_calls(run_id);
    CREATE INDEX IF NOT EXISTS idx_model_calls_run ON model_calls(run_id);
    CREATE INDEX IF NOT EXISTS idx_events_run_ts ON events(run_id, ts);
    CREATE INDEX IF NOT EXISTS idx_memory_collection ON memory_items(collection);
    """
    with get_conn(db_path) as conn:
        conn.executescript(schema)
        _ensure_memory_item_columns(conn)
        _ensure_memory_fts(conn)


def _ensure_memory_item_columns(conn: sqlite3.Connection) -> None:
    """Ensure newer memory_items columns exist for older DB files."""
    cols = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(memory_items)").fetchall()
    }
    if "embedding_model" not in cols:
        conn.execute(
            "ALTER TABLE memory_items ADD COLUMN embedding_model TEXT NOT NULL DEFAULT 'fts5-bm25'"
        )
    if "dims" not in cols:
        conn.execute(
            "ALTER TABLE memory_items ADD COLUMN dims INTEGER NOT NULL DEFAULT 0"
        )


def _ensure_memory_fts(conn: sqlite3.Connection) -> None:
    """Create and backfill FTS5 index for memory text, if supported."""
    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_items_fts
            USING fts5(id UNINDEXED, collection UNINDEXED, text)
            """
        )
        conn.executescript(
            """
            CREATE TRIGGER IF NOT EXISTS memory_items_ai AFTER INSERT ON memory_items BEGIN
              INSERT INTO memory_items_fts(id, collection, text)
              VALUES (new.id, new.collection, new.text);
            END;

            CREATE TRIGGER IF NOT EXISTS memory_items_ad AFTER DELETE ON memory_items BEGIN
              INSERT INTO memory_items_fts(memory_items_fts, rowid, id, collection, text)
              VALUES('delete', old.rowid, old.id, old.collection, old.text);
            END;

            CREATE TRIGGER IF NOT EXISTS memory_items_au AFTER UPDATE ON memory_items BEGIN
              INSERT INTO memory_items_fts(memory_items_fts, rowid, id, collection, text)
              VALUES('delete', old.rowid, old.id, old.collection, old.text);
              INSERT INTO memory_items_fts(id, collection, text)
              VALUES (new.id, new.collection, new.text);
            END;
            """
        )
        conn.execute(
            """
            INSERT INTO memory_items_fts(id, collection, text)
            SELECT m.id, m.collection, m.text
            FROM memory_items m
            WHERE NOT EXISTS (
              SELECT 1 FROM memory_items_fts f WHERE f.id = m.id
            )
            """
        )
    except sqlite3.OperationalError:
        pass
