"""SQLite-backed persistence layer.

Implements CRUD operations and audit/event logging for agents, runs,
steps, tool calls, and memory items.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from .db import get_conn, utc_now


class Repository:
    """SQLite-backed persistence and query API for Overmind.

    Stores agents, runs, steps, tool calls, memory items, and events.

    Attributes:
        db_path: Filesystem path to the SQLite database.
    """

    def __init__(self, db_path: str):
        """Create a repository backed by a SQLite database.

        Args:
            db_path: Filesystem path to the SQLite database file.
        """
        self.db_path = db_path

    @staticmethod
    def _id() -> str:
        """Generate a new UUID string.

        Returns:
            UUID4 as a string.
        """
        return str(uuid.uuid4())

    def create_agent(
        self,
        name: str,
        role: str,
        model: str,
        tools_allowed: list[str],
        status: str = "active",
    ) -> dict[str, Any]:
        """Create and persist a new agent.

        Args:
            name: Agent display name.
            role: Free-form role/description.
            model: Model identifier string.
            tools_allowed: List of tool names the agent can invoke.
            status: Initial status (e.g. "active").

        Returns:
            The created agent row as a dict.
        """
        now = utc_now()
        row = {
            "id": self._id(),
            "name": name,
            "role": role,
            "model": model,
            "tools_allowed": tools_allowed,
            "status": status,
            "created_at": now,
            "updated_at": now,
            "version": 1,
        }
        with get_conn(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO agents(id, name, role, model, tools_allowed, status, created_at, updated_at, version)
                VALUES(:id, :name, :role, :model, :tools_allowed, :status, :created_at, :updated_at, :version)
                """,
                {**row, "tools_allowed": json.dumps(row["tools_allowed"])},
            )
        return row

    def list_agents(self) -> list[dict[str, Any]]:
        """Return all agents ordered by most recent creation.

        Returns:
            List of agent rows.
        """
        with get_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM agents ORDER BY created_at DESC"
            ).fetchall()
        return list(rows)

    def get_agent(self, agent_id: str) -> dict[str, Any] | None:
        """Fetch a single agent by ID.

        Args:
            agent_id: Agent ID.

        Returns:
            Agent row dict if found, else None.
        """
        with get_conn(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM agents WHERE id = ?", (agent_id,)
            ).fetchone()
        return row

    def update_agent(
        self, agent_id: str, updates: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Update an agent, bumping its version and updated timestamp.

        Args:
            agent_id: Agent ID to update.
            updates: Partial update payload. Missing keys keep existing values.

        Returns:
            Updated agent row if the agent exists, else None.
        """
        existing = self.get_agent(agent_id)
        if not existing:
            return None

        merged = {
            "name": updates.get("name", existing["name"]),
            "role": updates.get("role", existing["role"]),
            "model": updates.get("model", existing["model"]),
            "tools_allowed": updates.get("tools_allowed", existing["tools_allowed"]),
            "status": updates.get("status", existing["status"]),
            "updated_at": utc_now(),
            "version": int(existing["version"]) + 1,
            "id": agent_id,
        }
        with get_conn(self.db_path) as conn:
            conn.execute(
                """
                UPDATE agents
                SET name=:name, role=:role, model=:model, tools_allowed=:tools_allowed,
                    status=:status, updated_at=:updated_at, version=:version
                WHERE id=:id
                """,
                {**merged, "tools_allowed": json.dumps(merged["tools_allowed"])},
            )
        return self.get_agent(agent_id)

    def disable_agent(self, agent_id: str) -> dict[str, Any] | None:
        """Disable an agent.

        Args:
            agent_id: Agent ID.

        Returns:
            Updated agent row if the agent exists, else None.
        """
        return self.update_agent(agent_id, {"status": "disabled"})

    def create_run(
        self, agent_id: str, task: str, step_limit: int = 8
    ) -> dict[str, Any]:
        """Create and persist a new run.

        Args:
            agent_id: ID of the agent that will execute the run.
            task: Task string.
            step_limit: Maximum number of planned steps to execute.

        Returns:
            The created run row as a dict.
        """
        now = utc_now()
        row = {
            "id": self._id(),
            "agent_id": agent_id,
            "task": task,
            "status": "queued",
            "step_limit": step_limit,
            "created_at": now,
            "started_at": None,
            "finished_at": None,
        }
        with get_conn(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO runs(id, agent_id, task, status, step_limit, created_at, started_at, finished_at)
                VALUES(:id, :agent_id, :task, :status, :step_limit, :created_at, :started_at, :finished_at)
                """,
                row,
            )
        return row

    def list_runs(self) -> list[dict[str, Any]]:
        """Return all runs ordered by most recent creation.

        Returns:
            List of run rows.
        """
        with get_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC"
            ).fetchall()
        return list(rows)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Fetch a single run by ID.

        Args:
            run_id: Run ID.

        Returns:
            Run row dict if found, else None.
        """
        with get_conn(self.db_path) as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return row

    def update_run_status(self, run_id: str, status: str) -> None:
        """Update a run's status and derived timestamps.

        When transitioning to `running`, sets `started_at` if absent. When
        transitioning to a terminal status, sets `finished_at`.

        Args:
            run_id: Run ID.
            status: New status string.

        Returns:
            None.
        """
        run = self.get_run(run_id)
        if not run:
            return
        started_at = run.get("started_at")
        finished_at = run.get("finished_at")
        now = utc_now()
        if status == "running" and not started_at:
            started_at = now
        if status in {"succeeded", "failed", "canceled"}:
            finished_at = now

        with get_conn(self.db_path) as conn:
            conn.execute(
                "UPDATE runs SET status=?, started_at=?, finished_at=? WHERE id=?",
                (status, started_at, finished_at, run_id),
            )

    def create_step(
        self,
        run_id: str,
        idx: int,
        step_type: str,
        input_json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create and persist a step row for a run.

        Args:
            run_id: Parent run ID.
            idx: Step index within the run.
            step_type: Logical step type (e.g. "tool").
            input_json: Optional input payload.

        Returns:
            The created step row as a dict.
        """
        row = {
            "id": self._id(),
            "run_id": run_id,
            "idx": idx,
            "type": step_type,
            "input_json": input_json or {},
            "output_json": None,
            "started_at": utc_now(),
            "finished_at": None,
            "error": None,
        }
        with get_conn(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO steps(id, run_id, idx, type, input_json, output_json, started_at, finished_at, error)
                VALUES(:id, :run_id, :idx, :type, :input_json, :output_json, :started_at, :finished_at, :error)
                """,
                {
                    **row,
                    "input_json": json.dumps(row["input_json"]),
                    "output_json": None,
                },
            )
        return row

    def finish_step(
        self,
        step_id: str,
        output_json: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Mark a step as finished, storing output and optional error.

        Args:
            step_id: Step ID.
            output_json: Output payload to store.
            error: Optional error message.

        Returns:
            None.
        """
        with get_conn(self.db_path) as conn:
            conn.execute(
                "UPDATE steps SET output_json=?, finished_at=?, error=? WHERE id=?",
                (
                    json.dumps(output_json or {}),
                    utc_now(),
                    error,
                    step_id,
                ),
            )

    def list_steps(self, run_id: str) -> list[dict[str, Any]]:
        """List steps for a run ordered by index.

        Args:
            run_id: Run ID.

        Returns:
            List of step rows.
        """
        with get_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM steps WHERE run_id=? ORDER BY idx ASC, started_at ASC",
                (run_id,),
            ).fetchall()
        return list(rows)

    def create_tool_call(
        self,
        run_id: str,
        step_id: str | None,
        tool_name: str,
        args_json: dict[str, Any],
        result_json: dict[str, Any],
        allowed: bool,
        latency_ms: int,
    ) -> dict[str, Any]:
        """Persist a tool call audit record.

        Args:
            run_id: Run ID.
            step_id: Optional step ID associated with the call.
            tool_name: Tool name.
            args_json: Tool arguments.
            result_json: Tool result.
            allowed: Whether the tool call was allowed by policy.
            latency_ms: Call latency in milliseconds.

        Returns:
            The stored tool call row as a dict.
        """
        row = {
            "id": self._id(),
            "run_id": run_id,
            "step_id": step_id,
            "tool_name": tool_name,
            "args_json": args_json,
            "result_json": result_json,
            "allowed": 1 if allowed else 0,
            "latency_ms": latency_ms,
            "created_at": utc_now(),
        }
        with get_conn(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO tool_calls(id, run_id, step_id, tool_name, args_json, result_json, allowed, latency_ms, created_at)
                VALUES(:id, :run_id, :step_id, :tool_name, :args_json, :result_json, :allowed, :latency_ms, :created_at)
                """,
                {
                    **row,
                    "args_json": json.dumps(row["args_json"]),
                    "result_json": json.dumps(row["result_json"]),
                },
            )
        row["allowed"] = bool(row["allowed"])
        return row

    def list_tool_calls(self, run_id: str) -> list[dict[str, Any]]:
        """List tool calls for a run in chronological order.

        Args:
            run_id: Run ID.

        Returns:
            List of tool call rows.
        """
        with get_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM tool_calls WHERE run_id=? ORDER BY created_at ASC",
                (run_id,),
            ).fetchall()
        for row in rows:
            row["allowed"] = bool(row["allowed"])
        return list(rows)

    def create_event(
        self,
        run_id: str | None,
        event_type: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Create and persist an event row.

        Args:
            run_id: Optional run ID this event relates to.
            event_type: Event type string.
            payload: Event payload.

        Returns:
            The created event row as a dict.
        """
        row = {
            "id": self._id(),
            "run_id": run_id,
            "type": event_type,
            "payload_json": payload,
            "ts": utc_now(),
        }
        with get_conn(self.db_path) as conn:
            conn.execute(
                "INSERT INTO events(id, run_id, type, payload_json, ts) VALUES(:id, :run_id, :type, :payload_json, :ts)",
                {**row, "payload_json": json.dumps(payload)},
            )
        return row

    def list_events(self, run_id: str) -> list[dict[str, Any]]:
        """List events for a run in chronological order.

        Args:
            run_id: Run ID.

        Returns:
            List of event rows.
        """
        with get_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE run_id=? ORDER BY ts ASC",
                (run_id,),
            ).fetchall()
        return list(rows)

    def add_memory_item(
        self,
        collection: str,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Persist a memory item and its embedding.

        Args:
            collection: Logical collection name.
            text: Text to store.
            embedding: Vector embedding.
            metadata: Optional metadata payload.

        Returns:
            The created memory item row as a dict.
        """
        row = {
            "id": self._id(),
            "collection": collection,
            "text": text,
            "embedding": embedding,
            "metadata_json": metadata or {},
            "created_at": utc_now(),
        }
        with get_conn(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO memory_items(id, collection, text, embedding, metadata_json, created_at)
                VALUES(:id, :collection, :text, :embedding, :metadata_json, :created_at)
                """,
                {
                    **row,
                    "embedding": json.dumps(embedding),
                    "metadata_json": json.dumps(row["metadata_json"]),
                },
            )
        return row

    def list_memory_items(self, collection: str | None = None) -> list[dict[str, Any]]:
        """List memory items, optionally filtered by collection.

        Args:
            collection: Optional collection name filter.

        Returns:
            List of memory item rows.
        """
        query = "SELECT * FROM memory_items"
        args: tuple[str, ...] = ()
        if collection:
            query += " WHERE collection=?"
            args = (collection,)
        query += " ORDER BY created_at DESC"
        with get_conn(self.db_path) as conn:
            rows = conn.execute(query, args).fetchall()
        return list(rows)
