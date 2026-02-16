"""Tool execution gateway.

Enforces an agent allowlist, dispatches supported tools (files, memory,
Docker shell), and records tool call audit events.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .docker_runner import DockerRunner
from .memory import LocalVectorMemory
from .repository import Repository


class ToolGateway:
    """Authorize and dispatch tool calls, then persist audit records."""

    def __init__(
        self,
        repo: Repository,
        memory: LocalVectorMemory,
        docker_runner: DockerRunner,
        workspace_root: str,
        max_file_bytes: int = 100_000,
    ):
        """Create a tool gateway for dispatching and auditing tool calls.

        Args:
            repo: Repository used to persist tool call audit records and events.
            memory: Memory subsystem used by memory tools.
            docker_runner: Docker sandbox used by `run_shell`.
            workspace_root: Root directory used to constrain file access.
            max_file_bytes: Maximum bytes for read/write operations.
        """
        self.repo = repo
        self.memory = memory
        self.docker_runner = docker_runner
        self.workspace_root = Path(workspace_root).resolve()
        self.max_file_bytes = max_file_bytes

    def _safe_path(self, rel_path: str) -> Path:
        """Resolve a relative path within the workspace root.

        Args:
            rel_path: Path relative to `workspace_root`.

        Returns:
            Resolved `Path` inside the workspace.

        Raises:
            ValueError: If the resolved path escapes the workspace.
        """
        target = (self.workspace_root / rel_path).resolve()
        if not str(target).startswith(str(self.workspace_root)):
            raise ValueError("Path escapes workspace")
        return target

    def call(
        self,
        *,
        run_id: str,
        step_id: str | None,
        agent: dict[str, Any],
        tool_name: str,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Authorize, dispatch, and audit a tool call.

        Args:
            run_id: Run ID.
            step_id: Optional step ID associated with the call.
            agent: Agent row dict (must include `tools_allowed` and `id`).
            tool_name: Tool name.
            args: Tool arguments.

        Returns:
            Tool result dict. Always includes `ok`.
        """
        allowed_tools = set(agent.get("tools_allowed") or [])
        start = time.monotonic()
        if tool_name not in allowed_tools:
            result = {
                "ok": False,
                "error": {
                    "code": "tool_not_allowed",
                    "message": f"Tool '{tool_name}' is not allowed for agent {agent['id']}",
                },
            }
            latency_ms = int((time.monotonic() - start) * 1000)
            self.repo.create_tool_call(
                run_id, step_id, tool_name, args, result, False, latency_ms
            )
            self.repo.create_event(
                run_id, "tool.denied", {"tool_name": tool_name, "args": args}
            )
            return result

        try:
            result = self._dispatch(tool_name, args)
        except Exception as exc:  # noqa: BLE001
            result = {
                "ok": False,
                "error": {
                    "code": "tool_error",
                    "message": str(exc),
                },
            }

        latency_ms = int((time.monotonic() - start) * 1000)
        self.repo.create_tool_call(
            run_id, step_id, tool_name, args, result, True, latency_ms
        )
        self.repo.create_event(
            run_id,
            "tool.called",
            {
                "tool_name": tool_name,
                "args": args,
                "ok": result.get("ok", False),
                "latency_ms": latency_ms,
            },
        )
        return result

    def _dispatch(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Dispatch the tool implementation for a supported tool name.

        Args:
            tool_name: Tool name.
            args: Tool arguments.

        Returns:
            Tool result dict.
        """
        if tool_name == "run_shell":
            command = str(args.get("command", "")).strip()
            if not command:
                return {
                    "ok": False,
                    "error": {"code": "bad_args", "message": "command is required"},
                }
            timeout_s = int(args.get("timeout_s", 20))
            allow_write = bool(args.get("allow_write", False))
            write_subdir = args.get("write_subdir")
            return self.docker_runner.run_shell(
                command=command,
                timeout_s=timeout_s,
                allow_write=allow_write,
                write_subdir=write_subdir,
            )

        if tool_name == "read_file":
            path = self._safe_path(str(args.get("path", "")))
            if not path.exists() or not path.is_file():
                return {
                    "ok": False,
                    "error": {"code": "not_found", "message": "file not found"},
                }
            content = path.read_text(encoding="utf-8")
            if len(content.encode("utf-8")) > self.max_file_bytes:
                return {
                    "ok": False,
                    "error": {"code": "size_limit", "message": "file exceeds max size"},
                }
            return {
                "ok": True,
                "path": str(path.relative_to(self.workspace_root)),
                "content": content,
            }

        if tool_name == "write_file":
            path = self._safe_path(str(args.get("path", "")))
            content = str(args.get("content", ""))
            size = len(content.encode("utf-8"))
            if size > self.max_file_bytes:
                return {
                    "ok": False,
                    "error": {
                        "code": "size_limit",
                        "message": "content exceeds max size",
                    },
                }
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return {
                "ok": True,
                "path": str(path.relative_to(self.workspace_root)),
                "bytes": size,
            }

        if tool_name == "store_memory":
            collection = str(args.get("collection", "default"))
            text = str(args.get("text", ""))
            metadata = args.get("metadata") or {}
            item = self.memory.store(
                text=text, collection=collection, metadata=metadata
            )
            return {"ok": True, "item": item}

        if tool_name == "search_memory":
            collection = str(args.get("collection", "default"))
            query = str(args.get("query", ""))
            top_k = int(args.get("top_k", 5))
            results = self.memory.search(
                query=query, collection=collection, top_k=top_k
            )
            return {"ok": True, "results": results}

        return {"ok": False, "error": {"code": "unknown_tool", "message": tool_name}}
