"""Host shell runner.

Executes shell commands directly on the host OS with workspace-scoped working
directory selection and timeout enforcement.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


class ShellRunner:
    """Run shell commands on the host operating system.

    Attributes:
        workspace_root: Absolute workspace path used as default working directory.
        default_timeout_s: Default command timeout in seconds.
    """

    def __init__(
        self,
        workspace_root: str,
        default_timeout_s: int = 20,
    ) -> None:
        """Create a runner for executing shell commands on the host.

        Args:
            workspace_root: Workspace root directory.
            default_timeout_s: Default timeout (seconds) for command execution.
        """
        self.workspace_root = str(Path(workspace_root).resolve())
        self.default_timeout_s = default_timeout_s

    def is_available(self) -> bool:
        """Check whether an interactive shell is available on PATH."""
        return shutil.which("sh") is not None

    def _safe_subdir(self, write_subdir: str | None, run_id: str | None) -> Path:
        default_subdir = (
            f".overmind_runs/{run_id}" if run_id else ".overmind_runs/session"
        )
        subdir = (write_subdir or default_subdir).lstrip("/")
        target = Path(self.workspace_root, subdir).resolve()
        if not str(target).startswith(self.workspace_root):
            raise ValueError("write_subdir escapes workspace")
        return target

    def run_shell(
        self,
        command: str,
        timeout_s: int | None = None,
        allow_write: bool = False,
        write_subdir: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute a shell command on the host.

        Args:
            command: Shell command to run (`sh -lc`).
            timeout_s: Optional timeout override.
            allow_write: If true, run from a writable workspace subdirectory.
            write_subdir: Optional relative writable subdirectory.
            run_id: Optional run identifier used for default writable subdir.

        Returns:
            Structured execution result with success/error details.
        """
        timeout = int(timeout_s or self.default_timeout_s)
        if timeout < 1:
            return {
                "ok": False,
                "error": {
                    "code": "bad_config",
                    "message": "timeout_s must be >= 1",
                    "details": {"timeout_s": timeout},
                },
            }

        cwd = Path(self.workspace_root)
        if allow_write:
            try:
                cwd = self._safe_subdir(write_subdir, run_id)
            except ValueError as exc:
                return {
                    "ok": False,
                    "error": {
                        "code": "bad_config",
                        "message": str(exc),
                    },
                    "command": command,
                }
            cwd.mkdir(parents=True, exist_ok=True)

        start = time.monotonic()
        try:
            result = subprocess.run(
                ["sh", "-lc", command],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                cwd=str(cwd),
            )
            latency_ms = int((time.monotonic() - start) * 1000)
            return {
                "ok": result.returncode == 0,
                "exit_code": result.returncode,
                "stdout": result.stdout[:100_000],
                "stderr": result.stderr[:100_000],
                "latency_ms": latency_ms,
                "command": command,
                "working_dir": str(cwd.relative_to(self.workspace_root)),
            }
        except subprocess.TimeoutExpired:
            latency_ms = int((time.monotonic() - start) * 1000)
            return {
                "ok": False,
                "error": {
                    "code": "timeout",
                    "message": f"Command timed out after {timeout}s",
                    "details": {"timeout_s": timeout, "backend": "host"},
                },
                "latency_ms": latency_ms,
                "command": command,
                "working_dir": str(cwd.relative_to(self.workspace_root)),
            }
