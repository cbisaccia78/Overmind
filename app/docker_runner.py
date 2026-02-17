"""Docker sandbox runner.

Executes shell commands inside a constrained Docker container, optionally
mounting a writable subdirectory of the workspace.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SandboxConfig:
    """Execution constraints for docker sandbox runs.

    Attributes:
        backend: Sandbox backend identifier (currently only "docker").
        enforce_pinned_image: If True, require images pinned by digest.
        pids_limit: Container PID limit.
        writable_tmp_mb: Size of a writable `/tmp` tmpfs in MB.
    """

    backend: str = "docker"
    enforce_pinned_image: bool = False
    pids_limit: int = 128
    writable_tmp_mb: int = 64


class DockerRunner:
    """Run shell commands inside a constrained Docker container.

    Attributes:
        workspace_root: Absolute host workspace path mounted into the container.
        image: Docker image name.
        default_timeout_s: Default command timeout in seconds.
        memory_limit: Docker memory limit.
        cpu_limit: Docker CPU limit.
        sandbox: Additional sandbox hardening configuration.
    """

    def __init__(
        self,
        workspace_root: str,
        image: str = "alpine:3.20",
        default_timeout_s: int = 20,
        memory_limit: str = "256m",
        cpu_limit: str = "0.5",
        sandbox: SandboxConfig | None = None,
    ):
        """Create a runner for executing commands in a Docker sandbox.

        Args:
            workspace_root: Host path to mount at `/workspace` inside the container.
            image: Docker image to run.
            default_timeout_s: Default timeout (seconds) for command execution.
            memory_limit: Docker memory limit string (e.g. "256m").
            cpu_limit: Docker CPU limit string (e.g. "0.5").
            sandbox: Additional sandbox hardening configuration.
        """
        self.workspace_root = str(Path(workspace_root).resolve())
        self.image = image
        self.default_timeout_s = default_timeout_s
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.sandbox = sandbox or SandboxConfig()

    def _probe_docker(self) -> tuple[bool, dict[str, Any] | None]:
        """Probe docker cli/daemon availability with diagnostics."""
        docker_path = shutil.which("docker")
        if docker_path is None:
            return False, {
                "reason": "docker_binary_missing",
                "message": "docker binary not found on PATH",
            }
        try:
            probe = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if probe.returncode != 0:
                return False, {
                    "reason": "docker_info_failed",
                    "message": "docker info returned non-zero exit code",
                    "exit_code": probe.returncode,
                    "stderr": (probe.stderr or "")[:1000],
                }
            return True, None
        except subprocess.TimeoutExpired:
            return False, {
                "reason": "docker_info_timeout",
                "message": "docker info timed out after 3s",
            }

    def is_available(self) -> bool:
        """Check whether Docker CLI and daemon are available.

        Returns:
            True if `docker` is on PATH and `docker info` succeeds.
        """
        ok, _ = self._probe_docker()
        return ok

    def run_shell(
        self,
        command: str,
        timeout_s: int | None = None,
        allow_write: bool = False,
        write_subdir: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute a shell command inside a restricted Docker container.

        The workspace root is mounted read-only by default. If `allow_write` is
        True, a single subdirectory can be mounted read-write at
        `/workspace_writable` and used as the working directory.

        Args:
            command: Shell command to run (executed with `sh -lc`).
            timeout_s: Timeout in seconds. Defaults to `default_timeout_s`.
            allow_write: Whether to allow writing to a mounted subdirectory.
            write_subdir: Subdirectory (relative to workspace_root) to mount as
                writable when `allow_write` is True.
            run_id: Optional run identifier used to isolate writable work dirs.

        Returns:
            A result dict with either:
            - `ok=True` and execution fields (`stdout`, `stderr`, `exit_code`, ...)
            - `ok=False` and an `error` object.
        """
        if self.sandbox.backend != "docker":
            return {
                "ok": False,
                "error": {
                    "code": "unsupported_backend",
                    "message": f"Unsupported sandbox backend '{self.sandbox.backend}'",
                },
            }

        if self.sandbox.enforce_pinned_image and "@sha256:" not in self.image:
            return {
                "ok": False,
                "error": {
                    "code": "image_not_pinned",
                    "message": "Docker image must be pinned by digest",
                    "details": {"image": self.image},
                },
            }

        available, probe_error = self._probe_docker()
        if not available:
            return {
                "ok": False,
                "error": {
                    "code": "docker_unavailable",
                    "message": "Docker daemon is unavailable or inaccessible",
                    "details": probe_error or {},
                },
            }

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
        mounts = ["-v", f"{self.workspace_root}:/workspace:ro"]
        workdir = "/workspace"

        if allow_write:
            default_subdir = f".overmind_runs/{run_id}" if run_id else ".overmind_runs/session"
            subdir = (write_subdir or default_subdir).lstrip("/")
            host_write = Path(self.workspace_root, subdir).resolve()
            if str(host_write).startswith(self.workspace_root):
                host_write.mkdir(parents=True, exist_ok=True)
                mounts.extend(["-v", f"{host_write}:/workspace_writable:rw"])
                workdir = "/workspace_writable"

        docker_cmd = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--cpus",
            self.cpu_limit,
            "--memory",
            self.memory_limit,
            "--pids-limit",
            str(self.sandbox.pids_limit),
            "--tmpfs",
            f"/tmp:rw,size={self.sandbox.writable_tmp_mb}m",
            "--workdir",
            workdir,
            *mounts,
            self.image,
            "sh",
            "-lc",
            command,
        ]

        start = time.monotonic()
        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            latency_ms = int((time.monotonic() - start) * 1000)
            return {
                "ok": result.returncode == 0,
                "exit_code": result.returncode,
                "stdout": result.stdout[:100_000],
                "stderr": result.stderr[:100_000],
                "latency_ms": latency_ms,
                "command": command,
                "docker_cmd": " ".join(shlex.quote(part) for part in docker_cmd[:-1])
                + " '<redacted-command>'",
            }
        except subprocess.TimeoutExpired:
            latency_ms = int((time.monotonic() - start) * 1000)
            return {
                "ok": False,
                "error": {
                    "code": "timeout",
                    "message": f"Command timed out after {timeout}s",
                    "details": {
                        "timeout_s": timeout,
                        "backend": self.sandbox.backend,
                    },
                },
                "latency_ms": latency_ms,
                "command": command,
            }
