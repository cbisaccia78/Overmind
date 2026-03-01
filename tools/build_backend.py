from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    templates_dir = repo_root / "app" / "templates"
    static_dir = repo_root / "app" / "static"
    entrypoint = repo_root / "app" / "desktop_backend.py"
    dist_dir = repo_root / "build" / "backend"
    work_dir = repo_root / "build" / "pyinstaller"
    spec_dir = work_dir

    data_sep = ";" if os.name == "nt" else ":"

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--name",
        "overmind-backend",
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(work_dir),
        "--specpath",
        str(spec_dir),
        "--add-data",
        f"{templates_dir}{data_sep}app/templates",
        "--add-data",
        f"{static_dir}{data_sep}app/static",
        "--copy-metadata",
        "fastmcp",
        str(entrypoint),
    ]

    subprocess.run(cmd, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
