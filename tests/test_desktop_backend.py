from __future__ import annotations

import runpy
import sys
from types import SimpleNamespace

import app.desktop_backend as desktop_backend
import uvicorn


def test_desktop_backend_main_invokes_uvicorn(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_parse_args(self):
        del self
        return SimpleNamespace(host="0.0.0.0", port=9999)

    def _fake_run(app, host, port, log_level):
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port
        captured["log_level"] = log_level

    monkeypatch.setattr(
        desktop_backend.argparse.ArgumentParser,
        "parse_args",
        _fake_parse_args,
    )
    monkeypatch.setattr(desktop_backend.uvicorn, "run", _fake_run)

    desktop_backend.main()

    assert captured == {
        "app": desktop_backend.fastapi_app,
        "host": "0.0.0.0",
        "port": 9999,
        "log_level": "info",
    }


def test_desktop_backend_module_entrypoint(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_run(app, host, port, log_level):
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port
        captured["log_level"] = log_level

    sys.modules.pop("app.desktop_backend", None)
    monkeypatch.setattr(uvicorn, "run", _fake_run)
    monkeypatch.setattr(sys, "argv", ["desktop_backend.py"])

    runpy.run_module("app.desktop_backend", run_name="__main__")

    assert captured == {
        "app": desktop_backend.fastapi_app,
        "host": "127.0.0.1",
        "port": 8765,
        "log_level": "info",
    }
