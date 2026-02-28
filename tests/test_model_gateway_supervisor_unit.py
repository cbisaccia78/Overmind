from __future__ import annotations

from unittest.mock import MagicMock

from app.model_gateway import ModelGateway


def test_advise_supervisor_uses_deterministic_fallback_and_persists_call():
    repo = MagicMock()
    gateway = ModelGateway(repo=repo, openai_tools_provider=None)

    result = gateway.advise_supervisor(
        task="navigate to docs and summarize",
        agent={"id": "a1", "model": "stub-v1"},
        context={"run_id": "r1"},
        state={"directive": {"mode": "execution", "phase": "execute_objective"}},
    )

    assert result["ok"] is True
    assert result["advice"]["mode"] in {"exploration", "execution", "recovery"}
    assert isinstance(result["advice"]["micro_plan"], list)
    repo.create_model_call.assert_called_once()


def test_sanitize_supervisor_advice_clamps_invalid_payload():
    advice = ModelGateway._sanitize_supervisor_advice(
        {
            "mode": "not-a-mode",
            "phase": "",
            "rationale": "",
            "micro_plan": ["", "first", 2, "third"],
            "success_criteria": [],
        }
    )

    assert advice["mode"] == "execution"
    assert advice["phase"] == "execute_objective"
    assert advice["rationale"]
    assert advice["micro_plan"]
    assert advice["success_criteria"]


def test_summarize_context_uses_deterministic_fallback_and_persists_call():
    repo = MagicMock()
    gateway = ModelGateway(repo=repo, openai_tools_provider=None)

    result = gateway.summarize_context(
        task="collect stories",
        agent={"id": "a1", "model": "stub-v1"},
        context={"run_id": "r2"},
        state={
            "older_history": [
                {
                    "step_type": "tool",
                    "tool_name": "mcp.playwright.browser_snapshot",
                    "status": "ok",
                },
                {
                    "step_type": "tool",
                    "tool_name": "mcp.playwright.browser_evaluate",
                    "status": "failed",
                    "error": "timeout: request timed out",
                },
            ]
        },
    )

    assert result["ok"] is True
    assert result["summary"]["objective_status"]
    assert isinstance(result["summary"]["next_focus"], list)
    assert result["summary"]["attempted_paths"]
    repo.create_model_call.assert_called_once()


def test_sanitize_context_summary_clamps_invalid_payload():
    summary = ModelGateway._sanitize_context_summary(
        {
            "objective_status": "",
            "progress_summary": "",
            "completed_milestones": ["", "first", 2],
            "open_issues": "not-a-list",
            "attempted_paths": [],
            "constraints": [None, "blocked by captcha"],
            "next_focus": [],
        }
    )

    assert summary["objective_status"] == "objective status unknown"
    assert summary["progress_summary"]
    assert summary["completed_milestones"] == ["first", "2"]
    assert summary["constraints"] == ["blocked by captcha"]
    assert summary["next_focus"]
