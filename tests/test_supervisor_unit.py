from __future__ import annotations

from app.supervisor import Supervisor


class _AdviceGateway:
    def __init__(self, response: dict | list[dict]):
        if isinstance(response, list):
            self.responses = response or [{}]
        else:
            self.responses = [response]
        self.calls: list[dict] = []
        self._idx = 0

    def advise_supervisor(self, *, task: str, agent: dict, context: dict, state: dict):
        self.calls.append(
            {"task": task, "agent": agent, "context": context, "state": state}
        )
        if self._idx < len(self.responses):
            response = self.responses[self._idx]
            self._idx += 1
            return response
        return self.responses[-1]


def test_supervisor_heuristic_without_model_gateway():
    supervisor = Supervisor()
    directive = supervisor.decide(
        task="navigate and inspect",
        history=[],
        next_idx=1,
        step_limit=20,
        agent={"id": "a1", "model": "stub-v1"},
        context={"run_id": "r1"},
    )

    payload = directive.as_dict()
    assert payload["source"] == "heuristic"
    assert payload["mode"] in {"exploration", "execution", "recovery"}
    assert payload["budget"]["max_steps"] >= 1


def test_supervisor_merges_llm_advice_with_clamps():
    gateway = _AdviceGateway(
        {
            "ok": True,
            "advice": {
                "mode": "recovery",
                "phase": "recover",
                "rationale": "recent low progress",
                "micro_plan": [
                    "try alternate action",
                    "validate effect",
                    "if blocked ask user",
                ],
                "success_criteria": ["state changes", "no repeated low-progress loops"],
                "budget": {"max_steps": 999},  # should be ignored
            },
        }
    )
    supervisor = Supervisor(model_gateway=gateway)
    directive = supervisor.decide(
        task="do task",
        history=[
            {
                "step_type": "tool",
                "input": {"tool_name": "mcp.playwright.browser_snapshot"},
                "output": {
                    "ok": True,
                    "validation": {"low_progress": True},
                    "observation": {"summary": "snapshot"},
                },
            }
        ],
        next_idx=3,
        step_limit=10,
        agent={"id": "a1", "model": "gpt-5.2"},
        context={"run_id": "r1"},
    )

    payload = directive.as_dict()
    assert payload["source"] == "llm"
    assert payload["mode"] == "recovery"
    assert payload["phase"] == "recover"
    assert payload["micro_plan"][0] == "try alternate action"
    assert payload["budget"]["max_steps"] <= 8  # deterministic budget ownership
    assert gateway.calls


def test_supervisor_advice_is_adaptively_gated_in_stable_execution():
    gateway = _AdviceGateway(
        {
            "ok": True,
            "advice": {
                "mode": "execution",
                "phase": "execute_objective",
                "rationale": "noop",
                "micro_plan": ["noop"],
                "success_criteria": ["noop"],
            },
        }
    )
    supervisor = Supervisor(model_gateway=gateway)

    history = []
    for idx in range(4):
        history.append(
            {
                "step_type": "tool",
                "input": {"tool_name": "mcp.playwright.browser_click"},
                "output": {"ok": True, "validation": {"low_progress": False}},
            }
        )

    directive = supervisor.decide(
        task="continue execution",
        history=history,
        next_idx=5,
        step_limit=30,
        agent={"id": "a1", "model": "gpt-5.2"},
        context={"run_id": "r1"},
    )

    payload = directive.as_dict()
    assert payload["source"] == "heuristic"
    assert payload["advisory"]["attempted"] is False
    assert payload["advisory"]["reason"] == "stable_execution"
    assert not gateway.calls


def test_supervisor_gate_cooldown_and_disable_on_repeated_bad_advice():
    gateway = _AdviceGateway({"ok": False, "error": {"message": "timeout"}})
    supervisor = Supervisor(model_gateway=gateway)

    history = [
        {
            "step_type": "tool",
            "input": {"tool_name": "mcp.playwright.browser_snapshot"},
            "output": {"ok": True, "validation": {"low_progress": True}},
        },
        {
            "step_type": "tool",
            "input": {"tool_name": "mcp.playwright.browser_snapshot"},
            "output": {"ok": True, "validation": {"low_progress": True}},
        },
    ]

    _ = supervisor.decide(
        task="recover progress",
        history=history,
        next_idx=4,
        step_limit=30,
        agent={"id": "a1", "model": "gpt-5.2"},
        context={"run_id": "r2"},
    )
    first_attempts = len(gateway.calls)
    assert first_attempts == 1

    cooled = supervisor.decide(
        task="recover progress",
        history=history,
        next_idx=5,
        step_limit=30,
        agent={"id": "a1", "model": "gpt-5.2"},
        context={"run_id": "r2"},
    )
    assert len(gateway.calls) == first_attempts  # cooldown skip
    assert cooled.as_dict()["advisory"]["reason"] == "gate_cooldown"

    # Next eligible index re-attempts and after repeated failure will disable.
    _ = supervisor.decide(
        task="recover progress",
        history=history,
        next_idx=6,
        step_limit=30,
        agent={"id": "a1", "model": "gpt-5.2"},
        context={"run_id": "r2"},
    )
    assert len(gateway.calls) == 2

    disabled = supervisor.decide(
        task="recover progress",
        history=history,
        next_idx=7,
        step_limit=30,
        agent={"id": "a1", "model": "gpt-5.2"},
        context={"run_id": "r2"},
    )
    assert len(gateway.calls) == 2
    assert disabled.as_dict()["advisory"]["reason"] in {
        "gate_cooldown",
        "gate_disabled",
    }


def test_supervisor_advice_error_includes_advisory_metadata():
    gateway = _AdviceGateway({"ok": False, "error": {"message": "timeout"}})
    supervisor = Supervisor(model_gateway=gateway)

    directive = supervisor.decide(
        task="recover progress",
        history=[
            {
                "step_type": "tool",
                "input": {"tool_name": "mcp.playwright.browser_snapshot"},
                "output": {"ok": True, "validation": {"low_progress": True}},
            }
        ],
        next_idx=3,
        step_limit=30,
        agent={"id": "a1", "model": "gpt-5.2"},
        context={"run_id": "r-meta"},
    )

    payload = directive.as_dict()
    assert payload["source"] == "heuristic"
    assert payload["advisory"]["attempted"] is True
    assert payload["advisory"]["applied"] is False
    assert payload["advisory"]["reason"] == "advice_error"
    assert payload["advisory"]["gate"]["total_attempts"] == 1


def test_supervisor_outcome_feedback_cools_down_advisory_attempts():
    gateway = _AdviceGateway(
        {
            "ok": True,
            "advice": {
                "mode": "recovery",
                "phase": "recover_v2",
                "rationale": "switch path",
                "micro_plan": ["use a different interaction path"],
                "success_criteria": ["state changes"],
            },
        }
    )
    supervisor = Supervisor(model_gateway=gateway)

    history = [
        {
            "step_type": "tool",
            "input": {"tool_name": "mcp.playwright.browser_snapshot"},
            "output": {"ok": True, "validation": {"low_progress": True}},
        }
    ]
    applied = supervisor.decide(
        task="recover progress",
        history=history,
        next_idx=3,
        step_limit=30,
        agent={"id": "a1", "model": "gpt-5.2"},
        context={"run_id": "r-feedback"},
    )
    assert applied.as_dict()["advisory"]["applied"] is True
    assert len(gateway.calls) == 1

    history.append(
        {
            "step_type": "tool",
            "input": {"tool_name": "mcp.playwright.browser_click"},
            "output": {"ok": True, "validation": {"low_progress": True}},
        }
    )
    cooled = supervisor.decide(
        task="recover progress",
        history=history,
        next_idx=4,
        step_limit=30,
        agent={"id": "a1", "model": "gpt-5.2"},
        context={"run_id": "r-feedback"},
    )
    payload = cooled.as_dict()
    assert payload["advisory"]["attempted"] is False
    assert payload["advisory"]["reason"] == "gate_cooldown"
    assert payload["advisory"]["gate"]["total_outcome_failures"] >= 1
    assert len(gateway.calls) == 1
