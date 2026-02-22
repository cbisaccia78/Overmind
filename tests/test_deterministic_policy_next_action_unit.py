from __future__ import annotations

from app.model_driven_policy import ModelDrivenPolicy


class _QueueGateway:
    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def infer(self, *, task: str, agent: dict, context: dict) -> dict:
        self.calls.append({"task": task, "agent": agent, "context": context})
        return self.responses.pop(0)


def test_next_action_uses_model_for_first_step():
    gateway = _QueueGateway(
        [
            {
                "ok": True,
                "tool_name": "read_file",
                "args": {"path": "README.md"},
            }
        ]
    )
    policy = ModelDrivenPolicy(model_gateway=gateway)

    action = policy.next_action(
        "read the readme",
        agent={"tools_allowed": ["read_file"]},
        context={"run_id": "r1"},
        history=[],
    )

    assert action == {
        "kind": "tool_call",
        "tool_name": "read_file",
        "args": {"path": "README.md"},
    }
    assert "Task: read the readme" in gateway.calls[0]["task"]


def test_next_action_does_not_apply_keyword_heuristics():
    gateway = _QueueGateway(
        [
            {
                "ok": True,
                "tool_name": "run_shell",
                "args": {"command": "echo delegated"},
            }
        ]
    )
    policy = ModelDrivenPolicy(model_gateway=gateway)

    action = policy.next_action(
        "please curl this site",
        agent={"tools_allowed": ["run_shell"]},
        context={"run_id": "r1"},
        history=[],
    )

    assert action == {
        "kind": "tool_call",
        "tool_name": "run_shell",
        "args": {"command": "echo delegated"},
    }


def test_next_action_successful_tool_asks_model_for_followup():
    gateway = _QueueGateway(
        [
            {
                "ok": True,
                "tool_name": "final_answer",
                "args": {"message": "done"},
            }
        ]
    )
    policy = ModelDrivenPolicy(model_gateway=gateway)

    action = policy.next_action(
        "summarize file",
        agent={"tools_allowed": ["read_file", "final_answer"]},
        context={"run_id": "r1"},
        history=[
            {
                "step_type": "tool",
                "input": {"tool_name": "read_file", "args": {"path": "README.md"}},
                "output": {"ok": True, "stdout": "hello"},
            }
        ],
    )

    assert action == {"kind": "final_answer", "message": "done"}
    assert "Last tool: read_file" in gateway.calls[0]["task"]
    assert "Tool completed with exit_code=None" in gateway.calls[0]["task"]


def test_next_action_failure_replans_with_failure_context():
    gateway = _QueueGateway(
        [
            {
                "ok": True,
                "tool_name": "run_shell",
                "args": {"command": "echo retry"},
            }
        ]
    )
    policy = ModelDrivenPolicy(model_gateway=gateway)

    action = policy.next_action(
        "do task",
        agent={"tools_allowed": ["run_shell"]},
        context={"run_id": "r1"},
        history=[
            {
                "step_type": "tool",
                "input": {"tool_name": "run_shell", "args": {"command": "bad"}},
                "output": {"ok": False, "error": {"message": "timeout"}},
            }
        ],
    )

    assert action == {
        "kind": "tool_call",
        "tool_name": "run_shell",
        "args": {"command": "echo retry"},
    }
    assert "Most recent failure" in gateway.calls[0]["task"]
    assert "timeout" in gateway.calls[0]["task"]


def test_next_action_missing_tool_call_error_asks_user():
    gateway = _QueueGateway(
        [
            {
                "ok": False,
                "error": {"message": "openai response missing valid tool call"},
            }
        ]
    )
    policy = ModelDrivenPolicy(model_gateway=gateway)

    action = policy.next_action(
        "do task",
        agent={"tools_allowed": ["run_shell"]},
        context={"run_id": "r1"},
        history=[],
    )

    assert action == {
        "kind": "ask_user",
        "message": (
            "I could not determine the next tool action from the model response. "
            "Please provide a specific next step."
        ),
    }


def test_next_action_other_inference_error_returns_final_answer():
    gateway = _QueueGateway(
        [
            {
                "ok": False,
                "error": {"message": "openai request failed: timeout"},
            }
        ]
    )
    policy = ModelDrivenPolicy(model_gateway=gateway)

    action = policy.next_action(
        "do task",
        agent={"tools_allowed": ["run_shell"]},
        context={"run_id": "r1"},
        history=[],
    )

    assert action == {
        "kind": "final_answer",
        "message": "I could not infer the next tool action: openai request failed: timeout",
    }


def test_next_action_after_non_tool_step_still_delegates_to_model():
    gateway = _QueueGateway(
        [
            {
                "ok": True,
                "tool_name": "read_file",
                "args": {"path": "notes.txt"},
            }
        ]
    )
    policy = ModelDrivenPolicy(model_gateway=gateway)

    action = policy.next_action(
        "continue",
        agent={"tools_allowed": ["read_file"]},
        context={"run_id": "r1"},
        history=[
            {
                "step_type": "ask_user",
                "input": {"prompt": "Need clarification"},
                "output": {"ok": True, "status": "awaiting_input"},
            }
        ],
    )

    assert action == {
        "kind": "tool_call",
        "tool_name": "read_file",
        "args": {"path": "notes.txt"},
    }
    assert "Recent run history" in gateway.calls[0]["task"]
    assert "ask_user" in gateway.calls[0]["task"]


def test_action_from_inference_maps_ask_user_and_final_answer():
    ask = ModelDrivenPolicy._action_from_inference(
        {"tool_name": "ask_user", "args": {"message": "Need input"}}
    )
    done = ModelDrivenPolicy._action_from_inference(
        {"tool_name": "final_answer", "args": {"message": "All done"}}
    )

    assert ask == {"kind": "ask_user", "message": "Need input"}
    assert done == {"kind": "final_answer", "message": "All done"}


def test_summarize_tool_output_and_mcp_text_helpers():
    summary = ModelDrivenPolicy._summarize_tool_output(
        {
            "exit_code": 0,
            "stdout": "ok",
        }
    )
    assert "Tool completed with exit_code=0" in summary
    assert "ok" in summary

    mcp_summary = ModelDrivenPolicy._summarize_tool_output(
        {
            "mcp": {
                "result": {
                    "content": [
                        {"type": "text", "text": "from mcp"},
                    ]
                }
            }
        }
    )
    assert "from mcp" in mcp_summary


def test_render_recent_history_includes_tools_and_statuses():
    rendered = ModelDrivenPolicy._render_recent_history(
        [
            {
                "step_type": "tool",
                "input": {"tool_name": "a"},
                "output": {"ok": True},
            },
            {
                "step_type": "tool",
                "input": {"tool_name": "b"},
                "output": {"ok": False},
            },
            {
                "step_type": "verify",
                "output": {"ok": True},
            },
            {"step_type": "ask_user", "output": {"ok": True}},
        ]
    )

    assert "tool a -> ok" in rendered
    assert "tool b -> failed" in rendered
    assert "verify -> ok" in rendered
    assert "ask_user" in rendered
