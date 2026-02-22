from __future__ import annotations

from app.deterministic_policy import DeterministicPolicy


class _QueueGateway:
    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def infer(self, *, task: str, agent: dict, context: dict) -> dict:
        self.calls.append({"task": task, "agent": agent, "context": context})
        return self.responses.pop(0)


def test_next_action_asks_for_url_when_curl_missing_url():
    policy = DeterministicPolicy(model_gateway=_QueueGateway([]))

    action = policy.next_action(
        "please curl this site",
        agent={"tools_allowed": ["run_shell"]},
        context={"run_id": "r1"},
        history=[],
    )

    assert action == {
        "kind": "ask_user",
        "message": "Please provide the full URL to fetch.",
    }


def test_next_action_builds_curl_tool_call_with_url():
    policy = DeterministicPolicy(model_gateway=_QueueGateway([]))

    action = policy.next_action(
        "curl https://example.com now",
        agent={"tools_allowed": ["run_shell"]},
        context={"run_id": "r1"},
        history=[],
    )

    assert action["kind"] == "tool_call"
    assert action["tool_name"] == "run_shell"
    assert action["args"]["command"] == "curl -L --max-time 20 https://example.com"


def test_next_action_returns_inference_error_without_tool_steps():
    policy = DeterministicPolicy(
        model_gateway=_QueueGateway(
            [
                {"ok": False, "error": {"message": "no model"}},
            ]
        )
    )

    action = policy.next_action(
        "do something",
        agent={"tools_allowed": ["read_file"]},
        context={"run_id": "r1"},
        history=[],
    )

    assert action["kind"] == "final_answer"
    assert "I could not infer a tool action: no model" == action["message"]


def test_next_action_tool_failure_and_final_answer_passthrough():
    policy = DeterministicPolicy(model_gateway=_QueueGateway([]))

    failed_action = policy.next_action(
        "task",
        agent={"tools_allowed": []},
        context={"run_id": "r1"},
        history=[
            {
                "step_type": "tool",
                "input": {"tool_name": "read_file"},
                "output": {"ok": False, "error": {"message": "boom"}},
            }
        ],
    )
    assert failed_action == {"kind": "final_answer", "message": "Tool failed: boom"}

    done_action = policy.next_action(
        "task",
        agent={"tools_allowed": []},
        context={"run_id": "r1"},
        history=[
            {
                "step_type": "tool",
                "input": {"tool_name": "final_answer"},
                "output": {"ok": True, "message": "all done"},
            }
        ],
    )
    assert done_action == {"kind": "final_answer", "message": "all done"}


def test_next_action_non_openai_summarizes_last_output(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    policy = DeterministicPolicy(model_gateway=_QueueGateway([]))

    action = policy.next_action(
        "task",
        agent={"tools_allowed": ["run_shell"]},
        context={"run_id": "r1"},
        history=[
            {
                "step_type": "tool",
                "input": {"tool_name": "run_shell"},
                "output": {"ok": True, "stdout": "hello", "exit_code": 0},
            }
        ],
    )

    assert action["kind"] == "final_answer"
    assert "exit_code=0" in action["message"]
    assert "hello" in action["message"]


def test_next_action_openai_report_flow_and_heuristics(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    policy = DeterministicPolicy(model_gateway=_QueueGateway([]))

    report_action = policy.next_action(
        "analyze and report this",
        agent={"tools_allowed": ["run_shell", "store_memory"]},
        context={"run_id": "r1"},
        history=[
            {
                "step_type": "tool",
                "input": {"tool_name": "run_shell"},
                "output": {"ok": True, "stdout": "content", "exit_code": 0},
            }
        ],
    )
    assert report_action["kind"] == "tool_call"
    assert report_action["tool_name"] == "store_memory"
    assert report_action["args"]["metadata"] == {"source": "analysis"}

    write_done = policy.next_action(
        "task",
        agent={"tools_allowed": ["run_shell"]},
        context={"run_id": "r1"},
        history=[
            {
                "step_type": "tool",
                "input": {"tool_name": "write_file"},
                "output": {"ok": True, "path": "notes.txt"},
            }
        ],
    )
    assert write_done == {"kind": "final_answer", "message": "Wrote file: notes.txt"}

    poem_done = policy.next_action(
        "task",
        agent={"tools_allowed": ["run_shell"]},
        context={"run_id": "r1"},
        history=[
            {
                "step_type": "tool",
                "input": {"tool_name": "run_shell"},
                "output": {"ok": True, "command": "echo hi > poem.txt", "exit_code": 0},
            }
        ],
    )
    assert poem_done == {
        "kind": "final_answer",
        "message": "Created poem.txt via run_shell.",
    }


def test_next_action_openai_followup_infer_paths(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    gateway = _QueueGateway(
        [
            {"ok": False, "error": {"message": "planner unavailable"}},
            {"ok": True, "tool_name": "read_file", "args": {"path": "README.md"}},
        ]
    )
    policy = DeterministicPolicy(model_gateway=gateway)

    first = policy.next_action(
        "continue task",
        agent={"tools_allowed": ["read_file"]},
        context={"run_id": "r1"},
        history=[
            {
                "step_type": "tool",
                "input": {"tool_name": "run_shell"},
                "output": {"ok": True, "stdout": "x", "exit_code": 0},
            }
        ],
    )
    assert first == {
        "kind": "final_answer",
        "message": "I could not infer the next tool action: planner unavailable",
    }

    second = policy.next_action(
        "continue task",
        agent={"tools_allowed": ["read_file"]},
        context={"run_id": "r1"},
        history=[
            {
                "step_type": "tool",
                "input": {"tool_name": "run_shell"},
                "output": {"ok": True, "stdout": "y", "exit_code": 0},
            }
        ],
    )
    assert second == {
        "kind": "tool_call",
        "tool_name": "read_file",
        "args": {"path": "README.md"},
    }
    assert "Last tool result summary" in gateway.calls[0]["task"]


def test_policy_helpers_extract_url_and_truncate_summary():
    assert DeterministicPolicy._extract_url("visit https://example.com/path now") == (
        "https://example.com/path"
    )
    assert DeterministicPolicy._extract_url("no url") is None

    long_stdout = "x" * 1200
    summary = DeterministicPolicy._summarize_tool_output(
        {"exit_code": 0, "stdout": long_stdout, "stderr": "ignored"}
    )
    assert "exit_code=0" in summary
    assert summary.endswith("...")
    assert len(summary) < 900
