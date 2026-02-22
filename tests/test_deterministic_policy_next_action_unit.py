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


def test_next_action_prompt_prioritizes_latest_user_input():
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
        (
            "navigate to X.com and wait for next steps.\n\n"
            "User input: check the home feed\n\n"
            "User input: write a snarky reply to a popular post"
        ),
        agent={"tools_allowed": ["read_file"]},
        context={"run_id": "r1"},
        history=[],
    )

    assert action == {
        "kind": "tool_call",
        "tool_name": "read_file",
        "args": {"path": "notes.txt"},
    }
    prompt = gateway.calls[0]["task"]
    assert "Task context:" in prompt
    assert "Latest user input (highest priority):" in prompt
    assert "write a snarky reply to a popular post" in prompt
    assert "Prior user inputs:" in prompt


def test_next_action_stall_recovery_prompt_forbids_repeating_identical_action():
    gateway = _QueueGateway(
        [
            {
                "ok": True,
                "tool_name": "mcp.playwright.browser_click",
                "args": {"element": "Reply"},
            }
        ]
    )
    policy = ModelDrivenPolicy(model_gateway=gateway)

    action = policy.next_action(
        "navigate to X.com and start posting",
        agent={
            "tools_allowed": [
                "mcp.playwright.browser_snapshot",
                "mcp.playwright.browser_click",
            ]
        },
        context={"run_id": "r1"},
        history=[
            {
                "step_type": "tool",
                "input": {"tool_name": "mcp.playwright.browser_snapshot", "args": {}},
                "output": {"ok": True, "mcp": {"result": {"content": ["snap1"]}}},
            },
            {
                "step_type": "tool",
                "input": {"tool_name": "mcp.playwright.browser_snapshot", "args": {}},
                "output": {"ok": True, "mcp": {"result": {"content": ["snap2"]}}},
            },
            {
                "step_type": "tool",
                "input": {"tool_name": "mcp.playwright.browser_snapshot", "args": {}},
                "output": {"ok": True, "mcp": {"result": {"content": ["snap3"]}}},
            },
        ],
    )

    assert action == {
        "kind": "tool_call",
        "tool_name": "mcp.playwright.browser_click",
        "args": {"element": "Reply"},
    }
    prompt = gateway.calls[0]["task"]
    assert "Stall detected:" in prompt
    assert "Stall recovery constraint (next turn only):" in prompt
    assert "Pattern observed across the last 3 successful tool actions." in prompt
    assert (
        "Do not call any of these tools in your next action: "
        "`mcp.playwright.browser_snapshot`." in prompt
    )


def test_next_action_reprompts_once_when_model_violates_stall_constraint():
    gateway = _QueueGateway(
        [
            {
                "ok": True,
                "tool_name": "mcp.playwright.browser_snapshot",
                "args": {},
            },
            {
                "ok": True,
                "tool_name": "mcp.playwright.browser_click",
                "args": {"element": "Reply"},
            },
        ]
    )
    policy = ModelDrivenPolicy(model_gateway=gateway)

    action = policy.next_action(
        "navigate to X.com and start posting",
        agent={
            "tools_allowed": [
                "mcp.playwright.browser_snapshot",
                "mcp.playwright.browser_click",
            ]
        },
        context={"run_id": "r1"},
        history=[
            {
                "step_type": "tool",
                "input": {"tool_name": "mcp.playwright.browser_snapshot", "args": {}},
                "output": {"ok": True},
            },
            {
                "step_type": "tool",
                "input": {"tool_name": "mcp.playwright.browser_snapshot", "args": {}},
                "output": {"ok": True},
            },
            {
                "step_type": "tool",
                "input": {"tool_name": "mcp.playwright.browser_snapshot", "args": {}},
                "output": {"ok": True},
            },
        ],
    )

    assert action == {
        "kind": "tool_call",
        "tool_name": "mcp.playwright.browser_click",
        "args": {"element": "Reply"},
    }
    assert len(gateway.calls) == 2
    assert "Stall recovery constraint (next turn only):" in gateway.calls[0]["task"]
    assert "Stall recovery violation:" in gateway.calls[1]["task"]


def test_next_action_stall_recovery_blocks_two_tool_cycle():
    gateway = _QueueGateway(
        [
            {
                "ok": True,
                "tool_name": "mcp.playwright.browser_run_code",
                "args": {"code": "return 1;"},
            }
        ]
    )
    policy = ModelDrivenPolicy(model_gateway=gateway)

    history = []
    for tool_name in [
        "mcp.playwright.browser_snapshot",
        "mcp.playwright.browser_navigate",
        "mcp.playwright.browser_snapshot",
        "mcp.playwright.browser_navigate",
        "mcp.playwright.browser_snapshot",
        "mcp.playwright.browser_navigate",
    ]:
        args = {} if tool_name.endswith("snapshot") else {"url": "https://x.com"}
        history.append(
            {
                "step_type": "tool",
                "input": {"tool_name": tool_name, "args": args},
                "output": {"ok": True},
            }
        )

    action = policy.next_action(
        "navigate to X.com and start posting",
        agent={
            "tools_allowed": [
                "mcp.playwright.browser_snapshot",
                "mcp.playwright.browser_navigate",
                "mcp.playwright.browser_run_code",
            ]
        },
        context={"run_id": "r1"},
        history=history,
    )

    assert action == {
        "kind": "tool_call",
        "tool_name": "mcp.playwright.browser_run_code",
        "args": {"code": "return 1;"},
    }
    prompt = gateway.calls[0]["task"]
    assert "two-tool loop in recent actions" in prompt
    assert "`mcp.playwright.browser_navigate`" in prompt
    assert "`mcp.playwright.browser_snapshot`" in prompt


def test_next_action_does_not_trigger_stall_prompt_before_threshold():
    gateway = _QueueGateway(
        [
            {
                "ok": True,
                "tool_name": "mcp.playwright.browser_snapshot",
                "args": {},
            }
        ]
    )
    policy = ModelDrivenPolicy(model_gateway=gateway)

    action = policy.next_action(
        "navigate to X.com and inspect page",
        agent={"tools_allowed": ["mcp.playwright.browser_snapshot"]},
        context={"run_id": "r1"},
        history=[
            {
                "step_type": "tool",
                "input": {"tool_name": "mcp.playwright.browser_snapshot", "args": {}},
                "output": {"ok": True},
            },
            {
                "step_type": "tool",
                "input": {"tool_name": "mcp.playwright.browser_snapshot", "args": {}},
                "output": {"ok": True},
            },
        ],
    )

    assert action == {
        "kind": "tool_call",
        "tool_name": "mcp.playwright.browser_snapshot",
        "args": {},
    }
    prompt = gateway.calls[0]["task"]
    assert "Stall detected:" not in prompt
    assert "Last tool: mcp.playwright.browser_snapshot" in prompt


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
                "input": {"tool_name": "b", "args": {"url": "https://x.com"}},
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
    assert 'tool b args={"url":"https://x.com"} -> failed' in rendered
    assert "verify -> ok" in rendered
    assert "ask_user" in rendered
