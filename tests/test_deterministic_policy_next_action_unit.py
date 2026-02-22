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


def test_next_action_openai_failure_replans_with_memory(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    gateway = _QueueGateway(
        [
            {"ok": True, "tool_name": "read_file", "args": {"path": "README.md"}},
        ]
    )
    policy = DeterministicPolicy(model_gateway=gateway)

    action = policy.next_action(
        "complete the task",
        agent={"tools_allowed": ["run_shell", "read_file"]},
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
        "tool_name": "read_file",
        "args": {"path": "README.md"},
    }
    assert "Prior failures" in gateway.calls[0]["task"]


def test_next_action_openai_failure_repeated_candidate_asks_user(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    gateway = _QueueGateway(
        [
            {"ok": True, "tool_name": "run_shell", "args": {"command": "bad"}},
        ]
    )
    policy = DeterministicPolicy(model_gateway=gateway)

    action = policy.next_action(
        "complete the task",
        agent={"tools_allowed": ["run_shell", "read_file"]},
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
        "kind": "ask_user",
        "message": "The next planned action repeats a previously failed tool call. Please provide additional guidance.",
    }


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


def test_next_action_openai_missing_tool_call_asks_user(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    gateway = _QueueGateway(
        [
            {
                "ok": False,
                "error": {"message": "openai response missing valid tool call"},
            }
        ]
    )
    policy = DeterministicPolicy(model_gateway=gateway)

    action = policy.next_action(
        "continue task",
        agent={"tools_allowed": ["mcp.playwright.browser_navigate"]},
        context={"run_id": "r1", "planning_contract": {"mode": "generic"}},
        history=[
            {
                "step_type": "tool",
                "input": {
                    "tool_name": "mcp.playwright.browser_navigate",
                    "args": {"url": "https://x.com/home"},
                },
                "output": {"ok": True},
            }
        ],
    )

    assert action["kind"] == "ask_user"
    assert "could not determine the next tool action" in action["message"]


def test_next_action_openai_followup_maps_ask_user_and_final_answer(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    gateway = _QueueGateway(
        [
            {
                "ok": True,
                "tool_name": "ask_user",
                "args": {"message": "Please log in, then reply continue."},
            },
            {
                "ok": True,
                "tool_name": "final_answer",
                "args": {"message": "All done."},
            },
        ]
    )
    policy = DeterministicPolicy(model_gateway=gateway)

    ask_action = policy.next_action(
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
    assert ask_action == {
        "kind": "ask_user",
        "message": "Please log in, then reply continue.",
    }

    done_action = policy.next_action(
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
    assert done_action == {"kind": "final_answer", "message": "All done."}


def test_next_action_generic_repeated_success_loop_asks_user(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    policy = DeterministicPolicy(model_gateway=_QueueGateway([]))

    repeated_history = [
        {
            "step_type": "tool",
            "input": {
                "tool_name": "mcp.playwright.browser_navigate",
                "args": {"url": "https://x.com"},
            },
            "output": {"ok": True},
        }
        for _ in range(5)
    ]

    action = policy.next_action(
        "navigate around x.com and keep exploring",
        agent={"tools_allowed": ["mcp.playwright.browser_navigate"]},
        context={"run_id": "r1", "planning_contract": {"mode": "generic"}},
        history=repeated_history,
    )

    assert action == {
        "kind": "ask_user",
        "message": "I am repeating the same successful action without making progress. Please clarify what should happen next.",
    }


def test_next_action_repeated_success_loop_resets_after_ask_user_boundary(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    gateway = _QueueGateway(
        [
            {
                "ok": True,
                "tool_name": "store_memory",
                "args": {
                    "collection": "runs",
                    "text": "continue with a different step",
                    "metadata": {"source": "analysis"},
                },
            }
        ]
    )
    policy = DeterministicPolicy(model_gateway=gateway)

    repeated_history = [
        {
            "step_type": "tool",
            "input": {
                "tool_name": "mcp.playwright.browser_navigate",
                "args": {"url": "https://x.com"},
            },
            "output": {"ok": True},
        }
        for _ in range(5)
    ]
    repeated_history.append(
        {
            "step_type": "ask_user",
            "input": {"prompt": "Please clarify what should happen next."},
            "output": {"ok": True, "status": "awaiting_input"},
        }
    )
    repeated_history.append(
        {
            "step_type": "tool",
            "input": {
                "tool_name": "mcp.playwright.browser_navigate",
                "args": {"url": "https://x.com"},
            },
            "output": {"ok": True},
        }
    )

    action = policy.next_action(
        "navigate around x.com and keep exploring",
        agent={
            "tools_allowed": [
                "mcp.playwright.browser_navigate",
                "store_memory",
            ]
        },
        context={"run_id": "r1", "planning_contract": {"mode": "generic"}},
        history=repeated_history,
    )

    assert action == {
        "kind": "tool_call",
        "tool_name": "store_memory",
        "args": {
            "collection": "runs",
            "text": "continue with a different step",
            "metadata": {"source": "analysis"},
        },
    }


def test_next_action_wait_for_user_pauses_after_successful_navigation(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    gateway = _QueueGateway(
        [
            {
                "ok": True,
                "tool_name": "mcp.playwright.browser_snapshot",
                "args": {},
            }
        ]
    )
    policy = DeterministicPolicy(model_gateway=gateway)

    action = policy.next_action(
        "navigate to x.com and then wait for me to tell you next steps",
        agent={"tools_allowed": ["mcp.playwright.browser_navigate"]},
        context={"run_id": "r1", "planning_contract": {"mode": "generic"}},
        history=[
            {
                "step_type": "tool",
                "input": {
                    "tool_name": "mcp.playwright.browser_navigate",
                    "args": {"url": "https://x.com"},
                },
                "output": {"ok": True},
            }
        ],
    )

    assert action == {
        "kind": "ask_user",
        "message": "Reached the requested page. Waiting for your next instruction.",
    }
    assert gateway.calls == []


def test_next_action_uses_latest_user_input_for_wait_detection(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    gateway = _QueueGateway(
        [
            {
                "ok": True,
                "tool_name": "mcp.playwright.browser_snapshot",
                "args": {},
            }
        ]
    )
    policy = DeterministicPolicy(model_gateway=gateway)

    action = policy.next_action(
        (
            "navigate to x.com and then wait for me to tell you next steps.\n\n"
            "User input: start interacting with content in a way that generates "
            "user feedback"
        ),
        agent={
            "tools_allowed": [
                "mcp.playwright.browser_navigate",
                "mcp.playwright.browser_snapshot",
            ]
        },
        context={
            "run_id": "r1",
            "planning_contract": {
                "mode": "web_interaction_feedback",
                "min_exploration_steps": 1,
                "min_interaction_steps": 1,
                "min_feedback_signals": 1,
            },
            "progress_state": {
                "exploration_count": 0,
                "interaction_count": 0,
                "feedback_signal_count": 0,
                "consecutive_same_success": 1,
                "contract_satisfied": False,
            },
        },
        history=[
            {
                "step_type": "tool",
                "input": {
                    "tool_name": "mcp.playwright.browser_navigate",
                    "args": {"url": "https://x.com"},
                },
                "output": {"ok": True},
            }
        ],
    )

    assert action == {
        "kind": "tool_call",
        "tool_name": "mcp.playwright.browser_snapshot",
        "args": {},
    }
    assert len(gateway.calls) == 1


def test_next_action_emits_verify_for_screenshot_artifacts(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    policy = DeterministicPolicy(model_gateway=_QueueGateway([]))

    action = policy.next_action(
        "open wikipedia and take a screenshot",
        agent={"tools_allowed": ["mcp.browser_take_screenshot"]},
        context={"run_id": "r1"},
        history=[
            {
                "step_type": "tool",
                "input": {"tool_name": "mcp.browser_take_screenshot"},
                "output": {"ok": True, "filename": "wikipedia.png"},
            }
        ],
    )

    assert action["kind"] == "verify"
    assert action["checks"][0] == {"type": "file_exists", "path": "wikipedia.png"}
    assert action["checks"][1] == {
        "type": "file_min_bytes",
        "path": "wikipedia.png",
        "min_bytes": 8000,
    }
    assert action["checks"][2] == {
        "type": "png_signature",
        "path": "wikipedia.png",
    }
    assert action["checks"][3] == {
        "type": "png_dimensions_min",
        "path": "wikipedia.png",
        "min_width": 600,
        "min_height": 400,
    }
    assert action["checks"][4] == {
        "type": "file_entropy_min",
        "path": "wikipedia.png",
        "min_entropy": 3.5,
    }


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

    mcp_summary = DeterministicPolicy._summarize_tool_output(
        {
            "mcp": {
                "result": {
                    "content": [
                        {"type": "text", "text": "### Page snapshot"},
                        {"type": "text", "text": '- button "Like" [ref=e22]'},
                    ]
                }
            }
        }
    )
    assert "Page snapshot" in mcp_summary
    assert "[ref=e22]" in mcp_summary


def test_next_action_uses_contract_prompt_for_wikipedia_research(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    gateway = _QueueGateway(
        [
            {
                "ok": True,
                "tool_name": "mcp.playwright.browser_navigate",
                "args": {"url": "https://en.wikipedia.org/wiki/Astronomy"},
            }
        ]
    )
    policy = DeterministicPolicy(model_gateway=gateway)

    action = policy.next_action(
        "go to wikipedia and navigate pages, summarize each, and take screenshots",
        agent={"tools_allowed": ["mcp.playwright.browser_navigate", "final_answer"]},
        context={
            "run_id": "r1",
            "planning_contract": {
                "mode": "multi_page_research",
                "domain": "wikipedia.org",
                "min_unique_pages": 3,
                "require_screenshot_per_page": True,
                "require_summary_per_page": True,
            },
            "progress_state": {
                "unique_urls": [],
                "unique_pages_done": 0,
                "screenshot_count": 0,
                "summary_count": 0,
                "consecutive_same_success": 0,
                "contract_satisfied": False,
            },
        },
        history=[],
    )

    assert action == {
        "kind": "tool_call",
        "tool_name": "mcp.playwright.browser_navigate",
        "args": {"url": "https://en.wikipedia.org/wiki/Astronomy"},
    }
    assert "Planning contract" in gateway.calls[0]["task"]
    assert "Current progress" in gateway.calls[0]["task"]


def test_next_action_contract_completion_returns_final_answer(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    policy = DeterministicPolicy(model_gateway=_QueueGateway([]))

    action = policy.next_action(
        "go to wikipedia and navigate pages, summarize each, and take screenshots",
        agent={"tools_allowed": ["mcp.playwright.browser_navigate", "final_answer"]},
        context={
            "run_id": "r1",
            "planning_contract": {
                "mode": "multi_page_research",
                "min_unique_pages": 3,
            },
            "progress_state": {
                "unique_pages_done": 3,
                "screenshot_count": 3,
                "summary_count": 3,
                "consecutive_same_success": 0,
                "contract_satisfied": True,
            },
        },
        history=[
            {
                "step_type": "tool",
                "input": {
                    "tool_name": "mcp.playwright.browser_take_screenshot",
                    "args": {"filename": "a.png"},
                },
                "output": {"ok": True, "filename": "a.png"},
            }
        ],
    )

    assert action["kind"] == "final_answer"
    assert "Completed the browsing contract" in action["message"]


def test_build_task_contract_detects_web_interaction_feedback_goal():
    contract = DeterministicPolicy._build_task_contract(
        "interact with content on x.com in a way that generates user feedback"
    )
    assert contract["mode"] == "web_interaction_feedback"
    assert contract["require_dom_exploration"] is True
    assert contract["require_interaction"] is True
    assert contract["require_feedback_signal"] is True


def test_derive_progress_state_for_web_interaction_feedback_contract():
    contract = {
        "mode": "web_interaction_feedback",
        "min_exploration_steps": 1,
        "min_interaction_steps": 1,
        "min_feedback_signals": 1,
    }
    progress = DeterministicPolicy._derive_progress_state(
        history=[
            {
                "step_type": "tool",
                "input": {
                    "tool_name": "mcp.playwright.browser_snapshot",
                    "args": {},
                },
                "output": {"ok": True},
            },
            {
                "step_type": "tool",
                "input": {
                    "tool_name": "mcp.playwright.browser_click",
                    "args": {"ref": "node-1"},
                },
                "output": {"ok": True},
            },
            {
                "step_type": "tool",
                "input": {
                    "tool_name": "mcp.playwright.browser_wait_for",
                    "args": {"text": "liked"},
                },
                "output": {"ok": True},
            },
        ],
        planning_contract=contract,
    )

    assert progress["exploration_count"] >= 1
    assert progress["interaction_count"] >= 1
    assert progress["feedback_signal_count"] >= 1
    assert progress["contract_satisfied"] is True


def test_next_action_uses_strategy_prompt_for_web_interaction_contract(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    gateway = _QueueGateway(
        [
            {
                "ok": True,
                "tool_name": "mcp.playwright.browser_snapshot",
                "args": {},
            }
        ]
    )
    policy = DeterministicPolicy(model_gateway=gateway)

    action = policy.next_action(
        "interact with content on x.com in a way that generates user feedback",
        agent={"tools_allowed": ["mcp.playwright.browser_snapshot"]},
        context={
            "run_id": "r1",
            "planning_contract": {
                "mode": "web_interaction_feedback",
                "min_exploration_steps": 1,
                "min_interaction_steps": 1,
                "min_feedback_signals": 1,
            },
            "progress_state": {
                "exploration_count": 0,
                "interaction_count": 0,
                "feedback_signal_count": 0,
                "consecutive_same_success": 0,
                "contract_satisfied": False,
            },
        },
        history=[],
    )

    assert action == {
        "kind": "tool_call",
        "tool_name": "mcp.playwright.browser_snapshot",
        "args": {},
    }
    assert "Follow phased execution" in gateway.calls[0]["task"]


def test_next_action_contract_inference_missing_tool_call_uses_fallback_tool(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    gateway = _QueueGateway(
        [
            {
                "ok": False,
                "error": {"message": "openai response missing valid tool call"},
            }
        ]
    )
    policy = DeterministicPolicy(model_gateway=gateway)

    action = policy.next_action(
        "interact with content on x.com in a way that generates user feedback",
        agent={"tools_allowed": ["mcp.playwright.browser_snapshot"]},
        context={
            "run_id": "r1",
            "planning_contract": {
                "mode": "web_interaction_feedback",
                "min_exploration_steps": 1,
                "min_interaction_steps": 1,
                "min_feedback_signals": 1,
            },
            "progress_state": {
                "exploration_count": 0,
                "interaction_count": 0,
                "feedback_signal_count": 0,
                "consecutive_same_success": 0,
                "contract_satisfied": False,
            },
        },
        history=[
            {
                "step_type": "tool",
                "input": {
                    "tool_name": "mcp.playwright.browser_navigate",
                    "args": {"url": "https://x.com"},
                },
                "output": {"ok": True},
            }
        ],
    )

    assert action == {
        "kind": "tool_call",
        "tool_name": "mcp.playwright.browser_snapshot",
        "args": {},
    }


def test_next_action_contract_fallback_click_uses_snapshot_ref(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    gateway = _QueueGateway(
        [
            {
                "ok": False,
                "error": {"message": "openai response missing valid tool call"},
            }
        ]
    )
    policy = DeterministicPolicy(model_gateway=gateway)

    action = policy.next_action(
        "interact with content on x.com in a way that generates user feedback",
        agent={"tools_allowed": ["mcp.playwright.browser_click"]},
        context={
            "run_id": "r1",
            "planning_contract": {
                "mode": "web_interaction_feedback",
                "min_exploration_steps": 1,
                "min_interaction_steps": 1,
                "min_feedback_signals": 1,
            },
            "progress_state": {
                "exploration_count": 1,
                "interaction_count": 0,
                "feedback_signal_count": 0,
                "consecutive_same_success": 0,
                "contract_satisfied": False,
            },
        },
        history=[
            {
                "step_type": "tool",
                "input": {"tool_name": "mcp.playwright.browser_snapshot", "args": {}},
                "output": {
                    "ok": True,
                    "mcp": {
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": '- button "Like" [ref=e197]',
                                }
                            ]
                        }
                    },
                },
            }
        ],
    )

    assert action == {
        "kind": "tool_call",
        "tool_name": "mcp.playwright.browser_click",
        "args": {"ref": "e197"},
    }
