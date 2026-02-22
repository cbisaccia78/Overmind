from __future__ import annotations

from app.deterministic_policy import DeterministicPolicy


class _QueueGateway:
    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def infer(self, *, task: str, agent: dict, context: dict) -> dict:
        self.calls.append({"task": task, "agent": agent, "context": context})
        return self.responses.pop(0)


def test_deterministic_policy_first_step_uses_raw_task_inference():
    gateway = _QueueGateway(
        [{"ok": True, "tool_name": "store_memory", "args": {"text": "x"}}]
    )
    policy = DeterministicPolicy(model_gateway=gateway)

    action = policy.next_action(
        "remember:notes:x",
        agent={"tools_allowed": ["store_memory"]},
        context={"run_id": "r1"},
        history=[],
    )

    assert action == {
        "kind": "tool_call",
        "tool_name": "store_memory",
        "args": {"text": "x"},
    }
    assert gateway.calls[0]["task"] == "remember:notes:x"


def test_deterministic_policy_finishes_after_successful_tool():
    policy = DeterministicPolicy(model_gateway=_QueueGateway([]))

    action = policy.next_action(
        "task",
        agent={"tools_allowed": ["store_memory"]},
        context={"run_id": "r1"},
        history=[
            {
                "step_type": "tool",
                "input": {"tool_name": "store_memory", "args": {"text": "x"}},
                "output": {"ok": True, "stdout": "stored"},
            }
        ],
    )

    assert action["kind"] == "final_answer"
    assert "Tool completed" in action["message"]


def test_deterministic_policy_finishes_after_failed_tool():
    policy = DeterministicPolicy(model_gateway=_QueueGateway([]))

    action = policy.next_action(
        "task",
        agent={"tools_allowed": ["store_memory"]},
        context={"run_id": "r1"},
        history=[
            {
                "step_type": "tool",
                "input": {"tool_name": "store_memory", "args": {"text": "x"}},
                "output": {"ok": False, "error": {"message": "boom"}},
            }
        ],
    )

    assert action == {"kind": "final_answer", "message": "Tool failed: boom"}
