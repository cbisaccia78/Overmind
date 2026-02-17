from __future__ import annotations

from app.deterministic_policy import DeterministicPolicy


class _FakeModelGateway:
    def __init__(self, mapping: dict[str, dict]):
        self.mapping = mapping

    def infer(self, *, task: str, agent: dict, context: dict) -> dict:
        return self.mapping[task]


def test_policy_parses_all_task_prefixes():
    policy = DeterministicPolicy(
        model_gateway=_FakeModelGateway(
            {
                "shell:echo hi": {
                    "ok": True,
                    "tool_name": "run_shell",
                    "args": {"command": "echo hi"},
                },
                "read:seed.txt": {
                    "ok": True,
                    "tool_name": "read_file",
                    "args": {"path": "seed.txt"},
                },
                "write:notes.txt:hello": {
                    "ok": True,
                    "tool_name": "write_file",
                    "args": {"path": "notes.txt", "content": "hello"},
                },
                "remember:notes:store this": {
                    "ok": True,
                    "tool_name": "store_memory",
                    "args": {"collection": "notes", "text": "store this"},
                },
                "remember::implicit": {
                    "ok": True,
                    "tool_name": "store_memory",
                    "args": {"collection": "default", "text": "implicit"},
                },
                "recall:notes:find this": {
                    "ok": True,
                    "tool_name": "search_memory",
                    "args": {"collection": "notes", "query": "find this", "top_k": 5},
                },
                "recall::find me": {
                    "ok": True,
                    "tool_name": "search_memory",
                    "args": {"collection": "default", "query": "find me", "top_k": 5},
                },
                "plain task": {
                    "ok": True,
                    "tool_name": "store_memory",
                    "args": {
                        "collection": "runs",
                        "text": "plain task",
                        "metadata": {"source": "task"},
                    },
                },
            }
        )
    )
    agent = {"id": "a1"}
    context = {"run_id": "r1", "step_limit": 8}

    shell = policy.plan("shell:echo hi", agent=agent, context=context)
    assert shell[1].tool_name == "run_shell"
    assert shell[1].args["command"] == "echo hi"

    read = policy.plan("read:seed.txt", agent=agent, context=context)
    assert read[1].tool_name == "read_file"
    assert read[1].args["path"] == "seed.txt"

    write = policy.plan("write:notes.txt:hello", agent=agent, context=context)
    assert write[1].tool_name == "write_file"
    assert write[1].args == {"path": "notes.txt", "content": "hello"}

    remember = policy.plan("remember:notes:store this", agent=agent, context=context)
    assert remember[1].tool_name == "store_memory"
    assert remember[1].args["collection"] == "notes"
    assert remember[1].args["text"] == "store this"

    remember_default = policy.plan("remember::implicit", agent=agent, context=context)
    assert remember_default[1].args["collection"] == "default"

    recall = policy.plan("recall:notes:find this", agent=agent, context=context)
    assert recall[1].tool_name == "search_memory"
    assert recall[1].args["collection"] == "notes"
    assert recall[1].args["query"] == "find this"
    assert recall[1].args["top_k"] == 5

    recall_default = policy.plan("recall::find me", agent=agent, context=context)
    assert recall_default[1].args["collection"] == "default"

    default = policy.plan("plain task", agent=agent, context=context)
    assert default[1].tool_name == "store_memory"
    assert default[1].args["collection"] == "runs"
    assert default[1].args["metadata"] == {"source": "task"}

    assert shell[0].args["agent_id"] == "a1"
    assert shell[0].args["run_id"] == "r1"

    for actions in [shell, read, write, remember, recall, default]:
        assert actions[0].step_type == "plan"
        assert actions[-1].step_type == "eval"


def test_policy_can_classify_natural_language_tasks_without_prefixes():
    policy = DeterministicPolicy(
        model_gateway=_FakeModelGateway(
            {
                "Please remember this for me": {
                    "ok": True,
                    "tool_name": "store_memory",
                    "args": {
                        "collection": "default",
                        "text": "Please remember this for me",
                    },
                },
                "find what we saved before": {
                    "ok": True,
                    "tool_name": "search_memory",
                    "args": {
                        "collection": "default",
                        "query": "find what we saved before",
                        "top_k": 5,
                    },
                },
            }
        )
    )
    agent = {"id": "a2"}
    context = {"run_id": "r2"}

    memory_plan = policy.plan(
        "Please remember this for me",
        agent=agent,
        context=context,
    )
    assert memory_plan[1].tool_name == "store_memory"
    assert memory_plan[1].args["collection"] == "default"

    search_plan = policy.plan(
        "find what we saved before",
        agent=agent,
        context=context,
    )
    assert search_plan[1].tool_name == "search_memory"
    assert search_plan[1].args["collection"] == "default"


def test_policy_falls_back_when_model_inference_fails():
    policy = DeterministicPolicy(
        model_gateway=_FakeModelGateway(
            {
                "run diagnostics": {
                    "ok": False,
                    "error": {
                        "code": "model_error",
                        "message": "backend unavailable",
                    },
                }
            }
        )
    )

    actions = policy.plan(
        "run diagnostics",
        agent={"id": "a3"},
        context={"run_id": "r3"},
    )

    assert actions[1].tool_name == "store_memory"
    assert actions[1].args["metadata"]["model_error"] == "backend unavailable"
