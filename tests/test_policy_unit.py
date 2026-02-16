from __future__ import annotations

from app.policy import StubPolicy


def test_policy_parses_all_task_prefixes():
    policy = StubPolicy()

    shell = policy.plan("shell:echo hi")
    assert shell[1].tool_name == "run_shell"
    assert shell[1].args["command"] == "echo hi"

    read = policy.plan("read:seed.txt")
    assert read[1].tool_name == "read_file"
    assert read[1].args["path"] == "seed.txt"

    write = policy.plan("write:notes.txt:hello")
    assert write[1].tool_name == "write_file"
    assert write[1].args == {"path": "notes.txt", "content": "hello"}

    remember = policy.plan("remember:notes:store this")
    assert remember[1].tool_name == "store_memory"
    assert remember[1].args["collection"] == "notes"
    assert remember[1].args["text"] == "store this"

    remember_default = policy.plan("remember::implicit")
    assert remember_default[1].args["collection"] == "default"

    recall = policy.plan("recall:notes:find this")
    assert recall[1].tool_name == "search_memory"
    assert recall[1].args["collection"] == "notes"
    assert recall[1].args["query"] == "find this"
    assert recall[1].args["top_k"] == 5

    recall_default = policy.plan("recall::find me")
    assert recall_default[1].args["collection"] == "default"

    default = policy.plan("plain task")
    assert default[1].tool_name == "store_memory"
    assert default[1].args["collection"] == "runs"
    assert default[1].args["metadata"] == {"source": "task"}

    for actions in [shell, read, write, remember, recall, default]:
        assert actions[0].step_type == "plan"
        assert actions[-1].step_type == "eval"
