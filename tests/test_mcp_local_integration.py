from __future__ import annotations

from pathlib import Path
import sys


def _write_fake_mcp_server(script_path: Path) -> None:
    script_path.write_text(
        """
import json
import sys


def send(payload):
    sys.stdout.write(json.dumps(payload))
    sys.stdout.write('\\n')
    sys.stdout.flush()


def recv():
    line = sys.stdin.readline()
    if not line:
        return None
    return json.loads(line)


while True:
    msg = recv()
    if msg is None:
        break
    method = msg.get('method')
    msg_id = msg.get('id')

    if method == 'initialize' and msg_id is not None:
        send(
            {
                'jsonrpc': '2.0',
                'id': msg_id,
                'result': {
                    'protocolVersion': '2024-11-05',
                    'capabilities': {'tools': {}},
                    'serverInfo': {'name': 'fake-mcp', 'version': '0.1.0'},
                },
            }
        )
        continue

    if method == 'tools/list' and msg_id is not None:
        send(
            {
                'jsonrpc': '2.0',
                'id': msg_id,
                'result': {
                    'tools': [
                        {
                            'name': 'echo',
                            'description': 'Echo tool',
                            'inputSchema': {
                                'type': 'object',
                                'properties': {
                                    'text': {'type': 'string'}
                                },
                                'required': ['text'],
                                'additionalProperties': False,
                            },
                        }
                    ]
                },
            }
        )
        continue

    if method == 'tools/call' and msg_id is not None:
        params = msg.get('params') or {}
        arguments = params.get('arguments') or {}
        text = str(arguments.get('text') or '')
        send(
            {
                'jsonrpc': '2.0',
                'id': msg_id,
                'result': {
                    'isError': False,
                    'structuredContent': {'echo': text},
                    'content': [{'type': 'text', 'text': text}],
                },
            }
        )
        continue
""".strip(),
        encoding="utf-8",
    )


def test_local_mcp_server_can_be_registered_and_called(client, tmp_path: Path):
    script = tmp_path / "fake_mcp_server.py"
    _write_fake_mcp_server(script)

    save = client.post(
        "/api/settings/mcp/servers",
        json={
            "id": "localtest",
            "command": sys.executable,
            "args": [str(script)],
            "env": {},
            "enabled": True,
        },
    )
    assert save.status_code == 200
    saved_payload = save.json()
    assert saved_payload["ok"] is True
    assert any(server["id"] == "localtest" for server in saved_payload["servers"])

    tools_page = client.get("/agents")
    assert tools_page.status_code == 200
    assert "mcp.localtest.echo" in tools_page.text

    agent = client.post(
        "/api/agents",
        json={
            "name": "mcp-agent",
            "role": "operator",
            "model": "stub-v1",
            "tools_allowed": ["mcp.localtest.echo"],
        },
    ).json()
    run = client.post(
        "/api/runs",
        json={"agent_id": agent["id"], "task": "echo", "step_limit": 2},
    ).json()

    call = client.post(
        f"/api/runs/{run['id']}/tools/call",
        json={"tool_name": "mcp.localtest.echo", "args": {"text": "hello mcp"}},
    )
    assert call.status_code == 200
    payload = call.json()
    assert payload["ok"] is True
    assert payload["mcp"]["server_id"] == "localtest"
    result = payload["mcp"]["result"]
    structured = result.get("structuredContent") or {}
    if "echo" in structured:
        assert structured["echo"] == "hello mcp"
    else:
        content = result.get("content") or []
        assert any(
            (isinstance(part, dict) and part.get("text") == "hello mcp")
            or (isinstance(part, str) and "hello mcp" in part)
            for part in content
        )
