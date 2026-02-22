from __future__ import annotations

import os


def test_settings_page_renders(client):
    response = client.get("/settings")
    assert response.status_code == 200
    assert "OpenAI API Key" in response.text
    assert "DeepSeek API Key" in response.text


def test_save_and_clear_openai_key_from_settings(client):
    save = client.post(
        "/settings/openai",
        data={"action": "save", "openai_api_key": "sk-test-1234"},
        follow_redirects=False,
    )
    assert save.status_code == 200
    assert "OpenAI API key saved" in save.text

    status = client.get("/api/settings/openai-key")
    assert status.status_code == 200
    payload = status.json()
    assert payload["configured"] is True

    assert (
        client.app.state.services.repo.get_setting("openai_api_key") == "sk-test-1234"
    )
    assert os.getenv("OPENAI_API_KEY") == "sk-test-1234"

    clear = client.post(
        "/settings/openai",
        data={"action": "clear", "openai_api_key": ""},
        follow_redirects=False,
    )
    assert clear.status_code == 200
    assert "OpenAI API key removed" in clear.text

    status_after = client.get("/api/settings/openai-key").json()
    assert status_after["configured"] is False
    assert client.app.state.services.repo.get_setting("openai_api_key") is None


def test_save_and_clear_deepseek_key_from_settings(client):
    save = client.post(
        "/settings/deepseek",
        data={"action": "save", "deepseek_api_key": "dsk-test-1234"},
        follow_redirects=False,
    )
    assert save.status_code == 200
    assert "DeepSeek API key saved" in save.text

    status = client.get("/api/settings/deepseek-key")
    assert status.status_code == 200
    payload = status.json()
    assert payload["configured"] is True

    assert (
        client.app.state.services.repo.get_setting("deepseek_api_key")
        == "dsk-test-1234"
    )
    assert os.getenv("DEEPSEEK_API_KEY") == "dsk-test-1234"

    clear = client.post(
        "/settings/deepseek",
        data={"action": "clear", "deepseek_api_key": ""},
        follow_redirects=False,
    )
    assert clear.status_code == 200
    assert "DeepSeek API key removed" in clear.text

    status_after = client.get("/api/settings/deepseek-key").json()
    assert status_after["configured"] is False
    assert client.app.state.services.repo.get_setting("deepseek_api_key") is None
