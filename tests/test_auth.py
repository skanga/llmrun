import json
import base64

import pytest

from llmrun.auth import AuthError, default_auth_path, load_auth, redact_secret


def test_env_openai_api_key_wins(tmp_path, monkeypatch):
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(json.dumps({"OPENAI_API_KEY": "sk-file"}), encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")

    auth = load_auth(auth_path)

    assert auth.provider == "openai"
    assert auth.api_key == "sk-env"


def test_old_provider_env_var_is_ignored_after_rename(tmp_path, monkeypatch):
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(json.dumps({"OPENAI_API_KEY": "sk-file"}), encoding="utf-8")
    monkeypatch.setenv("ANYLLM_PROVIDER", "codex")

    auth = load_auth(auth_path)

    assert auth.provider == "openai"
    assert auth.api_key == "sk-file"


def test_auth_json_openai_api_key_used_when_env_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(json.dumps({"OPENAI_API_KEY": "sk-file"}), encoding="utf-8")

    auth = load_auth(auth_path)

    assert auth.provider == "openai"
    assert auth.api_key == "sk-file"


def test_default_auth_path_uses_codex_home(monkeypatch, tmp_path):
    monkeypatch.delenv("CODEX_AUTH_JSON_PATH", raising=False)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))

    assert default_auth_path() == tmp_path / "codex-home" / "auth.json"


def test_openai_api_key_rejects_non_openai_provider(tmp_path, monkeypatch):
    auth_path = tmp_path / "auth.json"
    auth_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")

    with pytest.raises(AuthError, match="OPENAI_API_KEY can only be used"):
        load_auth(auth_path, provider="groq")


def test_auth_json_openai_api_key_rejects_non_openai_provider(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(json.dumps({"OPENAI_API_KEY": "sk-file"}), encoding="utf-8")

    with pytest.raises(AuthError, match="Auth file OPENAI_API_KEY"):
        load_auth(auth_path, provider="groq")


def test_codex_oauth_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "access",
                    "refresh_token": "refresh",
                    "expires_at": "2099-01-01T00:00:00Z",
                }
            }
        ),
        encoding="utf-8",
    )

    auth = load_auth(auth_path)

    assert auth.provider == "codex"
    assert auth.oauth.access_token == "access"


def test_codex_oauth_fallback_preserves_account_id(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "account_id": "acct_123",
                "tokens": {
                    "access_token": "access",
                    "refresh_token": "refresh",
                },
            }
        ),
        encoding="utf-8",
    )

    auth = load_auth(auth_path)

    assert auth.oauth.account_id == "acct_123"


def test_codex_oauth_rejects_non_codex_provider(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps({"tokens": {"access_token": "access"}}), encoding="utf-8"
    )

    with pytest.raises(AuthError, match="Codex OAuth tokens can only"):
        load_auth(auth_path, provider="groq")


def test_oauth_account_id_can_be_read_from_jwt_payload(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    payload = {"https://api.openai.com/profile/account_id": "acct_jwt"}
    encoded = (
        base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8"))
        .decode("ascii")
        .rstrip("=")
    )
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps({"tokens": {"access_token": f"header.{encoded}.sig"}}),
        encoding="utf-8",
    )

    auth = load_auth(auth_path)

    assert auth.oauth.account_id == "acct_jwt"


def test_missing_auth_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(AuthError, match="OPENAI_API_KEY"):
        load_auth(tmp_path / "missing.json")


def test_malformed_auth_json_raises_without_secret(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    auth_path = tmp_path / "auth.json"
    auth_path.write_text('{"OPENAI_API_KEY": "sk-secret",', encoding="utf-8")

    with pytest.raises(AuthError) as exc:
        load_auth(auth_path)

    assert "sk-secret" not in str(exc.value)


def test_non_object_auth_json_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    auth_path = tmp_path / "auth.json"
    auth_path.write_text("[]", encoding="utf-8")

    with pytest.raises(AuthError, match="expected an object"):
        load_auth(auth_path)


def test_redact_secret_keeps_shape_not_value():
    assert redact_secret("sk-1234567890abcdef") == "sk-1...cdef"
    assert redact_secret("tiny") == "<redacted>"
