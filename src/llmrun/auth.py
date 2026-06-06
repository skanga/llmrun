from __future__ import annotations

import json
import os
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class AuthError(RuntimeError):
    pass


@dataclass(frozen=True)
class OAuthTokens:
    access_token: str
    refresh_token: str | None = None
    expires_at: str | None = None
    account_id: str | None = None


@dataclass(frozen=True)
class AuthInfo:
    provider: str
    api_key: str | None = None
    oauth: OAuthTokens | None = None
    source: str | None = None
    auth_path: Path | None = None


def redact_secret(value: str | None) -> str:
    if not value or len(value) < 12:
        return "<redacted>"
    return f"{value[:4]}...{value[-4:]}"


def default_auth_path() -> Path:
    explicit = os.environ.get("CODEX_AUTH_JSON_PATH")
    if explicit:
        return Path(explicit).expanduser()
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        return Path(codex_home).expanduser() / "auth.json"
    return Path.home() / ".codex" / "auth.json"


def load_auth(auth_path: Path | None = None, provider: str | None = None) -> AuthInfo:
    provider_override = provider or os.environ.get("LLMRUN_PROVIDER")
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key and provider_override != "codex":
        provider = provider_override or "openai"
        if provider != "openai":
            raise AuthError("OPENAI_API_KEY can only be used with the openai provider.")
        return AuthInfo(provider="openai", api_key=env_key, source="OPENAI_API_KEY")

    path = auth_path or default_auth_path()
    data = _read_auth_json(path)

    file_key = _find_openai_api_key(data)
    if file_key and provider_override != "codex":
        provider = provider_override or "openai"
        if provider != "openai":
            raise AuthError(
                "Auth file OPENAI_API_KEY can only be used with the openai provider."
            )
        return AuthInfo(
            provider="openai", api_key=file_key, source=str(path), auth_path=path
        )

    oauth = _find_oauth_tokens(data)
    if oauth and provider_override != "openai":
        provider = provider_override or "codex"
        if provider != "codex":
            raise AuthError(
                "Codex OAuth tokens can only be used with the codex provider."
            )
        return AuthInfo(provider="codex", oauth=oauth, source=str(path), auth_path=path)

    raise AuthError(
        "No usable auth found. Set OPENAI_API_KEY or log in with Codex so auth.json contains OAuth tokens."
    )


def _read_auth_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise AuthError(
            f"No auth found at {path}. Set OPENAI_API_KEY or configure CODEX_AUTH_JSON_PATH."
        )
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as exc:
        raise AuthError(f"Malformed auth.json at {path}: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise AuthError(f"Malformed auth.json at {path}: expected an object.")
    return value


def _find_openai_api_key(data: dict[str, Any]) -> str | None:
    for key in ("OPENAI_API_KEY", "openai_api_key", "api_key"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _find_oauth_tokens(data: dict[str, Any]) -> OAuthTokens | None:
    candidates: list[Any] = [
        data.get("tokens"),
        data.get("oauth"),
        data.get("codex_oauth"),
        data,
    ]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        access = candidate.get("access_token") or candidate.get("id_token")
        if isinstance(access, str) and access:
            refresh = candidate.get("refresh_token")
            expires_at = candidate.get("expires_at") or candidate.get("expiry")
            return OAuthTokens(
                access_token=access,
                refresh_token=refresh if isinstance(refresh, str) else None,
                expires_at=expires_at if isinstance(expires_at, str) else None,
                account_id=_find_account_id(candidate, data, access),
            )
    return None


def _find_account_id(
    candidate: dict[str, Any], root: dict[str, Any], access_token: str
) -> str | None:
    for source in (candidate, root):
        value = source.get("account_id")
        if isinstance(value, str) and value:
            return value
    return _account_id_from_jwt(access_token)


def _account_id_from_jwt(access_token: str) -> str | None:
    parts = access_token.split(".")
    if len(parts) < 2:
        return None
    payload = parts[1]
    payload += "=" * (-len(payload) % 4)
    try:
        data = json.loads(
            base64.urlsafe_b64decode(payload.encode("ascii")).decode("utf-8")
        )
    except (ValueError, UnicodeDecodeError):
        return None
    for key in ("https://api.openai.com/profile/account_id", "account_id"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    return None
