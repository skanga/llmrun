from __future__ import annotations

import base64
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

try:
    from platformdirs import user_data_dir
except ModuleNotFoundError:

    def user_data_dir(
        appname: str | None = None,
        appauthor: str | Literal[False] | None = None,
        version: str | None = None,
        roaming: bool = False,
        ensure_exists: bool = False,
        use_site_for_root: bool = False,
    ) -> str:
        base = os.environ.get("APPDATA") or str(Path.home() / ".local" / "share")
        return str(Path(base) / (appname or ""))


def default_state_dir() -> Path:
    override = os.environ.get("LLMRUN_STATE_DIR")
    if override:
        return Path(override).expanduser()
    return Path(user_data_dir("llmrun", appauthor=False))


@dataclass
class Turn:
    prompt: str
    output: str
    created_at: str
    attachments: list[dict[str, str]] = field(default_factory=list)


@dataclass
class Session:
    name: str
    provider: str
    model: str
    response_id: str | None
    created_at: str
    updated_at: str
    base_url: str | None = None
    turns: list[Turn] = field(default_factory=list)


class StateStore:
    def __init__(self, root: Path | None = None):
        self.root = root or default_state_dir()
        self.sessions_dir = self.root / "sessions"
        self.templates_dir = self.root / "templates"
        self.fragments_dir = self.root / "fragments"
        self.config_path = self.root / "config.json"
        for path in (self.sessions_dir, self.templates_dir, self.fragments_dir):
            path.mkdir(parents=True, exist_ok=True)

    def update_session(
        self,
        name: str,
        *,
        provider: str,
        model: str,
        response_id: str | None,
        base_url: str | None = None,
        prompt: str,
        output: str,
        attachments: list[Any] | None = None,
    ) -> Session:
        existing = self.get_session(name)
        now = _now()
        if existing is None:
            session = Session(
                name=name,
                provider=provider,
                model=model,
                response_id=response_id,
                base_url=base_url,
                created_at=now,
                updated_at=now,
                turns=[],
            )
        else:
            session = existing
            session.provider = provider
            session.model = model
            session.response_id = response_id
            session.base_url = base_url
            session.updated_at = now
        metadata = [
            _attachment_metadata(attachment) for attachment in attachments or []
        ]
        session.turns.append(
            Turn(prompt=prompt, output=output, created_at=now, attachments=metadata)
        )
        self._write_json(self._session_path(name), _session_to_dict(session))
        return session

    def get_session(self, name: str) -> Session | None:
        path = self._existing_session_path(name)
        if not path.exists():
            return None
        data = self._read_json(path)
        turns = [Turn(**turn) for turn in data.get("turns", [])]
        return Session(
            name=data["name"],
            provider=data["provider"],
            model=data["model"],
            response_id=data.get("response_id"),
            base_url=data.get("base_url"),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            turns=turns,
        )

    def list_sessions(self) -> list[str]:
        return _list_names(self.sessions_dir, ".json")

    def delete_session(self, name: str) -> bool:
        path = self._existing_session_path(name)
        if not path.exists():
            return False
        path.unlink()
        return True

    def export_session(self, name: str, format: str) -> str:
        session = self.get_session(name)
        if session is None:
            raise KeyError(f"Unknown session: {name}")
        if format == "json":
            return json.dumps(_session_to_dict(session), indent=2)
        if format in {"markdown", "md"}:
            parts = [f"# Session: {session.name}", ""]
            for turn in session.turns:
                parts.extend(
                    [
                        "## User",
                        "",
                        turn.prompt,
                        "",
                        "## Assistant",
                        "",
                        turn.output,
                        "",
                    ]
                )
            return "\n".join(parts).rstrip() + "\n"
        raise ValueError("format must be json or markdown")

    def set_template(self, name: str, text: str) -> None:
        self._text_path(self.templates_dir, name).write_text(text, encoding="utf-8")

    def get_template(self, name: str) -> str | None:
        path = self._existing_text_path(self.templates_dir, name)
        return path.read_text(encoding="utf-8") if path.exists() else None

    def list_templates(self) -> list[str]:
        return _list_names(self.templates_dir, ".txt")

    def delete_template(self, name: str) -> bool:
        return _delete_if_exists(self._existing_text_path(self.templates_dir, name))

    def set_fragment(self, name: str, text: str) -> None:
        self._text_path(self.fragments_dir, name).write_text(text, encoding="utf-8")

    def get_fragment(self, name: str) -> str | None:
        path = self._existing_text_path(self.fragments_dir, name)
        return path.read_text(encoding="utf-8") if path.exists() else None

    def list_fragments(self) -> list[str]:
        return _list_names(self.fragments_dir, ".txt")

    def delete_fragment(self, name: str) -> bool:
        return _delete_if_exists(self._existing_text_path(self.fragments_dir, name))

    def get_config(self, key: str) -> Any:
        return self._read_config().get(key)

    def set_config(self, key: str, value: Any) -> None:
        data = self._read_config()
        data[key] = value
        self._write_json(self.config_path, data)

    def unset_config(self, key: str) -> bool:
        data = self._read_config()
        existed = key in data
        data.pop(key, None)
        self._write_json(self.config_path, data)
        return existed

    def _session_path(self, name: str) -> Path:
        return _encoded_path(self.sessions_dir, name, ".json")

    def _existing_session_path(self, name: str) -> Path:
        return _existing_path(self.sessions_dir, name, ".json")

    def _text_path(self, directory: Path, name: str) -> Path:
        return _encoded_path(directory, name, ".txt")

    def _existing_text_path(self, directory: Path, name: str) -> Path:
        return _existing_path(directory, name, ".txt")

    def _read_config(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return {}
        return self._read_json(self.config_path)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object in {path}")
        return data

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                tmp_name = handle.name
                json.dump(data, handle, indent=2)
            os.replace(tmp_name, path)
        finally:
            if tmp_name and Path(tmp_name).exists():
                Path(tmp_name).unlink()


def _session_to_dict(session: Session) -> dict[str, Any]:
    return asdict(session)


def _attachment_metadata(attachment: Any) -> dict[str, str]:
    if hasattr(attachment, "metadata"):
        return attachment.metadata()
    data = {
        "kind": attachment.get("kind"),
        "source": attachment.get("source"),
        "filename": attachment.get("filename"),
        "mime_type": attachment.get("mime_type"),
    }
    if attachment.get("detail"):
        data["detail"] = attachment["detail"]
    return {key: value for key, value in data.items() if isinstance(value, str)}


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", name.strip())
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError("Name must contain at least one safe character.")
    return cleaned


def _encoded_name(name: str) -> str:
    encoded = base64.urlsafe_b64encode(name.encode("utf-8")).decode("ascii").rstrip("=")
    return f"~{encoded}"


def _decode_name(stem: str) -> str:
    if not stem.startswith("~"):
        return stem
    payload = stem[1:]
    payload += "=" * (-len(payload) % 4)
    try:
        return base64.urlsafe_b64decode(payload.encode("ascii")).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return stem


def _encoded_path(directory: Path, name: str, suffix: str) -> Path:
    return directory / f"{_encoded_name(name)}{suffix}"


def _existing_path(directory: Path, name: str, suffix: str) -> Path:
    encoded = directory / f"{_encoded_name(name)}{suffix}"
    if encoded.exists():
        return encoded
    legacy = directory / f"{_safe_name(name)}{suffix}"
    if legacy.exists():
        return legacy
    return encoded


def _list_names(directory: Path, suffix: str) -> list[str]:
    return sorted({_decode_name(path.stem) for path in directory.glob(f"*{suffix}")})


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _delete_if_exists(path: Path) -> bool:
    if not path.exists():
        return False
    path.unlink()
    return True
