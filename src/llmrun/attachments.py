from __future__ import annotations

import base64
from dataclasses import dataclass
import mimetypes
from pathlib import Path
from urllib.parse import urlparse, unquote


@dataclass(frozen=True)
class Attachment:
    kind: str
    source: str
    filename: str
    mime_type: str
    payload: str | None = None
    detail: str | None = None

    def metadata(self) -> dict[str, str]:
        data = {
            "kind": self.kind,
            "source": self.source,
            "filename": self.filename,
            "mime_type": self.mime_type,
        }
        if self.detail:
            data["detail"] = self.detail
        return data


def build_attachments(
    *,
    images: list[str] | tuple[str, ...],
    files: list[str] | tuple[str, ...],
    image_detail: str,
) -> list[Attachment]:
    attachments: list[Attachment] = []
    attachments.extend(
        _build_attachment("image", source, image_detail=image_detail)
        for source in images
    )
    attachments.extend(
        _build_attachment("file", source, image_detail=None) for source in files
    )
    return attachments


def _build_attachment(kind: str, source: str, image_detail: str | None) -> Attachment:
    if _is_https_url(source):
        filename = _url_filename(source) or ("image" if kind == "image" else "file")
        mime_type = _guess_mime_type(filename, kind)
        return Attachment(
            kind=kind,
            source=source,
            filename=filename,
            mime_type=mime_type,
            detail=image_detail if kind == "image" else None,
        )
    if _is_url(source):
        raise ValueError("Attachment URLs must use https.")

    path = Path(source).expanduser()
    if not path.exists():
        raise ValueError(f"Attachment does not exist: {source}")
    if not path.is_file():
        raise ValueError(f"Attachment is not a file: {source}")

    mime_type = _guess_mime_type(path.name, kind)
    data = path.read_bytes()
    payload = f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"
    return Attachment(
        kind=kind,
        source=source,
        filename=path.name,
        mime_type=mime_type,
        payload=payload,
        detail=image_detail if kind == "image" else None,
    )


def _guess_mime_type(filename: str, kind: str) -> str:
    guessed, _ = mimetypes.guess_type(filename)
    if guessed:
        return guessed
    return "image/png" if kind == "image" else "application/octet-stream"


def _is_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in {"http", "https", "file"} and bool(
        parsed.netloc or parsed.scheme == "file"
    )


def _is_https_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme == "https" and bool(parsed.netloc)


def _url_filename(source: str) -> str | None:
    path = urlparse(source).path.rstrip("/")
    if not path:
        return None
    return unquote(path.rsplit("/", 1)[-1]) or None
