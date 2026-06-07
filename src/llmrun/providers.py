from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import os
from pathlib import Path
import time
from typing import Any, Callable

import httpx

from .attachments import Attachment, build_attachments
from .auth import AuthInfo, OAuthTokens
from .provider_config import capabilities_for_base_url

DEFAULT_MODEL = "gpt-5.4-mini"
CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
OPENAI_REFRESH_URL = "https://auth.openai.com/oauth/token"
CODEX_DEFAULT_INSTRUCTIONS = (
    "You are a helpful assistant. Answer the user's request directly."
)


@dataclass(frozen=True, init=False)
class ProviderRequest:
    input_text: str
    model: str
    instructions: str | None = None
    previous_response_id: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    reasoning_effort: str | None = None
    json_mode: bool = False
    stream: bool = False
    attachments: list[Attachment] = field(default_factory=list)

    def __init__(
        self,
        input_text: str | None = None,
        model: str = DEFAULT_MODEL,
        *,
        input: str | None = None,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
        json_mode: bool = False,
        stream: bool = False,
        attachments: list[Attachment] | None = None,
    ):
        text = input_text if input_text is not None else input
        if text is None:
            text = ""
        object.__setattr__(self, "input_text", text)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "instructions", instructions)
        object.__setattr__(self, "previous_response_id", previous_response_id)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "max_output_tokens", max_output_tokens)
        object.__setattr__(self, "reasoning_effort", reasoning_effort)
        object.__setattr__(self, "json_mode", json_mode)
        object.__setattr__(self, "stream", stream)
        object.__setattr__(self, "attachments", attachments or [])

    @property
    def input(self) -> str:
        return self.input_text


@dataclass(frozen=True)
class ProviderResult:
    text: str
    response_id: str | None


class ProviderError(RuntimeError):
    pass


class OpenAIProvider:
    name = "openai"

    def __init__(
        self,
        auth: AuthInfo,
        client_factory: Callable[..., Any] | None = None,
        base_url: str | None = None,
        sleep_func: Callable[[float], None] | None = None,
        monotonic_func: Callable[[], float] | None = None,
    ):
        if not auth.api_key:
            raise ValueError("OpenAI provider requires an API key.")
        self.auth = auth
        self.base_url = base_url
        self._client_factory = client_factory or _openai_client
        self._client = self._client_factory(auth.api_key, base_url=base_url)
        self._sleep = sleep_func or time.sleep
        self._monotonic = monotonic_func or time.monotonic

    def complete(
        self,
        request: ProviderRequest,
        *,
        on_delta: Callable[[str], None] | None = None,
    ) -> ProviderResult:
        if capabilities_for_base_url(self.base_url).prompt_endpoint == "chat":
            return self._complete_chat_prompt(request, on_delta=on_delta)
        kwargs = _responses_kwargs(request)
        try:
            response = self._client.responses.create(**kwargs)
        except Exception as exc:
            should_try_fallback = self.base_url and _responses_unsupported_for_fallback(
                exc
            )
            if should_try_fallback and _chat_vision_fallback_supported(request):
                return self._complete_chat_prompt(request, on_delta=on_delta)
            if should_try_fallback and _chat_pdf_fallback_supported(request):
                return self._complete_chat_prompt(
                    _request_with_pdf_pages_as_images(request), on_delta=on_delta
                )
            if should_try_fallback and _chat_text_fallback_supported(request):
                return self._complete_chat_prompt(request, on_delta=on_delta)
            if should_try_fallback and request.attachments:
                raise ProviderError(
                    "OpenAI-compatible provider request failed: Responses API rejected the request, "
                    "and Chat Completions fallback only supports image attachments and local PDFs."
                ) from exc
            raise ProviderError(
                f"OpenAI provider request failed: {_safe_exception_message(exc)}"
            ) from exc
        if request.stream:
            return _consume_openai_stream(response, on_delta)
        return ProviderResult(
            text=_response_text(response), response_id=getattr(response, "id", None)
        )

    def _complete_chat_prompt(
        self,
        request: ProviderRequest,
        *,
        on_delta: Callable[[str], None] | None = None,
    ) -> ProviderResult:
        if request.attachments and not _chat_vision_fallback_supported(request):
            if _chat_pdf_fallback_supported(request):
                return self._complete_chat_prompt(
                    _request_with_pdf_pages_as_images(request), on_delta=on_delta
                )
            raise ProviderError(
                "OpenAI-compatible chat completions prompt endpoint only supports "
                "text prompts, image attachments, and local PDFs."
            )
        kwargs = _chat_completion_kwargs(request)
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise ProviderError(
                f"OpenAI-compatible chat completions fallback failed: {_safe_exception_message(exc)}"
            ) from exc
        if request.stream:
            return _consume_chat_stream(response, on_delta)
        return ProviderResult(
            text=_chat_response_text(response),
            response_id=getattr(response, "id", None),
        )

    def transcribe_audio(self, path: str, *, model: str) -> str:
        try:
            with Path(path).open("rb") as handle:
                response = self._client.audio.transcriptions.create(
                    model=model, file=handle
                )
        except Exception as exc:
            raise ProviderError(
                f"OpenAI audio transcription failed: {_safe_exception_message(exc)}"
            ) from exc
        return _audio_text(response)

    def translate_audio(self, path: str, *, model: str) -> str:
        try:
            with Path(path).open("rb") as handle:
                response = self._client.audio.translations.create(
                    model=model, file=handle
                )
        except Exception as exc:
            raise ProviderError(
                f"OpenAI audio translation failed: {_safe_exception_message(exc)}"
            ) from exc
        return _audio_text(response)

    def generate_speech(
        self,
        text: str,
        *,
        model: str,
        voice: str,
        output_path: str,
        response_format: str | None = None,
    ) -> str:
        try:
            speech = self._client.audio.speech
            kwargs: dict[str, Any] = {"model": model, "voice": voice, "input": text}
            if response_format:
                kwargs["response_format"] = response_format
            streaming = getattr(speech, "with_streaming_response", None)
            if streaming is not None:
                path = Path(output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with streaming.create(**kwargs) as response:
                    response.stream_to_file(path)
            else:
                response = speech.create(**kwargs)
                _write_binary_response(response, output_path)
        except Exception as exc:
            raise ProviderError(
                f"OpenAI speech generation failed: {_safe_exception_message(exc)}"
            ) from exc
        return output_path

    def generate_images(
        self, prompt: str, *, model: str, output_path: str, count: int
    ) -> list[str]:
        if not capabilities_for_base_url(self.base_url).supports_images_endpoint:
            raise ProviderError(
                "Groq does not support image generation through the OpenAI-compatible images endpoint. "
                "Use an OpenAI image model, unset LLMRUN_BASE_URL, or use Codex OAuth image generation."
            )
        try:
            response = self._client.images.generate(model=model, prompt=prompt, n=count)
            return _write_image_responses(response, output_path, count=count)
        except Exception as exc:
            raise ProviderError(
                f"OpenAI image generation failed: {_safe_exception_message(exc)}"
            ) from exc

    def generate_image(self, prompt: str, *, model: str, output_path: str) -> str:
        return self.generate_images(
            prompt, model=model, output_path=output_path, count=1
        )[0]

    def edit_image(
        self, image_path: str, prompt: str, *, model: str, output_path: str
    ) -> str:
        if not capabilities_for_base_url(self.base_url).supports_images_endpoint:
            raise ProviderError(
                "Groq does not support image editing through the OpenAI-compatible images endpoint. "
                "Use an OpenAI image model, unset LLMRUN_BASE_URL, or use Codex OAuth image editing."
            )
        try:
            with Path(image_path).open("rb") as handle:
                response = self._client.images.edit(
                    model=model, image=handle, prompt=prompt
                )
            _write_image_response(response, output_path)
        except Exception as exc:
            raise ProviderError(
                f"OpenAI image edit failed: {_safe_exception_message(exc)}"
            ) from exc
        return output_path

    def generate_video(
        self,
        prompt: str,
        *,
        model: str | None,
        output_path: str,
        image_path: str | None = None,
        seconds: str | None = None,
        size: str | None = None,
        poll_interval: float = 2.0,
        timeout: float = 900.0,
    ) -> str:
        if not model:
            raise ProviderError("OpenAI video generation requires a model.")
        try:
            video = self._create_video_job(
                prompt=prompt,
                model=model,
                image_path=image_path,
                seconds=seconds,
                size=size,
            )
            video = self._poll_video_job(
                video, poll_interval=poll_interval, timeout=timeout
            )
            content = self._client.videos.download_content(video.id, variant="video")
            _write_binary_response(content, output_path)
        except Exception as exc:
            raise ProviderError(
                f"OpenAI video generation failed: {_safe_exception_message(exc)}"
            ) from exc
        return output_path

    def _create_video_job(
        self,
        *,
        prompt: str,
        model: str,
        image_path: str | None,
        seconds: str | None,
        size: str | None,
    ) -> Any:
        kwargs: dict[str, Any] = {"model": model, "prompt": prompt}
        if seconds:
            kwargs["seconds"] = seconds
        if size:
            kwargs["size"] = size
        if not image_path:
            return self._client.videos.create(**kwargs)
        with Path(image_path).open("rb") as handle:
            kwargs["input_reference"] = handle
            return self._client.videos.create(**kwargs)

    def _poll_video_job(
        self, video: Any, *, poll_interval: float, timeout: float
    ) -> Any:
        started = self._monotonic()
        while getattr(video, "status", None) in {"queued", "in_progress"}:
            if self._monotonic() - started > timeout:
                raise ProviderError(
                    f"Video generation timed out after {timeout:g} seconds."
                )
            self._sleep(poll_interval)
            video = self._client.videos.retrieve(video.id)
        if getattr(video, "status", None) == "failed":
            error = getattr(video, "error", None)
            message = getattr(error, "message", None) or "Video generation failed."
            raise ProviderError(message)
        if getattr(video, "status", None) != "completed":
            raise ProviderError(
                f"Video generation ended with unexpected status: {getattr(video, 'status', None)}"
            )
        return video


class CodexOAuthProvider:
    name = "codex"

    def __init__(
        self,
        auth: AuthInfo,
        http_client: Any | None = None,
        responses_url: str | None = None,
        refresh_url: str | None = None,
    ):
        if not auth.oauth:
            raise ValueError("Codex provider requires OAuth tokens.")
        self.auth = auth
        self.tokens = auth.oauth
        self.http = http_client or httpx.Client(timeout=60)
        self.responses_url = responses_url or os.environ.get(
            "LLMRUN_CODEX_RESPONSES_URL", CODEX_RESPONSES_URL
        )
        self.refresh_url = refresh_url or os.environ.get(
            "LLMRUN_OPENAI_REFRESH_URL", OPENAI_REFRESH_URL
        )

    def complete(
        self,
        request: ProviderRequest,
        *,
        on_delta: Callable[[str], None] | None = None,
    ) -> ProviderResult:
        self._refresh_if_needed()
        payload = _responses_kwargs(request)
        payload["stream"] = True
        payload.setdefault("instructions", CODEX_DEFAULT_INSTRUCTIONS)
        payload["input"] = _codex_input_items(request)
        payload["store"] = False
        payload.pop("max_output_tokens", None)
        payload.pop("max_completion_tokens", None)
        payload.pop("previous_response_id", None)
        try:
            response = self.http.post(
                self.responses_url,
                json=payload,
                headers=self._headers(),
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                _http_status_message(
                    "Codex OAuth provider request failed", exc.response
                )
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Codex OAuth provider request failed: {_safe_exception_message(exc)}"
            ) from exc
        if _is_sse_response(response):
            return _consume_sse_text(
                response.text, on_delta if request.stream else None
            )
        try:
            data = response.json()
        except ValueError as exc:
            result = _consume_sse_text(
                getattr(response, "text", ""), on_delta if request.stream else None
            )
            if result.text or result.response_id:
                return result
            raise ProviderError(
                "Codex OAuth provider returned an empty or unparseable response body."
            ) from exc
        return ProviderResult(
            text=_dict_response_text(data), response_id=data.get("id")
        )

    def _refresh_if_needed(self) -> None:
        if not self.tokens.refresh_token or not _is_expired(self.tokens.expires_at):
            return
        try:
            response = self.http.post(
                self.refresh_url,
                json={
                    "grant_type": "refresh_token",
                    "refresh_token": self.tokens.refresh_token,
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(
                "Codex OAuth refresh failed. Re-run Codex login and try again."
            ) from exc
        data = response.json()
        access = data.get("access_token")
        if not isinstance(access, str) or not access:
            raise ProviderError(
                "Codex OAuth refresh failed. Re-run Codex login and try again."
            )
        self.tokens = OAuthTokens(
            access_token=access,
            refresh_token=data.get("refresh_token") or self.tokens.refresh_token,
            expires_at=data.get("expires_at"),
            account_id=self.tokens.account_id,
        )

    def transcribe_audio(self, path: str, *, model: str) -> str:
        raise ProviderError(
            "Codex OAuth does not support audio workflows yet; use OPENAI_API_KEY"
        )

    def translate_audio(self, path: str, *, model: str) -> str:
        raise ProviderError(
            "Codex OAuth does not support audio workflows yet; use OPENAI_API_KEY"
        )

    def generate_speech(
        self,
        text: str,
        *,
        model: str,
        voice: str,
        output_path: str,
        response_format: str | None = None,
    ) -> str:
        raise ProviderError(
            "Codex OAuth does not support audio workflows yet; use OPENAI_API_KEY"
        )

    def generate_image(self, prompt: str, *, model: str, output_path: str) -> str:
        return self.generate_images(
            prompt, model=model, output_path=output_path, count=1
        )[0]

    def generate_images(
        self, prompt: str, *, model: str, output_path: str, count: int
    ) -> list[str]:
        paths = [
            _numbered_output_path(output_path, index, count)
            for index in range(1, count + 1)
        ]
        for path in paths:
            image_b64 = self._generate_image_b64(
                prompt, model=_codex_image_response_model(model)
            )
            _write_image_item({"b64_json": image_b64}, path)
        return [str(path) for path in paths]

    def edit_image(
        self, image_path: str, prompt: str, *, model: str, output_path: str
    ) -> str:
        attachments = build_attachments(
            images=[image_path], files=[], image_detail="auto"
        )
        image_b64 = self._generate_image_b64(
            prompt, model=_codex_image_response_model(model), attachments=attachments
        )
        _write_image_item({"b64_json": image_b64}, Path(output_path))
        return output_path

    def generate_video(
        self,
        prompt: str,
        *,
        model: str | None,
        output_path: str,
        image_path: str | None = None,
        seconds: str | None = None,
        size: str | None = None,
        poll_interval: float = 2.0,
        timeout: float = 900.0,
    ) -> str:
        raise ProviderError(
            "Codex OAuth does not support video generation yet; use OPENAI_API_KEY"
        )

    def _generate_image_b64(
        self, prompt: str, *, model: str, attachments: list[Attachment] | None = None
    ) -> str:
        self._refresh_if_needed()
        request = ProviderRequest(
            input_text=prompt, model=model, attachments=attachments
        )
        payload: dict[str, Any] = {
            "model": model,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": _input_content_items(request),
                }
            ],
            "instructions": CODEX_DEFAULT_INSTRUCTIONS,
            "tools": [{"type": "image_generation"}],
            "tool_choice": {"type": "image_generation"},
            "stream": True,
            "store": False,
        }
        try:
            response = self.http.post(
                self.responses_url, json=payload, headers=self._headers()
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                _http_status_message(
                    "Codex OAuth image generation failed", exc.response
                )
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Codex OAuth image generation failed: {_safe_exception_message(exc)}"
            ) from exc
        images = _sse_image_results(getattr(response, "text", ""))
        if not images:
            raise ProviderError(
                "Codex OAuth image generation returned no image result."
            )
        return images[-1]

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.tokens.access_token}",
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        }
        if self.tokens.account_id:
            headers["ChatGPT-Account-ID"] = self.tokens.account_id
        return headers


def make_provider(
    auth: AuthInfo, base_url: str | None = None
) -> OpenAIProvider | CodexOAuthProvider:
    if auth.provider == "openai":
        return OpenAIProvider(auth, base_url=base_url)
    if auth.provider == "codex":
        return CodexOAuthProvider(auth)
    raise ValueError(f"Unsupported provider: {auth.provider}")


def _openai_client(api_key: str, base_url: str | None = None) -> Any:
    from openai import OpenAI

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _responses_kwargs(request: ProviderRequest) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": request.model,
        "input": _responses_input(request),
        "stream": request.stream,
    }
    if request.instructions:
        kwargs["instructions"] = request.instructions
    if request.previous_response_id:
        kwargs["previous_response_id"] = request.previous_response_id
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.max_output_tokens is not None:
        kwargs["max_output_tokens"] = request.max_output_tokens
    if request.reasoning_effort:
        kwargs["reasoning"] = {"effort": request.reasoning_effort}
    if request.json_mode:
        kwargs["text"] = {"format": {"type": "json_object"}}
    return kwargs


def _responses_input(request: ProviderRequest) -> str | list[dict[str, Any]]:
    if not request.attachments:
        return request.input_text
    return [{"role": "user", "content": _input_content_items(request)}]


def _chat_vision_fallback_supported(request: ProviderRequest) -> bool:
    return bool(request.attachments) and all(
        attachment.kind == "image" for attachment in request.attachments
    )


def _chat_pdf_fallback_supported(request: ProviderRequest) -> bool:
    return bool(request.attachments) and all(
        attachment.kind == "image" or _local_pdf_attachment(attachment)
        for attachment in request.attachments
    )


def _chat_text_fallback_supported(request: ProviderRequest) -> bool:
    return not request.attachments


def _responses_unsupported_for_fallback(exc: Exception) -> bool:
    status_code = _exception_status_code(exc)
    if status_code in {404, 405}:
        return True
    if status_code in {400, 422}:
        return _message_indicates_responses_unsupported(exc)
    return _message_indicates_responses_unsupported(exc)


def _message_indicates_responses_unsupported(exc: Exception) -> bool:
    message = _safe_exception_message(exc).lower()
    unsupported_markers = (
        "responses api",
        "responses endpoint",
        "response api",
        "unsupported response",
        "unsupported input",
        "unsupported payload",
        "unsupported content",
        "missing responses",
        "not support responses",
        "does not support responses",
        "responses is not supported",
        "unsupported input format",
        "unsupported input-format",
        "input format is not supported",
        "input-format is not supported",
        "invalid input format",
        "invalid input-format",
    )
    failure_markers = (
        "unsupported",
        "not supported",
        "missing",
        "not found",
        "unknown endpoint",
        "invalid input",
        "invalid payload",
    )
    return any(marker in message for marker in unsupported_markers) or (
        "responses" in message and any(marker in message for marker in failure_markers)
    )


def _exception_status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    return status_code if isinstance(status_code, int) else None


def _request_with_pdf_pages_as_images(request: ProviderRequest) -> ProviderRequest:
    attachments: list[Attachment] = []
    for attachment in request.attachments:
        if attachment.kind == "image":
            attachments.append(attachment)
            continue
        attachments.extend(_pdf_page_attachments(attachment))
    return ProviderRequest(
        input_text=request.input_text,
        model=request.model,
        instructions=request.instructions,
        previous_response_id=request.previous_response_id,
        temperature=request.temperature,
        max_output_tokens=request.max_output_tokens,
        reasoning_effort=request.reasoning_effort,
        json_mode=request.json_mode,
        stream=request.stream,
        attachments=attachments,
    )


def _local_pdf_attachment(attachment: Attachment) -> bool:
    return (
        attachment.kind == "file"
        and attachment.mime_type == "application/pdf"
        and Path(attachment.source).is_file()
    )


def _pdf_page_attachments(attachment: Attachment) -> list[Attachment]:
    try:
        import fitz  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ProviderError(
            "PDF vision fallback requires PyMuPDF. Install it with the llmrun[pdf] extra."
        ) from exc

    pages: list[Attachment] = []
    try:
        with fitz.open(attachment.source) as document:
            matrix = fitz.Matrix(2, 2)
            for index in range(document.page_count):
                pixmap = document[index].get_pixmap(matrix=matrix)
                payload = f"data:image/png;base64,{base64.b64encode(pixmap.tobytes('png')).decode('ascii')}"
                pages.append(
                    Attachment(
                        kind="image",
                        source=f"{attachment.source}#page={index + 1}",
                        filename=f"{Path(attachment.filename).stem}-page-{index + 1}.png",
                        mime_type="image/png",
                        payload=payload,
                        detail="high",
                    )
                )
    except Exception as exc:
        if isinstance(exc, ProviderError):
            raise
        raise ProviderError(
            f"PDF vision fallback failed: {_safe_exception_message(exc)}"
        ) from exc
    return pages


def _chat_completion_kwargs(request: ProviderRequest) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if request.instructions:
        messages.append({"role": "system", "content": request.instructions})
    user_content: str | list[dict[str, Any]] = request.input_text
    if request.attachments:
        user_content = _chat_content_items(request)
    messages.append({"role": "user", "content": user_content})
    kwargs: dict[str, Any] = {
        "model": request.model,
        "messages": messages,
        "stream": request.stream,
    }
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.max_output_tokens is not None:
        kwargs["max_tokens"] = request.max_output_tokens
    if request.reasoning_effort:
        kwargs["reasoning_effort"] = request.reasoning_effort
    if request.json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    return kwargs


def _chat_content_items(request: ProviderRequest) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": request.input_text}]
    for attachment in request.attachments:
        image_url: dict[str, Any] = {"url": attachment.payload or attachment.source}
        if attachment.detail:
            image_url["detail"] = attachment.detail
        content.append({"type": "image_url", "image_url": image_url})
    return content


def _input_content_items(request: ProviderRequest) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": request.input_text}]
    for attachment in request.attachments:
        if attachment.kind == "image":
            item: dict[str, Any] = {
                "type": "input_image",
                "image_url": attachment.payload or attachment.source,
            }
            if attachment.detail:
                item["detail"] = attachment.detail
            content.append(item)
        elif attachment.kind == "file":
            item = {
                "type": "input_file",
                "filename": attachment.filename,
            }
            if attachment.payload:
                item["file_data"] = attachment.payload
            else:
                item["file_url"] = attachment.source
            content.append(item)
    return content


def _codex_input_items(request: ProviderRequest) -> list[dict[str, Any]]:
    return [
        {
            "type": "message",
            "role": "user",
            "content": _input_content_items(request),
        }
    ]


def _consume_openai_stream(
    events: Any,
    on_delta: Callable[[str], None] | None,
) -> ProviderResult:
    parts: list[str] = []
    response_id: str | None = None
    for event in events:
        event_type = getattr(event, "type", None)
        if event_type == "response.output_text.delta":
            delta = getattr(event, "delta", "")
            if delta:
                parts.append(delta)
                if on_delta:
                    on_delta(delta)
        elif event_type == "response.completed":
            response = getattr(event, "response", None)
            response_id = getattr(response, "id", None)
    return ProviderResult(text="".join(parts), response_id=response_id)


def _consume_chat_stream(
    events: Any,
    on_delta: Callable[[str], None] | None,
) -> ProviderResult:
    parts: list[str] = []
    response_id: str | None = None
    for event in events:
        event_id = _field(event, "id")
        if response_id is None and isinstance(event_id, str):
            response_id = event_id
        choices = _field(event, "choices")
        if not choices:
            continue
        choice = choices[0]
        delta = _field(choice, "delta")
        content = _field(delta, "content")
        if not isinstance(content, str):
            message = _field(choice, "message")
            content = _field(message, "content")
        if isinstance(content, str) and content:
            parts.append(content)
            if on_delta:
                on_delta(content)
    text = "".join(parts)
    if not text:
        raise ProviderError(
            "OpenAI-compatible chat completions stream returned no text."
        )
    return ProviderResult(text=text, response_id=response_id)


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _consume_sse_text(
    text: str,
    on_delta: Callable[[str], None] | None,
) -> ProviderResult:
    parts: list[str] = []
    response_id: str | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data_text = line[5:].strip()
        if not data_text or data_text == "[DONE]":
            continue
        try:
            event = json.loads(data_text)
        except json.JSONDecodeError:
            continue
        event_type = event.get("type")
        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str) and delta:
                parts.append(delta)
                if on_delta:
                    on_delta(delta)
        elif event_type == "response.completed":
            response = event.get("response")
            if isinstance(response, dict) and isinstance(response.get("id"), str):
                response_id = response["id"]
        elif event_type == "response.output_item.done":
            item = event.get("item")
            if isinstance(item, dict):
                response_id = response_id or item.get("id")
        elif event_type == "response.content_part.done":
            part = event.get("part")
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text_part = part["text"]
                if not parts:
                    parts.append(text_part)
                    if on_delta:
                        on_delta(text_part)
    return ProviderResult(text="".join(parts), response_id=response_id)


def _sse_image_results(text: str) -> list[str]:
    images: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data_text = line[5:].strip()
        if not data_text or data_text == "[DONE]":
            continue
        try:
            event = json.loads(data_text)
        except json.JSONDecodeError:
            continue
        for candidate in (event.get("item"), event):
            if (
                not isinstance(candidate, dict)
                or candidate.get("type") != "image_generation_call"
            ):
                continue
            result = candidate.get("result")
            if isinstance(result, str) and result:
                images.append(result)
    return images


def _is_sse_response(response: Any) -> bool:
    content_type = ""
    headers = getattr(response, "headers", None)
    if headers is not None:
        content_type = headers.get("content-type", "")
    response_text = getattr(response, "text", "")
    return (
        "text/event-stream" in content_type
        or response_text.lstrip().startswith("data:")
        or any(line.lstrip().startswith("data:") for line in response_text.splitlines())
    )


def _response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text
    if isinstance(response, dict):
        return _dict_response_text(response)
    return str(response)


def _chat_response_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                message = choice.get("message")
                if isinstance(message, dict) and isinstance(
                    message.get("content"), str
                ):
                    return message["content"]
    return str(response)


def _audio_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text
    if isinstance(response, dict) and isinstance(response.get("text"), str):
        return response["text"]
    return str(response)


def _dict_response_text(data: dict[str, Any]) -> str:
    output_text = data.get("output_text")
    if isinstance(output_text, str):
        return output_text
    texts: list[str] = []
    for item in data.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                texts.append(content["text"])
    return "".join(texts)


def _write_binary_response(response: Any, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(response, "write_to_file"):
        response.write_to_file(path)
        return
    content = getattr(response, "content", None)
    if isinstance(content, bytes):
        path.write_bytes(content)
        return
    if hasattr(response, "read"):
        content = response.read()
        if isinstance(content, bytes):
            path.write_bytes(content)
            return
    if isinstance(response, bytes):
        path.write_bytes(response)
        return
    raise ProviderError("Provider returned no writable binary content.")


def _write_image_response(response: Any, output_path: str) -> None:
    _write_image_responses(response, output_path, count=1)


def _write_image_responses(response: Any, output_path: str, *, count: int) -> list[str]:
    items = _response_data_items(response)
    if not items:
        items = [response]
    paths = [
        _numbered_output_path(output_path, index, len(items))
        for index in range(1, len(items) + 1)
    ]
    for item, path in zip(items, paths, strict=False):
        _write_image_item(item, path)
    return [str(path) for path in paths]


def _write_image_item(data: Any, output_path: Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    b64_json = getattr(data, "b64_json", None)
    if isinstance(data, dict):
        b64_json = data.get("b64_json")
    if isinstance(b64_json, str) and b64_json:
        path.write_bytes(base64.b64decode(b64_json))
        return
    url = getattr(data, "url", None)
    if isinstance(data, dict):
        url = data.get("url")
    if isinstance(url, str) and url:
        download = httpx.get(url, timeout=60)
        download.raise_for_status()
        path.write_bytes(download.content)
        return
    _write_binary_response(data, str(path))


def _response_data_items(response: Any) -> list[Any]:
    data = getattr(response, "data", None)
    if isinstance(response, dict):
        data = response.get("data")
    if isinstance(data, list):
        return data
    return []


def _numbered_output_path(output_path: str, index: int, total: int) -> Path:
    path = Path(output_path)
    if total == 1:
        return path
    return path.with_name(f"{path.stem}-{index}{path.suffix}")


def _codex_image_response_model(model: str) -> str:
    if model.startswith("gpt-image-") or model.startswith("dall-e-"):
        return DEFAULT_MODEL
    return model


def _is_expired(expires_at: str | None) -> bool:
    if not expires_at:
        return False
    try:
        normalized = expires_at.replace("Z", "+00:00")
        expiry = datetime.fromisoformat(normalized)
    except ValueError:
        return False
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=UTC)
    return expiry <= datetime.now(UTC) + timedelta(minutes=2)


def _http_status_message(prefix: str, response: httpx.Response) -> str:
    detail = _response_error_detail(response)
    message = f"{prefix}: HTTP {response.status_code}"
    if detail:
        message = f"{message}: {detail}"
    if response.status_code == 400:
        message = (
            f"{message}. The Codex OAuth backend may reject this model or payload; "
            "try --model with a model available to your Codex account, or set OPENAI_API_KEY to use the official OpenAI API."
        )
    return message


def _response_error_detail(response: httpx.Response) -> str | None:
    try:
        data = response.json()
    except ValueError:
        text = response.text.strip()
        return text[:300] if text else None
    if isinstance(data, dict):
        for key in ("detail", "error", "message"):
            value = data.get(key)
            if isinstance(value, str) and value:
                return value[:300]
            if isinstance(value, dict):
                nested = value.get("message")
                if isinstance(nested, str) and nested:
                    return nested[:300]
    return None


def _safe_exception_message(exc: Exception) -> str:
    message = str(exc).replace("\r", " ").replace("\n", " ")
    return message[:300] or exc.__class__.__name__
