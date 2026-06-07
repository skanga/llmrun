from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import os
from pathlib import Path
import sys
from typing import Any, Callable

from .attachments import build_attachments
from .auth import AuthError, AuthInfo, load_auth
from .prompts import build_prompt
from .provider_config import (
    capabilities_for_base_url,
    default_model_for_base_url,
    normalize_base_url,
    provider_preset,
)
from .providers import DEFAULT_MODEL, ProviderError, ProviderRequest, make_provider
from .state import StateStore

ProviderFactory = Callable[[AuthInfo, str | None], Any]
AuthLoader = Callable[[str | None], AuthInfo]


@dataclass(frozen=True)
class PromptOptions:
    prompt: str
    system: str | None = None
    model: str | None = None
    base_url: str | None = None
    provider_choice: str | None = None
    session: str | None = None
    fragment_names: list[str] = field(default_factory=list)
    image_sources: list[str] = field(default_factory=list)
    file_sources: list[str] = field(default_factory=list)
    image_detail: str = "auto"
    temperature: float | None = None
    max_output_tokens: int | None = None
    reasoning_effort: str | None = None
    stream: bool = True
    json_mode: bool = False


@dataclass(frozen=True)
class PromptRunResult:
    text: str
    response_id: str | None
    emitted_stream: bool


class PromptRunner:
    def __init__(
        self,
        *,
        store: StateStore | None = None,
        provider_factory: ProviderFactory | None = None,
        auth_loader: AuthLoader | None = None,
        stdin_reader: Callable[[], str] | None = None,
    ):
        self.store = store or StateStore()
        self.provider_factory = provider_factory or make_provider
        self.auth_loader = auth_loader or load_auth_for_provider
        self.stdin_reader = stdin_reader or sys.stdin.read

    def run(
        self, options: PromptOptions, *, on_delta: Callable[[str], None] | None = None
    ) -> PromptRunResult:
        auth = self.auth_loader(options.provider_choice)
        chosen_base_url = configured_base_url(
            store=self.store,
            provider_choice=options.provider_choice,
            explicit_base_url=options.base_url,
        )
        provider = self.provider_factory(auth, chosen_base_url)
        fragments_text = [
            require_fragment(self.store, name) for name in options.fragment_names
        ]
        stdin_text = self.stdin_reader() if options.prompt == "-" else None
        assembled = build_prompt(
            prompt=options.prompt,
            stdin_text=stdin_text,
            system=options.system,
            fragments=fragments_text,
        )
        attachments = build_attachments(
            images=options.image_sources,
            files=options.file_sources,
            image_detail=options.image_detail,
        )
        existing = self.store.get_session(options.session) if options.session else None
        chosen_model = prompt_model_default(
            options.model,
            existing_model=existing.model if existing else None,
            store=self.store,
            base_url=chosen_base_url,
            has_vision_input=has_vision_input(attachments),
        )
        provider_name = getattr(provider, "name", auth.provider)
        request_input = assembled.input
        previous_response_id = existing.response_id if existing else None
        if existing and should_replay_session(
            existing, provider_name=provider_name, base_url=chosen_base_url
        ):
            request_input = session_replay_input(existing.turns, assembled.input)
            previous_response_id = None
        request = ProviderRequest(
            input_text=request_input,
            instructions=assembled.instructions,
            model=chosen_model,
            previous_response_id=previous_response_id,
            temperature=options.temperature,
            max_output_tokens=options.max_output_tokens,
            reasoning_effort=options.reasoning_effort,
            json_mode=options.json_mode,
            stream=options.stream,
            attachments=attachments,
        )

        emitted_stream = False

        def write_delta(delta: str) -> None:
            nonlocal emitted_stream
            emitted_stream = True
            if on_delta:
                on_delta(delta)

        try:
            try:
                result = provider.complete(
                    request, on_delta=write_delta if options.stream else None
                )
            except NotImplementedError:
                request = ProviderRequest(
                    input_text=request.input_text,
                    instructions=request.instructions,
                    model=request.model,
                    previous_response_id=request.previous_response_id,
                    temperature=request.temperature,
                    max_output_tokens=request.max_output_tokens,
                    reasoning_effort=request.reasoning_effort,
                    json_mode=request.json_mode,
                    stream=False,
                    attachments=request.attachments,
                )
                result = provider.complete(request)
        except ProviderError:
            raise

        if options.session:
            self.store.update_session(
                options.session,
                provider=provider_name,
                model=chosen_model,
                response_id=result.response_id,
                base_url=chosen_base_url,
                prompt=assembled.input,
                output=result.text,
                attachments=attachments,
            )
        return PromptRunResult(
            text=result.text,
            response_id=result.response_id,
            emitted_stream=emitted_stream,
        )


@dataclass(frozen=True)
class AudioSpeechOptions:
    text: str
    voice: str | None = None
    model: str | None = None
    output: str | None = None
    provider_choice: str | None = None
    base_url: str | None = None


def run_audio_transcribe(
    path: str,
    *,
    model: str | None,
    provider_choice: str | None,
    base_url: str | None,
    store: StateStore | None = None,
    provider_factory: ProviderFactory | None = None,
    auth_loader: AuthLoader | None = None,
) -> str:
    store = store or StateStore()
    chosen_model = audio_model_default(
        model,
        store=store,
        provider_choice=provider_choice,
        explicit_base_url=base_url,
        openai_default="gpt-4o-transcribe",
        groq_default="whisper-large-v3-turbo",
    )
    provider = provider_from_auth(
        store=store,
        provider_choice=provider_choice,
        explicit_base_url=base_url,
        provider_factory=provider_factory,
        auth_loader=auth_loader,
    )
    return provider.transcribe_audio(path, model=chosen_model)


def run_audio_translate(
    path: str,
    *,
    model: str | None,
    provider_choice: str | None,
    base_url: str | None,
    store: StateStore | None = None,
    provider_factory: ProviderFactory | None = None,
    auth_loader: AuthLoader | None = None,
) -> str:
    store = store or StateStore()
    chosen_model = audio_model_default(
        model,
        store=store,
        provider_choice=provider_choice,
        explicit_base_url=base_url,
        openai_default="whisper-1",
        groq_default="whisper-large-v3",
    )
    provider = provider_from_auth(
        store=store,
        provider_choice=provider_choice,
        explicit_base_url=base_url,
        provider_factory=provider_factory,
        auth_loader=auth_loader,
    )
    return provider.translate_audio(path, model=chosen_model)


def run_audio_speech(
    options: AudioSpeechOptions,
    *,
    store: StateStore | None = None,
    provider_factory: ProviderFactory | None = None,
    auth_loader: AuthLoader | None = None,
) -> str:
    store = store or StateStore()
    base_url = configured_base_url(
        store=store,
        provider_choice=options.provider_choice,
        explicit_base_url=options.base_url,
    )
    capabilities = capabilities_for_base_url(base_url)
    chosen_model = audio_model_default(
        options.model,
        store=store,
        provider_choice=options.provider_choice,
        explicit_base_url=options.base_url,
        openai_default="gpt-4o-mini-tts",
        groq_default="canopylabs/orpheus-v1-english",
    )
    chosen_voice = options.voice or capabilities.speech_default_voice or "alloy"
    output_path = options.output or timestamped_output(
        "speech", capabilities.speech_output_suffix or ".mp3"
    )
    response_format = Path(output_path).suffix.lower().lstrip(".") or None
    if capabilities.is_groq:
        if chosen_voice not in capabilities.speech_voices:
            raise ValueError(
                f"Groq speech voice must be one of: {', '.join(capabilities.speech_voices)}"
            )
        if response_format != "wav":
            raise ValueError(
                "Groq speech output must use .wav because Groq only supports response_format=wav."
            )
    provider = provider_from_auth(
        store=store,
        provider_choice=options.provider_choice,
        explicit_base_url=options.base_url,
        provider_factory=provider_factory,
        auth_loader=auth_loader,
    )
    return provider.generate_speech(
        options.text,
        model=chosen_model,
        voice=chosen_voice,
        output_path=output_path,
        response_format=response_format,
    )


def run_images_generate(
    prompt: str,
    *,
    model: str | None,
    output: str,
    count: int,
    provider_choice: str | None,
    base_url: str | None,
    store: StateStore | None = None,
    provider_factory: ProviderFactory | None = None,
    auth_loader: AuthLoader | None = None,
) -> list[str]:
    store = store or StateStore()
    chosen_model = model or store.get_config("image_model") or "gpt-image-1"
    provider = provider_from_auth(
        store=store,
        provider_choice=provider_choice,
        explicit_base_url=base_url,
        provider_factory=provider_factory,
        auth_loader=auth_loader,
    )
    return provider.generate_images(
        prompt, model=chosen_model, output_path=output, count=count
    )


def run_image_edit(
    image_path: str,
    prompt: str,
    *,
    model: str | None,
    output: str,
    provider_choice: str | None,
    base_url: str | None,
    store: StateStore | None = None,
    provider_factory: ProviderFactory | None = None,
    auth_loader: AuthLoader | None = None,
) -> str:
    store = store or StateStore()
    chosen_model = model or store.get_config("image_model") or "gpt-image-1"
    provider = provider_from_auth(
        store=store,
        provider_choice=provider_choice,
        explicit_base_url=base_url,
        provider_factory=provider_factory,
        auth_loader=auth_loader,
    )
    return provider.edit_image(
        image_path, prompt, model=chosen_model, output_path=output
    )


def run_video_generate(
    prompt: str,
    *,
    model: str | None,
    image_path: str | None,
    output: str,
    seconds: str | None,
    size: str | None,
    poll_interval: float,
    timeout: float,
    provider_choice: str | None,
    base_url: str | None,
    store: StateStore | None = None,
    provider_factory: ProviderFactory | None = None,
    auth_loader: AuthLoader | None = None,
) -> str:
    store = store or StateStore()
    chosen_model = model or store.get_config("video_model")
    if not chosen_model:
        raise ValueError("Video generation requires --model or config video_model.")
    provider = provider_from_auth(
        store=store,
        provider_choice=provider_choice,
        explicit_base_url=base_url,
        provider_factory=provider_factory,
        auth_loader=auth_loader,
    )
    return provider.generate_video(
        prompt,
        model=chosen_model,
        output_path=output,
        image_path=image_path,
        seconds=seconds,
        size=size,
        poll_interval=poll_interval,
        timeout=timeout,
    )


def require_fragment(store: StateStore, name: str) -> str:
    text = store.get_fragment(name)
    if text is None:
        raise ValueError(f"Unknown fragment: {name}")
    return text


def session_replay_input(turns: list[Any], prompt: str) -> str:
    parts = ["Continue this conversation using the prior transcript below.", ""]
    for turn in turns:
        parts.extend(["User:", turn.prompt, "", "Assistant:", turn.output, ""])
    parts.extend(["User:", prompt])
    return "\n".join(parts)


def should_replay_session(
    existing: Any, *, provider_name: str, base_url: str | None
) -> bool:
    if provider_name == "codex":
        return True
    if base_url:
        return True
    if existing.provider != provider_name:
        return True
    return normalize_base_url(existing.base_url) != normalize_base_url(base_url)


def provider_from_auth(
    *,
    store: StateStore | None = None,
    provider_choice: str | None = None,
    explicit_base_url: str | None = None,
    provider_factory: ProviderFactory | None = None,
    auth_loader: AuthLoader | None = None,
) -> Any:
    store = store or StateStore()
    loader = auth_loader or load_auth_for_provider
    factory = provider_factory or make_provider
    auth = loader(provider_choice)
    chosen_base_url = configured_base_url(
        store=store,
        provider_choice=provider_choice,
        explicit_base_url=explicit_base_url,
    )
    return factory(auth, chosen_base_url)


def audio_model_default(
    model: str | None,
    *,
    store: StateStore,
    provider_choice: str | None,
    openai_default: str,
    groq_default: str,
    explicit_base_url: str | None = None,
) -> str:
    if model:
        return model
    base_url = configured_base_url(
        store=store,
        provider_choice=provider_choice,
        explicit_base_url=explicit_base_url,
    )
    if capabilities_for_base_url(base_url).is_groq:
        return groq_default
    return openai_default


def prompt_model_default(
    model: str | None,
    *,
    existing_model: str | None,
    store: StateStore,
    base_url: str | None,
    has_vision_input: bool = False,
) -> str:
    configured_model = store.get_config("default_model")
    if model or existing_model or configured_model:
        return model or existing_model or configured_model or DEFAULT_MODEL
    provider_default = default_model_for_base_url(base_url, vision=has_vision_input)
    if provider_default:
        return provider_default
    return DEFAULT_MODEL


def has_vision_input(attachments: list[Any]) -> bool:
    return any(
        attachment.kind == "image"
        or (
            attachment.kind == "file"
            and attachment.mime_type == "application/pdf"
            and Path(attachment.source).is_file()
        )
        for attachment in attachments
    )


def load_auth_for_provider(provider_choice: str | None) -> AuthInfo:
    preset = provider_preset(provider_choice)
    if preset:
        provider_key = os.environ.get(preset.api_key_env)
        if provider_key:
            return AuthInfo(
                provider="openai", api_key=provider_key, source=preset.api_key_env
            )
        try:
            return load_auth(provider="openai")
        except AuthError as exc:
            raise AuthError(
                f"No usable auth found for {preset.name}. Set {preset.api_key_env}."
            ) from exc
    return load_auth(provider=provider_choice)


def configured_base_url(
    *,
    store: StateStore,
    provider_choice: str | None,
    explicit_base_url: str | None = None,
) -> str | None:
    if provider_choice == "codex":
        return None
    if explicit_base_url:
        return explicit_base_url
    preset = provider_preset(provider_choice)
    if preset:
        return preset.base_url
    if provider_choice == "openai":
        return None
    return store.get_config("base_url") or os.environ.get("LLMRUN_BASE_URL")


def timestamped_output(prefix: str, suffix: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return str(Path.cwd() / f"{prefix}-{stamp}{suffix}")
