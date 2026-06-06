from __future__ import annotations

from typing import Any

import click

from .auth import AuthError
from .prompts import render_template
from .provider_config import (
    OPENAI_COMPATIBLE_PROVIDER_PRESETS,
    capabilities_for_base_url,
    default_model_for_base_url as provider_default_model_for_base_url,
    provider_preset as get_provider_preset,
)
from .providers import ProviderError, make_provider
from .runners import (
    AudioSpeechOptions,
    PromptOptions,
    PromptRunner,
    audio_model_default,
    configured_base_url,
    load_auth_for_provider,
    provider_from_auth as runner_provider_from_auth,
    require_fragment,
    run_audio_speech,
    run_audio_transcribe,
    run_audio_translate,
    run_image_edit,
    run_images_generate,
    run_video_generate,
    session_replay_input,
    should_replay_session,
    timestamped_output,
)
from .state import StateStore


class PromptGroup(click.Group):
    def resolve_command(
        self, ctx: click.Context, args: list[str]
    ) -> tuple[str | None, click.Command | None, list[str]]:
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError:
            if args:
                return "__prompt__", self.commands["__prompt__"], args
            raise


@click.group(
    cls=PromptGroup,
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 110},
    help=(
        "Run one prompt directly, continue sessions, or use media generation subcommands.\n\n"
        "\b\n"
        "Examples:\n"
        '  llmrun -m gpt-5.4-mini "Briefly explain Newtons 3rd law"\n'
        '  llmrun -i screenshot.png "Describe what you see"\n'
        '  llmrun --provider groq -f report.pdf "Summarize this PDF"'
    ),
)
@click.option("-S", "--system", help="System instructions to apply before the prompt.")
@click.option(
    "-m", "--model", help="Model name for prompt, audio, image, or video commands."
)
@click.option(
    "-b", "--base-url", help="OpenAI-compatible API base URL for this command."
)
@click.option(
    "-p",
    "--provider",
    "provider_choice",
    type=click.Choice(["openai", "codex", *OPENAI_COMPATIBLE_PROVIDER_PRESETS.keys()]),
    help="Provider preset or auth mode to use for this command.",
)
@click.option(
    "-s", "--session", help="Session name for storing and continuing a conversation."
)
@click.option(
    "--fragment",
    "fragments",
    multiple=True,
    help="Saved fragment to prepend. Repeat to include several.",
)
@click.option(
    "-i",
    "--image",
    "images",
    multiple=True,
    help="Attach an image path or HTTPS URL. Repeat for multiple images.",
)
@click.option(
    "-f",
    "--file",
    "files",
    multiple=True,
    help="Attach a file path or HTTPS URL. Repeat for multiple files.",
)
@click.option(
    "--image-detail",
    type=click.Choice(["auto", "low", "high"]),
    default="auto",
    show_default=True,
    help="Detail level for attached images sent through Responses or vision fallback.",
)
@click.option(
    "-t",
    "--temperature",
    type=float,
    help="Sampling temperature for models that support it.",
)
@click.option(
    "-n",
    "--max-output-tokens",
    type=int,
    help="Maximum output tokens requested from the model.",
)
@click.option(
    "--reasoning-effort", help="Reasoning effort hint for models that support it."
)
@click.option(
    "--stream/--no-stream",
    default=True,
    help="Stream text as it arrives, or print only after completion.",
)
@click.option(
    "-j",
    "--json",
    "json_mode",
    is_flag=True,
    help="Request JSON output when the provider supports it.",
)
@click.pass_context
def app(
    ctx: click.Context,
    system: str | None,
    model: str | None,
    base_url: str | None,
    provider_choice: str | None,
    session: str | None,
    fragments: tuple[str, ...],
    images: tuple[str, ...],
    files: tuple[str, ...],
    image_detail: str,
    temperature: float | None,
    max_output_tokens: int | None,
    reasoning_effort: str | None,
    stream: bool,
    json_mode: bool,
) -> None:
    ctx.obj = {
        "system": system,
        "model": model,
        "base_url": base_url,
        "provider": provider_choice,
        "session": session,
        "fragments": list(fragments),
        "images": list(images),
        "files": list(files),
        "image_detail": image_detail,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "reasoning_effort": reasoning_effort,
        "stream": stream,
        "json_mode": json_mode,
    }
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@app.command(
    "__prompt__", hidden=True, context_settings={"ignore_unknown_options": True}
)
@click.argument("prompt_parts", nargs=-1)
@click.pass_context
def prompt_command(ctx: click.Context, prompt_parts: tuple[str, ...]) -> None:
    opts: dict[str, Any] = ctx.obj or {}
    prompt = " ".join(prompt_parts)
    if not prompt:
        click.echo(ctx.parent.get_help() if ctx.parent else "")
        return
    _run_prompt(
        prompt=prompt,
        system=opts.get("system"),
        model=opts.get("model"),
        base_url=opts.get("base_url"),
        provider_choice=opts.get("provider"),
        session=opts.get("session"),
        fragment_names=opts.get("fragments", []),
        image_sources=opts.get("images", []),
        file_sources=opts.get("files", []),
        image_detail=opts.get("image_detail", "auto"),
        temperature=opts.get("temperature"),
        max_output_tokens=opts.get("max_output_tokens"),
        reasoning_effort=opts.get("reasoning_effort"),
        stream=opts.get("stream", True),
        json_mode=opts.get("json_mode", False),
    )


@app.command(help="Start a simple interactive prompt loop for one session.")
@click.option(
    "-s", "--session", required=True, help="Session name to continue in the chat loop."
)
def chat(session: str) -> None:
    while True:
        try:
            prompt = input("> ")
        except EOFError:
            break
        if prompt.strip() in {"exit", "quit"}:
            break
        _run_prompt(
            prompt=prompt,
            system=None,
            model=None,
            base_url=None,
            provider_choice=None,
            session=session,
            fragment_names=[],
            image_sources=[],
            file_sources=[],
            image_detail="auto",
            temperature=None,
            max_output_tokens=None,
            reasoning_effort=None,
            stream=True,
            json_mode=False,
        )


@app.group(help="List, show, export, or delete saved sessions.")
def sessions() -> None:
    pass


@sessions.command("list", help="List saved session names.")
def sessions_list() -> None:
    for name in StateStore().list_sessions():
        click.echo(name)


@sessions.command("show", help="Show a session transcript as Markdown.")
@click.argument("name")
def sessions_show(name: str) -> None:
    store = StateStore()
    if store.get_session(name) is None:
        raise click.ClickException(f"Unknown session: {name}")
    click.echo(store.export_session(name, "markdown"))


@sessions.command("delete", help="Delete a saved session.")
@click.argument("name")
def sessions_delete(name: str) -> None:
    if not StateStore().delete_session(name):
        raise click.ClickException(f"Unknown session: {name}")


@sessions.command("export", help="Export a session transcript.")
@click.argument("name")
@click.option(
    "-f",
    "--format",
    "format_",
    default="markdown",
    show_default=True,
    help="Export format: markdown or json.",
)
def sessions_export(name: str, format_: str) -> None:
    try:
        click.echo(StateStore().export_session(name, format_), nl=False)
    except (KeyError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@app.group(help="Manage reusable prompt templates.")
def templates() -> None:
    pass


@templates.command("add", help="Save a template.")
@click.argument("name")
@click.argument("text")
def templates_add(name: str, text: str) -> None:
    StateStore().set_template(name, text)


@templates.command("list", help="List template names.")
def templates_list() -> None:
    for name in StateStore().list_templates():
        click.echo(name)


@templates.command("show", help="Show a saved template.")
@click.argument("name")
def templates_show(name: str) -> None:
    text = StateStore().get_template(name)
    if text is None:
        raise click.ClickException(f"Unknown template: {name}")
    click.echo(text)


@templates.command("delete", help="Delete a saved template.")
@click.argument("name")
def templates_delete(name: str) -> None:
    if not StateStore().delete_template(name):
        raise click.ClickException(f"Unknown template: {name}")


@templates.command("run", help="Render a saved template and send it as a prompt.")
@click.argument("name")
@click.argument("prompt")
@click.option("-m", "--model", help="Model name for this prompt.")
@click.option("-s", "--session", help="Session name to store or continue.")
@click.option(
    "--stream/--no-stream",
    default=True,
    help="Stream text as it arrives, or print only after completion.",
)
def templates_run(
    name: str, prompt: str, model: str | None, session: str | None, stream: bool
) -> None:
    store = StateStore()
    text = store.get_template(name)
    if text is None:
        raise click.ClickException(f"Unknown template: {name}")
    _run_prompt(
        prompt=render_template(text, prompt=prompt),
        system=None,
        model=model,
        base_url=None,
        provider_choice=None,
        session=session,
        fragment_names=[],
        image_sources=[],
        file_sources=[],
        image_detail="auto",
        temperature=None,
        max_output_tokens=None,
        reasoning_effort=None,
        stream=stream,
        json_mode=False,
    )


@app.group(help="Manage reusable prompt fragments.")
def fragments() -> None:
    pass


@fragments.command("add", help="Save a fragment.")
@click.argument("name")
@click.argument("text")
def fragments_add(name: str, text: str) -> None:
    StateStore().set_fragment(name, text)


@fragments.command("list", help="List fragment names.")
def fragments_list() -> None:
    for name in StateStore().list_fragments():
        click.echo(name)


@fragments.command("show", help="Show a saved fragment.")
@click.argument("name")
def fragments_show(name: str) -> None:
    text = StateStore().get_fragment(name)
    if text is None:
        raise click.ClickException(f"Unknown fragment: {name}")
    click.echo(text)


@fragments.command("delete", help="Delete a saved fragment.")
@click.argument("name")
def fragments_delete(name: str) -> None:
    if not StateStore().delete_fragment(name):
        raise click.ClickException(f"Unknown fragment: {name}")


@app.group(help="Get, set, or unset llmrun configuration values.")
def config() -> None:
    pass


@config.command("get", help="Print a configuration value.")
@click.argument("key")
def config_get(key: str) -> None:
    value = StateStore().get_config(key)
    if value is None:
        raise click.ClickException(f"Unknown config key: {key}")
    click.echo(value)


@config.command("set", help="Set a configuration value.")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    StateStore().set_config(key, value)


@config.command("unset", help="Remove a configuration value.")
@click.argument("key")
def config_unset(key: str) -> None:
    StateStore().unset_config(key)


@app.group(help="Transcribe, translate, or synthesize audio.")
def audio() -> None:
    pass


@audio.command("transcribe", help="Convert an audio file to text.")
@click.argument("path")
@click.option("-m", "--model", help="Transcription model name.")
@click.pass_context
def audio_transcribe(ctx: click.Context, path: str, model: str | None) -> None:
    try:
        click.echo(
            run_audio_transcribe(
                path,
                model=model,
                provider_choice=_root_provider_choice(ctx),
                base_url=_root_base_url(ctx),
                provider_factory=make_provider,
            )
        )
    except (AuthError, ProviderError) as exc:
        raise click.ClickException(str(exc)) from exc


@audio.command("translate", help="Translate an audio file to English text.")
@click.argument("path")
@click.option("-m", "--model", help="Audio translation model name.")
@click.pass_context
def audio_translate(ctx: click.Context, path: str, model: str | None) -> None:
    try:
        click.echo(
            run_audio_translate(
                path,
                model=model,
                provider_choice=_root_provider_choice(ctx),
                base_url=_root_base_url(ctx),
                provider_factory=make_provider,
            )
        )
    except (AuthError, ProviderError) as exc:
        raise click.ClickException(str(exc)) from exc


@audio.command("speech", help="Convert text to speech and write an audio file.")
@click.argument("text")
@click.option("-v", "--voice", help="Voice name for speech output.")
@click.option("-m", "--model", help="Speech model name.")
@click.option(
    "-o", "--output", help="Audio output path. Defaults to timestamped speech output."
)
@click.pass_context
def audio_speech(
    ctx: click.Context, text: str, voice: str, model: str | None, output: str | None
) -> None:
    try:
        written = run_audio_speech(
            AudioSpeechOptions(
                text=text,
                voice=voice,
                model=model,
                output=output,
                provider_choice=_root_provider_choice(ctx),
                base_url=_root_base_url(ctx),
            ),
            provider_factory=make_provider,
        )
    except (AuthError, ProviderError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(written)


@app.group(help="Generate or edit images.")
def images() -> None:
    pass


@images.command("generate", help="Generate one or more images from a prompt.")
@click.argument("prompt")
@click.option("-m", "--model", help="Image generation model name.")
@click.option(
    "-o",
    "--output",
    required=True,
    help="Output file path. Multiple images use numbered variants.",
)
@click.option(
    "-n",
    "--count",
    type=click.IntRange(1, 10),
    default=1,
    show_default=True,
    help="Number of images to generate.",
)
@click.pass_context
def images_generate(
    ctx: click.Context, prompt: str, model: str | None, output: str, count: int
) -> None:
    try:
        written = run_images_generate(
            prompt,
            model=model,
            output=output,
            count=count,
            provider_choice=_root_provider_choice(ctx),
            base_url=_root_base_url(ctx),
            provider_factory=make_provider,
        )
    except (AuthError, ProviderError) as exc:
        raise click.ClickException(str(exc)) from exc
    for path in written:
        click.echo(path)


@images.command("edit", help="Edit an image using a prompt.")
@click.option("-i", "--image", "image_path", required=True, help="Input image path.")
@click.argument("prompt")
@click.option("-m", "--model", help="Image editing model name.")
@click.option("-o", "--output", required=True, help="Output image path.")
@click.pass_context
def images_edit(
    ctx: click.Context, image_path: str, prompt: str, model: str | None, output: str
) -> None:
    try:
        written = run_image_edit(
            image_path,
            prompt,
            model=model,
            output=output,
            provider_choice=_root_provider_choice(ctx),
            base_url=_root_base_url(ctx),
            provider_factory=make_provider,
        )
    except (AuthError, ProviderError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(written)


@app.group(help="Generate videos and download completed artifacts.")
def videos() -> None:
    pass


@videos.command(
    "generate",
    help="Generate a video from a prompt, optionally using a first frame image.",
)
@click.argument("prompt")
@click.option(
    "-m",
    "--model",
    help="Video generation model name. Required unless config video_model is set.",
)
@click.option("-i", "--image", "image_path", help="Optional first-frame image path.")
@click.option("-o", "--output", required=True, help="Output video path.")
@click.option(
    "--seconds",
    type=click.Choice(["4", "8", "12"]),
    help="Requested video duration in seconds.",
)
@click.option(
    "--size",
    type=click.Choice(["720x1280", "1280x720", "1024x1792", "1792x1024"]),
    help="Requested video size.",
)
@click.option(
    "--poll-interval",
    type=float,
    default=2.0,
    show_default=True,
    help="Seconds between video job status checks.",
)
@click.option(
    "--timeout",
    type=float,
    default=900.0,
    show_default=True,
    help="Poll timeout in seconds.",
)
@click.pass_context
def videos_generate(
    ctx: click.Context,
    prompt: str,
    model: str | None,
    image_path: str | None,
    output: str,
    seconds: str | None,
    size: str | None,
    poll_interval: float,
    timeout: float,
) -> None:
    try:
        written = run_video_generate(
            prompt,
            model=model,
            output=output,
            image_path=image_path,
            seconds=seconds,
            size=size,
            poll_interval=poll_interval,
            timeout=timeout,
            provider_choice=_root_provider_choice(ctx),
            base_url=_root_base_url(ctx),
            provider_factory=make_provider,
        )
    except (AuthError, ProviderError, NotImplementedError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(written)


def _run_prompt(
    *,
    prompt: str,
    system: str | None,
    model: str | None,
    base_url: str | None,
    provider_choice: str | None,
    session: str | None,
    fragment_names: list[str],
    image_sources: list[str],
    file_sources: list[str],
    image_detail: str,
    temperature: float | None,
    max_output_tokens: int | None,
    reasoning_effort: str | None,
    stream: bool,
    json_mode: bool,
) -> None:
    try:
        result = PromptRunner(provider_factory=make_provider).run(
            PromptOptions(
                prompt=prompt,
                system=system,
                model=model,
                base_url=base_url,
                provider_choice=provider_choice,
                session=session,
                fragment_names=fragment_names,
                image_sources=image_sources,
                file_sources=file_sources,
                image_detail=image_detail,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                reasoning_effort=reasoning_effort,
                stream=stream,
                json_mode=json_mode,
            ),
            on_delta=lambda delta: click.echo(delta, nl=False),
        )
    except (AuthError, ProviderError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    if not stream:
        click.echo(result.text)
    elif result.text and not result.emitted_stream:
        click.echo(result.text)
    elif result.text:
        click.echo()


def _require_fragment(store: StateStore, name: str) -> str:
    try:
        return require_fragment(store, name)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


def _session_replay_input(turns: list[Any], prompt: str) -> str:
    return session_replay_input(turns, prompt)


def _should_replay_session(
    existing: Any, *, provider_name: str, base_url: str | None
) -> bool:
    return should_replay_session(
        existing, provider_name=provider_name, base_url=base_url
    )


def _normalize_base_url(base_url: str | None) -> str | None:
    from .provider_config import normalize_base_url

    return normalize_base_url(base_url)


def _provider_from_auth(
    *,
    store: StateStore | None = None,
    provider_choice: str | None = None,
    explicit_base_url: str | None = None,
) -> Any:
    try:
        return runner_provider_from_auth(
            store=store,
            provider_choice=provider_choice,
            explicit_base_url=explicit_base_url,
            provider_factory=make_provider,
        )
    except AuthError as exc:
        raise click.ClickException(str(exc)) from exc


def _audio_model_default(
    model: str | None,
    *,
    store: StateStore,
    provider_choice: str | None,
    openai_default: str,
    groq_default: str,
    explicit_base_url: str | None = None,
) -> str:
    return audio_model_default(
        model,
        store=store,
        provider_choice=provider_choice,
        explicit_base_url=explicit_base_url,
        openai_default=openai_default,
        groq_default=groq_default,
    )


def _prompt_model_default(
    model: str | None,
    *,
    existing_model: str | None,
    store: StateStore,
    base_url: str | None,
) -> str:
    from .runners import prompt_model_default

    return prompt_model_default(
        model, existing_model=existing_model, store=store, base_url=base_url
    )


def _load_auth_for_provider(provider_choice: str | None) -> Any:
    return load_auth_for_provider(provider_choice)


def _configured_base_url(
    *,
    store: StateStore,
    provider_choice: str | None,
    explicit_base_url: str | None = None,
) -> str | None:
    return configured_base_url(
        store=store,
        provider_choice=provider_choice,
        explicit_base_url=explicit_base_url,
    )


def _root_provider_choice(ctx: click.Context) -> str | None:
    root = ctx.find_root()
    opts: dict[str, Any] = root.obj or {}
    return opts.get("provider")


def _root_base_url(ctx: click.Context) -> str | None:
    root = ctx.find_root()
    opts: dict[str, Any] = root.obj or {}
    return opts.get("base_url")


def _provider_preset(provider_choice: str | None) -> dict[str, str] | None:
    preset = get_provider_preset(provider_choice)
    if not preset:
        return None
    return {
        "base_url": preset.base_url,
        "api_key_env": preset.api_key_env,
        "default_model": preset.default_model,
    }


def _default_model_for_base_url(base_url: str | None) -> str | None:
    return provider_default_model_for_base_url(base_url)


def _is_groq_base_url(base_url: str | None) -> bool:
    return capabilities_for_base_url(base_url).is_groq


def _timestamped_output(prefix: str, suffix: str) -> str:
    return timestamped_output(prefix, suffix)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
