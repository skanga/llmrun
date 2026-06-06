from click.testing import CliRunner
from pathlib import Path

from llmrun.cli import app

runner = CliRunner()


def test_audio_transcribe_prints_text(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    audio = tmp_path / "note.mp3"
    audio.write_bytes(b"audio")
    seen = []

    class FakeProvider:
        def transcribe_audio(self, path, *, model):
            seen.append((path, model))
            return "transcript"

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider()
    )

    result = runner.invoke(
        app, ["audio", "transcribe", str(audio), "--model", "gpt-4o-transcribe"]
    )

    assert result.exit_code == 0
    assert result.stdout == "transcript\n"
    assert seen == [(str(audio), "gpt-4o-transcribe")]


def test_audio_transcribe_uses_groq_default_model(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OPENAI_API_KEY", "gsk-test")
    monkeypatch.setenv("LLMRUN_BASE_URL", "https://api.groq.com/openai/v1")
    audio = tmp_path / "note.mp3"
    audio.write_bytes(b"audio")
    seen = []

    class FakeProvider:
        def transcribe_audio(self, path, *, model):
            seen.append(model)
            return "transcript"

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider()
    )

    result = runner.invoke(app, ["audio", "transcribe", str(audio)])

    assert result.exit_code == 0
    assert seen == ["whisper-large-v3-turbo"]


def test_audio_transcribe_root_base_url_reaches_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OPENAI_API_KEY", "provider-key")
    audio = tmp_path / "note.mp3"
    audio.write_bytes(b"audio")
    seen = []

    class FakeProvider:
        def transcribe_audio(self, path, *, model):
            return "transcript"

    monkeypatch.setattr(
        "llmrun.cli.make_provider",
        lambda auth, base_url=None: seen.append(base_url) or FakeProvider(),
    )

    result = runner.invoke(
        app, ["--base-url", "https://audio.test/v1", "audio", "transcribe", str(audio)]
    )

    assert result.exit_code == 0
    assert seen == ["https://audio.test/v1"]


def test_audio_translate_prints_text(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    audio = tmp_path / "note.mp3"
    audio.write_bytes(b"audio")
    seen = []

    class FakeProvider:
        def translate_audio(self, path, *, model):
            seen.append((path, model))
            return "translation"

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider()
    )

    result = runner.invoke(
        app, ["audio", "translate", str(audio), "--model", "whisper-1"]
    )

    assert result.exit_code == 0
    assert result.stdout == "translation\n"
    assert seen == [(str(audio), "whisper-1")]


def test_audio_translate_uses_groq_default_model(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OPENAI_API_KEY", "gsk-test")
    monkeypatch.setenv("LLMRUN_BASE_URL", "https://api.groq.com/openai/v1")
    audio = tmp_path / "note.mp3"
    audio.write_bytes(b"audio")
    seen = []

    class FakeProvider:
        def translate_audio(self, path, *, model):
            seen.append(model)
            return "translation"

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider()
    )

    result = runner.invoke(app, ["audio", "translate", str(audio)])

    assert result.exit_code == 0
    assert seen == ["whisper-large-v3"]


def test_audio_translate_root_base_url_reaches_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OPENAI_API_KEY", "provider-key")
    audio = tmp_path / "note.mp3"
    audio.write_bytes(b"audio")
    seen = []

    class FakeProvider:
        def translate_audio(self, path, *, model):
            return "translation"

    monkeypatch.setattr(
        "llmrun.cli.make_provider",
        lambda auth, base_url=None: seen.append(base_url) or FakeProvider(),
    )

    result = runner.invoke(
        app, ["--base-url", "https://audio.test/v1", "audio", "translate", str(audio)]
    )

    assert result.exit_code == 0
    assert seen == ["https://audio.test/v1"]


def test_audio_speech_writes_requested_output(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    output = tmp_path / "speech.mp3"
    seen = []

    class FakeProvider:
        def generate_speech(
            self, text, *, model, voice, output_path, response_format=None
        ):
            seen.append((text, model, voice, output_path, response_format))
            output.write_bytes(b"mp3")
            return str(output)

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider()
    )

    result = runner.invoke(
        app, ["audio", "speech", "hello", "--voice", "alloy", "--output", str(output)]
    )

    assert result.exit_code == 0
    assert output.read_bytes() == b"mp3"
    assert str(output) in result.stdout
    assert seen == [("hello", "gpt-4o-mini-tts", "alloy", str(output), "mp3")]


def test_audio_speech_help_and_short_options(monkeypatch, tmp_path):
    help_result = runner.invoke(app, ["audio", "speech", "-h"])

    assert help_result.exit_code == 0
    assert "-v, --voice TEXT" in help_result.stdout
    assert "-o, --output TEXT" in help_result.stdout
    assert "Voice name for speech output." in help_result.stdout

    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    output = tmp_path / "speech.mp3"
    seen = []

    class FakeProvider:
        def generate_speech(
            self, text, *, model, voice, output_path, response_format=None
        ):
            seen.append((model, voice, output_path, response_format))
            output.write_bytes(b"mp3")
            return str(output)

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider()
    )

    result = runner.invoke(
        app,
        [
            "audio",
            "speech",
            "hello",
            "-m",
            "tts-test",
            "-v",
            "alloy",
            "-o",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert seen == [("tts-test", "alloy", str(output), "mp3")]


def test_audio_speech_uses_groq_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OPENAI_API_KEY", "gsk-test")
    monkeypatch.setenv("LLMRUN_BASE_URL", "https://api.groq.com/openai/v1")
    output = tmp_path / "speech.wav"
    seen = []

    class FakeProvider:
        def generate_speech(
            self, text, *, model, voice, output_path, response_format=None
        ):
            seen.append((model, voice, output_path, response_format))
            output.write_bytes(b"wav")
            return str(output)

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider()
    )

    result = runner.invoke(app, ["audio", "speech", "hello", "--output", str(output)])

    assert result.exit_code == 0
    assert seen == [("canopylabs/orpheus-v1-english", "troy", str(output), "wav")]


def test_audio_speech_root_base_url_controls_groq_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OPENAI_API_KEY", "gsk-test")
    output = tmp_path / "speech.wav"
    seen = []

    class FakeProvider:
        def generate_speech(
            self, text, *, model, voice, output_path, response_format=None
        ):
            seen.append((model, voice, output_path, response_format, self.base_url))
            output.write_bytes(b"wav")
            return str(output)

        def __init__(self, base_url):
            self.base_url = base_url

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider(base_url)
    )

    result = runner.invoke(
        app,
        [
            "--base-url",
            "https://api.groq.com/openai/v1",
            "audio",
            "speech",
            "hello",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert seen == [
        (
            "canopylabs/orpheus-v1-english",
            "troy",
            str(output),
            "wav",
            "https://api.groq.com/openai/v1",
        )
    ]


def test_audio_speech_rejects_groq_mp3_output(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OPENAI_API_KEY", "gsk-test")
    monkeypatch.setenv("LLMRUN_BASE_URL", "https://api.groq.com/openai/v1")

    result = runner.invoke(
        app,
        [
            "audio",
            "speech",
            "hello",
            "--voice",
            "troy",
            "--output",
            str(tmp_path / "speech.mp3"),
        ],
    )

    assert result.exit_code == 1
    assert "Groq speech output must use .wav" in result.stderr


def test_audio_speech_rejects_groq_invalid_voice(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OPENAI_API_KEY", "gsk-test")
    monkeypatch.setenv("LLMRUN_BASE_URL", "https://api.groq.com/openai/v1")

    result = runner.invoke(
        app,
        [
            "audio",
            "speech",
            "hello",
            "--voice",
            "alloy",
            "--output",
            str(tmp_path / "speech.wav"),
        ],
    )

    assert result.exit_code == 1
    assert "Groq speech voice must be one of" in result.stderr


def test_audio_speech_uses_timestamped_output_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.chdir(tmp_path)
    seen = []

    class FakeProvider:
        def generate_speech(
            self, text, *, model, voice, output_path, response_format=None
        ):
            seen.append(output_path)
            return output_path

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider()
    )

    result = runner.invoke(app, ["audio", "speech", "hello", "--voice", "alloy"])

    assert result.exit_code == 0
    assert len(seen) == 1
    assert seen[0].startswith(str(tmp_path / "speech-"))
    assert seen[0].endswith(".mp3")
    assert seen[0] in result.stdout


def test_audio_commands_report_codex_unsupported(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    class FakeProvider:
        def transcribe_audio(self, path, *, model):
            from llmrun.providers import ProviderError

            raise ProviderError(
                "Codex OAuth does not support audio workflows yet; use OPENAI_API_KEY"
            )

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider()
    )

    result = runner.invoke(app, ["audio", "transcribe", str(tmp_path / "note.mp3")])

    assert result.exit_code == 1
    assert "Codex OAuth does not support audio workflows" in result.stderr


def test_images_generate_writes_requested_output(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    output = tmp_path / "image.png"
    seen = []

    class FakeProvider:
        def generate_images(self, prompt, *, model, output_path, count):
            seen.append((prompt, model, output_path, count))
            output.write_bytes(b"png")
            return [str(output)]

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider()
    )

    result = runner.invoke(
        app, ["images", "generate", "draw a chart", "--output", str(output)]
    )

    assert result.exit_code == 0
    assert output.read_bytes() == b"png"
    assert seen == [("draw a chart", "gpt-image-1", str(output), 1)]


def test_images_generate_help_and_short_options(monkeypatch, tmp_path):
    help_result = runner.invoke(app, ["images", "generate", "-h"])

    assert help_result.exit_code == 0
    assert "-o, --output TEXT" in help_result.stdout
    assert "-n, --count INTEGER RANGE" in help_result.stdout
    assert "Number of images to generate." in help_result.stdout

    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    output = tmp_path / "image.png"
    seen = []

    class FakeProvider:
        def generate_images(self, prompt, *, model, output_path, count):
            seen.append((model, output_path, count))
            output.write_bytes(b"png")
            return [str(output)]

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider()
    )

    result = runner.invoke(
        app,
        [
            "images",
            "generate",
            "draw",
            "-m",
            "image-test",
            "-o",
            str(output),
            "-n",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert seen == [("image-test", str(output), 2)]


def test_images_generate_provider_codex_overrides_openai_key(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OPENAI_API_KEY", "gsk-test")
    monkeypatch.setenv("LLMRUN_BASE_URL", "https://api.groq.com/openai/v1")
    codex_auth = tmp_path / "auth.json"
    codex_auth.write_text(
        '{"tokens":{"access_token":"access","refresh_token":"refresh"}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_AUTH_JSON_PATH", str(codex_auth))
    output = tmp_path / "image.png"
    seen = []

    class FakeProvider:
        def __init__(self, provider_name, base_url):
            self.provider_name = provider_name
            self.base_url = base_url

        def generate_images(self, prompt, *, model, output_path, count):
            seen.append(
                (self.provider_name, self.base_url, prompt, model, output_path, count)
            )
            output.write_bytes(b"png")
            return [str(output)]

    monkeypatch.setattr(
        "llmrun.cli.make_provider",
        lambda auth, base_url=None: FakeProvider(auth.provider, base_url),
    )

    result = runner.invoke(
        app,
        [
            "--provider",
            "codex",
            "images",
            "generate",
            "draw a chart",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert seen == [("codex", None, "draw a chart", "gpt-image-1", str(output), 1)]


def test_images_generate_uses_configured_image_model(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    seen = []

    class FakeProvider:
        def generate_images(self, prompt, *, model, output_path, count):
            seen.append(model)
            return [output_path]

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider()
    )

    assert (
        runner.invoke(
            app, ["config", "set", "image_model", "gpt-image-custom"]
        ).exit_code
        == 0
    )
    result = runner.invoke(
        app, ["images", "generate", "draw", "--output", str(tmp_path / "image.png")]
    )

    assert result.exit_code == 0
    assert seen == ["gpt-image-custom"]


def test_images_generate_root_base_url_reaches_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OPENAI_API_KEY", "provider-key")
    output = tmp_path / "image.png"
    seen = []

    class FakeProvider:
        def generate_images(self, prompt, *, model, output_path, count):
            return [output_path]

    monkeypatch.setattr(
        "llmrun.cli.make_provider",
        lambda auth, base_url=None: seen.append(base_url) or FakeProvider(),
    )

    result = runner.invoke(
        app,
        [
            "--base-url",
            "https://images.test/v1",
            "images",
            "generate",
            "draw",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert seen == ["https://images.test/v1"]


def test_images_generate_count_writes_numbered_outputs(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    output = tmp_path / "image.png"
    seen = []

    class FakeProvider:
        def generate_images(self, prompt, *, model, output_path, count):
            seen.append((output_path, count))
            paths = [
                str(tmp_path / f"image-{index}.png") for index in range(1, count + 1)
            ]
            for path in paths:
                Path(path).write_bytes(b"png")
            return paths

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider()
    )

    result = runner.invoke(
        app,
        ["images", "generate", "draw a chart", "--output", str(output), "--count", "3"],
    )

    assert result.exit_code == 0
    assert seen == [(str(output), 3)]
    assert str(tmp_path / "image-1.png") in result.stdout
    assert str(tmp_path / "image-3.png") in result.stdout


def test_images_edit_passes_input_image(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    image = tmp_path / "input.png"
    image.write_bytes(b"png")
    output = tmp_path / "edited.png"
    seen = []

    class FakeProvider:
        def edit_image(self, image_path, prompt, *, model, output_path):
            seen.append((image_path, prompt, model, output_path))
            output.write_bytes(b"edited")
            return str(output)

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider()
    )

    result = runner.invoke(
        app,
        [
            "images",
            "edit",
            "--image",
            str(image),
            "make it blue",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert output.read_bytes() == b"edited"
    assert seen == [(str(image), "make it blue", "gpt-image-1", str(output))]


def test_images_edit_root_base_url_reaches_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OPENAI_API_KEY", "provider-key")
    image = tmp_path / "input.png"
    image.write_bytes(b"png")
    output = tmp_path / "edited.png"
    seen = []

    class FakeProvider:
        def edit_image(self, image_path, prompt, *, model, output_path):
            return output_path

    monkeypatch.setattr(
        "llmrun.cli.make_provider",
        lambda auth, base_url=None: seen.append(base_url) or FakeProvider(),
    )

    result = runner.invoke(
        app,
        [
            "--base-url",
            "https://images.test/v1",
            "images",
            "edit",
            "--image",
            str(image),
            "make it blue",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert seen == ["https://images.test/v1"]


def test_videos_generate_requires_model_or_config(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    result = runner.invoke(
        app,
        ["videos", "generate", "make a demo", "--output", str(tmp_path / "video.mp4")],
    )

    assert result.exit_code == 1
    assert "requires --model or config video_model" in result.stderr


def test_videos_generate_uses_configured_video_model(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    output = tmp_path / "video.mp4"
    seen = []

    class FakeProvider:
        def generate_video(
            self,
            prompt,
            *,
            model,
            output_path,
            image_path=None,
            seconds=None,
            size=None,
            poll_interval=2.0,
            timeout=900.0,
        ):
            seen.append(
                (
                    prompt,
                    model,
                    output_path,
                    image_path,
                    seconds,
                    size,
                    poll_interval,
                    timeout,
                )
            )
            output.write_bytes(b"mp4")
            return str(output)

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider()
    )

    assert runner.invoke(app, ["config", "set", "video_model", "sora-2"]).exit_code == 0
    result = runner.invoke(
        app,
        [
            "videos",
            "generate",
            "make a demo",
            "--output",
            str(output),
            "--seconds",
            "8",
            "--size",
            "1280x720",
            "--poll-interval",
            "0",
            "--timeout",
            "30",
        ],
    )

    assert result.exit_code == 0
    assert output.read_bytes() == b"mp4"
    assert seen == [
        ("make a demo", "sora-2", str(output), None, "8", "1280x720", 0.0, 30.0)
    ]
    assert str(output) in result.stdout


def test_videos_generate_help_lists_generation_controls():
    result = runner.invoke(app, ["videos", "generate", "-h"])

    assert result.exit_code == 0
    assert "-m, --model TEXT" in result.stdout
    assert "-i, --image TEXT" in result.stdout
    assert "-o, --output TEXT" in result.stdout
    assert "Poll timeout in seconds." in result.stdout


def test_videos_generate_passes_input_image(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    image = tmp_path / "frame.png"
    image.write_bytes(b"png")
    output = tmp_path / "video.mp4"
    seen = []

    class FakeProvider:
        def generate_video(
            self,
            prompt,
            *,
            model,
            output_path,
            image_path=None,
            seconds=None,
            size=None,
            poll_interval=2.0,
            timeout=900.0,
        ):
            seen.append(image_path)
            output.write_bytes(b"mp4")
            return str(output)

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider()
    )

    result = runner.invoke(
        app,
        [
            "videos",
            "generate",
            "make a demo",
            "--model",
            "sora-2-pro",
            "--image",
            str(image),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert seen == [str(image)]


def test_videos_generate_root_base_url_reaches_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OPENAI_API_KEY", "provider-key")
    output = tmp_path / "video.mp4"
    seen = []

    class FakeProvider:
        def generate_video(
            self,
            prompt,
            *,
            model,
            output_path,
            image_path=None,
            seconds=None,
            size=None,
            poll_interval=2.0,
            timeout=900.0,
        ):
            seen.append(self.base_url)
            return output_path

        def __init__(self, base_url):
            self.base_url = base_url

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider(base_url)
    )

    result = runner.invoke(
        app,
        [
            "--base-url",
            "https://videos.test/v1",
            "videos",
            "generate",
            "make a demo",
            "--model",
            "sora-2",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert seen == ["https://videos.test/v1"]


def test_videos_generate_reports_unsupported_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    class FakeProvider:
        def generate_video(
            self,
            prompt,
            *,
            model,
            output_path,
            image_path=None,
            seconds=None,
            size=None,
            poll_interval=2.0,
            timeout=900.0,
        ):
            raise NotImplementedError(
                "Video generation is not implemented for this provider yet."
            )

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider()
    )

    result = runner.invoke(
        app,
        [
            "videos",
            "generate",
            "make a demo",
            "--model",
            "sora-2",
            "--output",
            str(tmp_path / "video.mp4"),
        ],
    )

    assert result.exit_code == 1
    assert "Video generation is not implemented" in result.stderr
