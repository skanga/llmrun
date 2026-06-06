from pathlib import Path

import pytest

from llmrun.auth import AuthInfo
from llmrun.runners import (
    AudioSpeechOptions,
    PromptOptions,
    PromptRunner,
    configured_base_url,
    run_audio_speech,
)
from llmrun.state import StateStore


def test_prompt_runner_executes_prompt_and_updates_session(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    store = StateStore(tmp_path)
    seen = []

    class FakeProvider:
        name = "openai"

        def complete(self, request, on_delta=None):
            seen.append((request.input, request.model, request.stream))
            return type("Result", (), {"text": "ok", "response_id": "resp_1"})()

    runner = PromptRunner(
        store=store,
        provider_factory=lambda auth, base_url=None: FakeProvider(),
    )

    result = runner.run(PromptOptions(prompt="hello", session="work", stream=False))

    assert result.text == "ok"
    assert seen == [("hello", "gpt-5.4-mini", False)]
    assert store.get_session("work").turns[0].output == "ok"


def test_audio_speech_runner_applies_groq_voice_and_format(monkeypatch, tmp_path):
    store = StateStore(tmp_path / "state")
    output = tmp_path / "speech.wav"
    seen = []

    class FakeProvider:
        def generate_speech(
            self, text, *, model, voice, output_path, response_format=None
        ):
            seen.append((text, model, voice, output_path, response_format))
            Path(output_path).write_bytes(b"wav")
            return output_path

    written = run_audio_speech(
        AudioSpeechOptions(text="hello", output=str(output), provider_choice="groq"),
        store=store,
        provider_factory=lambda auth, base_url=None: FakeProvider(),
        auth_loader=lambda provider_choice: AuthInfo(
            provider="openai", api_key="gsk-test"
        ),
    )

    assert written == str(output)
    assert seen == [
        ("hello", "canopylabs/orpheus-v1-english", "troy", str(output), "wav")
    ]


def test_audio_speech_runner_rejects_invalid_groq_voice(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="Groq speech voice must be one of"):
        run_audio_speech(
            AudioSpeechOptions(
                text="hello",
                voice="alloy",
                output=str(tmp_path / "speech.wav"),
                provider_choice="groq",
            ),
            store=StateStore(tmp_path / "state"),
            provider_factory=lambda auth, base_url=None: object(),
            auth_loader=lambda provider_choice: AuthInfo(
                provider="openai", api_key="gsk-test"
            ),
        )


def test_old_base_url_env_var_is_ignored_after_rename(monkeypatch, tmp_path):
    monkeypatch.setenv("ANYLLM_BASE_URL", "https://old.example/v1")

    assert (
        configured_base_url(store=StateStore(tmp_path / "state"), provider_choice=None)
        is None
    )
