import pytest
import httpx
import base64
from datetime import UTC, datetime, timedelta
import sys
import types

from llmrun import providers
from llmrun.attachments import Attachment
from llmrun.auth import AuthInfo, OAuthTokens
from llmrun.providers import (
    CodexOAuthProvider,
    OpenAIProvider,
    ProviderError,
    ProviderRequest,
    make_provider,
)


class FakeResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("Response", (), {"id": "resp_1", "output_text": "hello"})()


class FakeOpenAI:
    def __init__(self, api_key, base_url=None):
        self.api_key = api_key
        self.base_url = base_url
        self.responses = FakeResponses()


def test_openai_provider_calls_responses_create(monkeypatch):
    created = {}

    def factory(api_key, base_url=None):
        client = FakeOpenAI(api_key)
        created["client"] = client
        return client

    provider = OpenAIProvider(
        AuthInfo(provider="openai", api_key="sk-test"), client_factory=factory
    )
    result = provider.complete(
        ProviderRequest(
            input="hi",
            instructions="inst",
            model="gpt-test",
            previous_response_id="prev",
            temperature=0.2,
            max_output_tokens=10,
            reasoning_effort="low",
            json_mode=True,
            stream=False,
        )
    )

    assert result.text == "hello"
    assert result.response_id == "resp_1"
    assert created["client"].responses.calls[0]["previous_response_id"] == "prev"
    assert (
        created["client"].responses.calls[0]["text"]["format"]["type"] == "json_object"
    )


def test_provider_request_defaults_to_empty_input():
    request = ProviderRequest(model="gpt-test")

    assert request.input == ""
    assert request.attachments == []


def test_openai_provider_requires_api_key():
    with pytest.raises(ValueError, match="requires an API key"):
        OpenAIProvider(AuthInfo(provider="openai"))


def test_make_provider_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported provider"):
        make_provider(AuthInfo(provider="other"))


def test_openai_provider_passes_base_url_to_client_factory():
    created = {}

    def factory(api_key, base_url=None):
        client = FakeOpenAI(api_key, base_url=base_url)
        created["client"] = client
        return client

    provider = OpenAIProvider(
        AuthInfo(provider="openai", api_key="sk-test"),
        client_factory=factory,
        base_url="https://example.test/v1",
    )

    provider.complete(ProviderRequest(input="hi", model="gpt-test"))

    assert created["client"].api_key == "sk-test"
    assert created["client"].base_url == "https://example.test/v1"


def test_openai_provider_wraps_responses_error_without_fallback():
    class Responses:
        def create(self, **kwargs):
            raise RuntimeError("bad\nresponse")

    class Client:
        responses = Responses()

    provider = OpenAIProvider(
        AuthInfo(provider="openai", api_key="sk-test"),
        client_factory=lambda api_key, base_url=None: Client(),
    )

    with pytest.raises(ProviderError) as exc:
        provider.complete(ProviderRequest(input="hi", model="gpt-test"))

    assert "bad response" in str(exc.value)


def test_openai_provider_base_url_attachment_error_when_no_chat_fallback_supported():
    class Responses:
        def create(self, **kwargs):
            raise RuntimeError("responses endpoint unsupported")

    class Client:
        responses = Responses()

    provider = OpenAIProvider(
        AuthInfo(provider="openai", api_key="provider-key"),
        client_factory=lambda api_key, base_url=None: Client(),
        base_url="https://provider.test/v1",
    )

    with pytest.raises(ProviderError, match="fallback only supports image attachments"):
        provider.complete(
            ProviderRequest(
                input="read",
                model="model",
                attachments=[
                    Attachment(
                        kind="file",
                        source="report.txt",
                        filename="report.txt",
                        mime_type="text/plain",
                        payload="data:text/plain;base64,abc",
                    )
                ],
            )
        )


def test_openai_chat_fallback_error_is_wrapped():
    class Responses:
        def create(self, **kwargs):
            raise RuntimeError("responses endpoint unsupported")

    class ChatCompletions:
        def create(self, **kwargs):
            raise RuntimeError("chat\nfailed")

    class Client:
        responses = Responses()
        chat = type("Chat", (), {"completions": ChatCompletions()})()

    provider = OpenAIProvider(
        AuthInfo(provider="openai", api_key="provider-key"),
        client_factory=lambda api_key, base_url=None: Client(),
        base_url="https://provider.test/v1",
    )

    with pytest.raises(ProviderError) as exc:
        provider.complete(
            ProviderRequest(
                input="read",
                model="model",
                attachments=[
                    Attachment(
                        kind="image",
                        source="scan.png",
                        filename="scan.png",
                        mime_type="image/png",
                        payload="data:image/png;base64,abc",
                    )
                ],
            )
        )

    assert "chat completions fallback failed" in str(exc.value)
    assert "chat failed" in str(exc.value)


def test_openai_provider_audio_and_image_methods_use_sdk_surfaces(tmp_path):
    audio = tmp_path / "note.mp3"
    audio.write_bytes(b"audio")
    image = tmp_path / "input.png"
    image.write_bytes(b"png")
    generated = tmp_path / "generated.png"
    edited = tmp_path / "edited.png"
    speech = tmp_path / "speech.mp3"
    calls = []

    class AudioTranscriptions:
        def create(self, **kwargs):
            calls.append(("transcribe", kwargs["model"], kwargs["file"].name))
            return type("Transcript", (), {"text": "hello"})()

    class AudioTranslations:
        def create(self, **kwargs):
            calls.append(("translate", kwargs["model"], kwargs["file"].name))
            return {"text": "hello translated"}

    class SpeechResponse:
        def write_to_file(self, path):
            path.write_bytes(b"mp3")

    class AudioSpeech:
        def create(self, **kwargs):
            calls.append(
                (
                    "speech",
                    kwargs["model"],
                    kwargs["voice"],
                    kwargs["input"],
                    kwargs.get("response_format"),
                )
            )
            return SpeechResponse()

    class Images:
        def generate(self, **kwargs):
            calls.append(("image_generate", kwargs["model"], kwargs["prompt"]))
            return {"data": [{"b64_json": base64.b64encode(b"png").decode("ascii")}]}

        def edit(self, **kwargs):
            calls.append(
                ("image_edit", kwargs["model"], kwargs["prompt"], kwargs["image"].name)
            )
            return {"data": [{"b64_json": base64.b64encode(b"edited").decode("ascii")}]}

    class Client:
        def __init__(self):
            self.audio = type(
                "Audio",
                (),
                {
                    "transcriptions": AudioTranscriptions(),
                    "translations": AudioTranslations(),
                    "speech": AudioSpeech(),
                },
            )()
            self.images = Images()

    provider = OpenAIProvider(
        AuthInfo(provider="openai", api_key="sk-test"),
        client_factory=lambda api_key, base_url=None: Client(),
    )

    assert provider.transcribe_audio(str(audio), model="gpt-4o-transcribe") == "hello"
    assert provider.translate_audio(str(audio), model="whisper-1") == "hello translated"
    assert provider.generate_speech(
        "say hi",
        model="gpt-4o-mini-tts",
        voice="alloy",
        output_path=str(speech),
        response_format="mp3",
    ) == str(speech)
    assert provider.generate_image(
        "draw", model="gpt-image-1", output_path=str(generated)
    ) == str(generated)
    assert provider.edit_image(
        str(image), "edit", model="gpt-image-1", output_path=str(edited)
    ) == str(edited)
    assert speech.read_bytes() == b"mp3"
    assert generated.read_bytes() == b"png"
    assert edited.read_bytes() == b"edited"
    assert calls[0][0] == "transcribe"
    assert calls[1][0] == "translate"
    assert calls[2] == ("speech", "gpt-4o-mini-tts", "alloy", "say hi", "mp3")
    assert calls[3] == ("image_generate", "gpt-image-1", "draw")
    assert calls[4][0] == "image_edit"


def test_openai_provider_wraps_audio_image_and_speech_errors(tmp_path):
    audio = tmp_path / "note.mp3"
    audio.write_bytes(b"audio")
    image = tmp_path / "input.png"
    image.write_bytes(b"png")

    class Failing:
        def create(self, **kwargs):
            raise RuntimeError("sdk failed")

        def edit(self, **kwargs):
            raise RuntimeError("sdk failed")

        def generate(self, **kwargs):
            raise RuntimeError("sdk failed")

    class Client:
        audio = type(
            "Audio",
            (),
            {
                "transcriptions": Failing(),
                "translations": Failing(),
                "speech": Failing(),
            },
        )()
        images = Failing()

    provider = OpenAIProvider(
        AuthInfo(provider="openai", api_key="sk-test"),
        client_factory=lambda api_key, base_url=None: Client(),
    )

    with pytest.raises(ProviderError, match="audio transcription failed"):
        provider.transcribe_audio(str(audio), model="transcribe")
    with pytest.raises(ProviderError, match="audio translation failed"):
        provider.translate_audio(str(audio), model="translate")
    with pytest.raises(ProviderError, match="speech generation failed"):
        provider.generate_speech(
            "hello",
            model="tts",
            voice="alloy",
            output_path=str(tmp_path / "speech.mp3"),
        )
    with pytest.raises(ProviderError, match="image generation failed"):
        provider.generate_images(
            "draw", model="image", output_path=str(tmp_path / "image.png"), count=1
        )
    with pytest.raises(ProviderError, match="image edit failed"):
        provider.edit_image(
            str(image), "edit", model="image", output_path=str(tmp_path / "edited.png")
        )


def test_openai_provider_speech_prefers_streaming_response_helper(tmp_path):
    speech = tmp_path / "speech.mp3"
    calls = []

    class StreamingResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def stream_to_file(self, path):
            calls.append(("stream_to_file", str(path)))
            path.write_bytes(b"mp3")

    class StreamingSpeech:
        def create(self, **kwargs):
            calls.append(
                ("streaming_create", kwargs["model"], kwargs["voice"], kwargs["input"])
            )
            return StreamingResponse()

    class Speech:
        with_streaming_response = StreamingSpeech()

        def create(self, **kwargs):
            raise AssertionError("non-streaming speech.create should not be used")

    class Client:
        def __init__(self):
            self.audio = type("Audio", (), {"speech": Speech()})()

    provider = OpenAIProvider(
        AuthInfo(provider="openai", api_key="sk-test"),
        client_factory=lambda api_key, base_url=None: Client(),
    )

    assert provider.generate_speech(
        "say hi", model="gpt-4o-mini-tts", voice="alloy", output_path=str(speech)
    ) == str(speech)
    assert speech.read_bytes() == b"mp3"
    assert calls == [
        ("streaming_create", "gpt-4o-mini-tts", "alloy", "say hi"),
        ("stream_to_file", str(speech)),
    ]


def test_openai_provider_generate_images_writes_numbered_files(tmp_path):
    output = tmp_path / "image.png"
    calls = []

    class Images:
        def generate(self, **kwargs):
            calls.append(kwargs)
            return {
                "data": [
                    {"b64_json": base64.b64encode(b"one").decode("ascii")},
                    {"b64_json": base64.b64encode(b"two").decode("ascii")},
                ]
            }

    class Client:
        def __init__(self):
            self.images = Images()

    provider = OpenAIProvider(
        AuthInfo(provider="openai", api_key="sk-test"),
        client_factory=lambda api_key, base_url=None: Client(),
    )

    paths = provider.generate_images(
        "draw", model="gpt-image-1", output_path=str(output), count=2
    )

    assert paths == [str(tmp_path / "image-1.png"), str(tmp_path / "image-2.png")]
    assert (tmp_path / "image-1.png").read_bytes() == b"one"
    assert (tmp_path / "image-2.png").read_bytes() == b"two"
    assert calls == [{"model": "gpt-image-1", "prompt": "draw", "n": 2}]


def test_openai_provider_rejects_groq_image_generation_before_sdk_call(tmp_path):
    class Images:
        def generate(self, **kwargs):
            raise AssertionError(
                "Groq image generation should be rejected before SDK call"
            )

    class Client:
        images = Images()

    provider = OpenAIProvider(
        AuthInfo(provider="openai", api_key="gsk-test"),
        client_factory=lambda api_key, base_url=None: Client(),
        base_url="https://api.groq.com/openai/v1",
    )

    with pytest.raises(ProviderError, match="Groq does not support image generation"):
        provider.generate_images(
            "draw",
            model="gpt-image-1",
            output_path=str(tmp_path / "image.png"),
            count=1,
        )


def test_openai_provider_rejects_groq_image_edit_before_sdk_call(tmp_path):
    image = tmp_path / "input.png"
    image.write_bytes(b"png")

    class Images:
        def edit(self, **kwargs):
            raise AssertionError(
                "Groq image editing should be rejected before SDK call"
            )

    class Client:
        images = Images()

    provider = OpenAIProvider(
        AuthInfo(provider="openai", api_key="gsk-test"),
        client_factory=lambda api_key, base_url=None: Client(),
        base_url="https://api.groq.com/openai/v1",
    )

    with pytest.raises(ProviderError, match="Groq does not support image editing"):
        provider.edit_image(
            str(image),
            "edit",
            model="gpt-image-1",
            output_path=str(tmp_path / "edited.png"),
        )


def test_openai_provider_generate_video_polls_and_downloads(tmp_path):
    output = tmp_path / "video.mp4"
    calls = []

    class VideoJob:
        def __init__(self, status, progress=0):
            self.id = "video_1"
            self.status = status
            self.progress = progress

    class Content:
        def write_to_file(self, path):
            calls.append(("write", str(path)))
            path.write_bytes(b"mp4")

    class Videos:
        def __init__(self):
            self.statuses = [VideoJob("in_progress", 50), VideoJob("completed", 100)]

        def create(self, **kwargs):
            calls.append(("create", kwargs))
            return VideoJob("queued", 0)

        def retrieve(self, video_id):
            calls.append(("retrieve", video_id))
            return self.statuses.pop(0)

        def download_content(self, video_id, **kwargs):
            calls.append(("download", video_id, kwargs))
            return Content()

    class Client:
        def __init__(self):
            self.videos = Videos()

    provider = OpenAIProvider(
        AuthInfo(provider="openai", api_key="sk-test"),
        client_factory=lambda api_key, base_url=None: Client(),
        sleep_func=lambda seconds: calls.append(("sleep", seconds)),
    )

    written = provider.generate_video(
        "make a demo",
        model="sora-2",
        output_path=str(output),
        seconds="8",
        size="1280x720",
        poll_interval=0.25,
        timeout=30,
    )

    assert written == str(output)
    assert output.read_bytes() == b"mp4"
    assert calls[0] == (
        "create",
        {
            "model": "sora-2",
            "prompt": "make a demo",
            "seconds": "8",
            "size": "1280x720",
        },
    )
    assert ("retrieve", "video_1") in calls
    assert ("sleep", 0.25) in calls
    assert calls[-2] == ("download", "video_1", {"variant": "video"})
    assert calls[-1] == ("write", str(output))


def test_openai_provider_generate_video_requires_model():
    provider = OpenAIProvider(
        AuthInfo(provider="openai", api_key="sk-test"),
        client_factory=lambda api_key, base_url=None: object(),
    )

    with pytest.raises(ProviderError, match="requires a model"):
        provider.generate_video("make a demo", model=None, output_path="video.mp4")


def test_openai_provider_generate_video_passes_input_reference(tmp_path):
    output = tmp_path / "video.mp4"
    image = tmp_path / "frame.png"
    image.write_bytes(b"png")
    seen = []

    class VideoJob:
        id = "video_1"
        status = "completed"

    class Content:
        def read(self):
            return b"mp4"

    class Videos:
        def create(self, **kwargs):
            seen.append(kwargs)
            assert kwargs["input_reference"].name == str(image)
            return VideoJob()

        def download_content(self, video_id, **kwargs):
            return Content()

    class Client:
        def __init__(self):
            self.videos = Videos()

    provider = OpenAIProvider(
        AuthInfo(provider="openai", api_key="sk-test"),
        client_factory=lambda api_key, base_url=None: Client(),
        sleep_func=lambda seconds: None,
    )

    provider.generate_video(
        "make a demo",
        model="sora-2-pro",
        output_path=str(output),
        image_path=str(image),
        poll_interval=0,
        timeout=30,
    )

    assert output.read_bytes() == b"mp4"
    assert seen[0]["model"] == "sora-2-pro"


def test_openai_provider_generate_video_failed_job_raises():
    class VideoJob:
        id = "video_1"
        status = "failed"
        error = type("Error", (), {"message": "render failed"})()

    class Videos:
        def create(self, **kwargs):
            return VideoJob()

    class Client:
        def __init__(self):
            self.videos = Videos()

    provider = OpenAIProvider(
        AuthInfo(provider="openai", api_key="sk-test"),
        client_factory=lambda api_key, base_url=None: Client(),
        sleep_func=lambda seconds: None,
    )

    with pytest.raises(ProviderError, match="render failed"):
        provider.generate_video(
            "make a demo",
            model="sora-2",
            output_path="video.mp4",
            poll_interval=0,
            timeout=30,
        )


def test_openai_provider_generate_video_failed_job_without_message_raises_default():
    class VideoJob:
        id = "video_1"
        status = "failed"
        error = object()

    class Videos:
        def create(self, **kwargs):
            return VideoJob()

    class Client:
        videos = Videos()

    provider = OpenAIProvider(
        AuthInfo(provider="openai", api_key="sk-test"),
        client_factory=lambda api_key, base_url=None: Client(),
        sleep_func=lambda seconds: None,
    )

    with pytest.raises(ProviderError, match="Video generation failed"):
        provider.generate_video(
            "make a demo",
            model="sora-2",
            output_path="video.mp4",
            poll_interval=0,
            timeout=30,
        )


def test_openai_provider_generate_video_unexpected_status_raises():
    class VideoJob:
        id = "video_1"
        status = "cancelled"

    class Videos:
        def create(self, **kwargs):
            return VideoJob()

    class Client:
        videos = Videos()

    provider = OpenAIProvider(
        AuthInfo(provider="openai", api_key="sk-test"),
        client_factory=lambda api_key, base_url=None: Client(),
        sleep_func=lambda seconds: None,
    )

    with pytest.raises(ProviderError, match="unexpected status"):
        provider.generate_video(
            "make a demo",
            model="sora-2",
            output_path="video.mp4",
            poll_interval=0,
            timeout=30,
        )


def test_openai_provider_generate_video_wraps_sdk_errors():
    class Videos:
        def create(self, **kwargs):
            raise RuntimeError("video backend down")

    class Client:
        videos = Videos()

    provider = OpenAIProvider(
        AuthInfo(provider="openai", api_key="sk-test"),
        client_factory=lambda api_key, base_url=None: Client(),
        sleep_func=lambda seconds: None,
    )

    with pytest.raises(ProviderError, match="video backend down"):
        provider.generate_video(
            "make a demo",
            model="sora-2",
            output_path="video.mp4",
            poll_interval=0,
            timeout=30,
        )


def test_openai_provider_generate_video_timeout_raises():
    class VideoJob:
        id = "video_1"
        status = "queued"

    class Videos:
        def create(self, **kwargs):
            return VideoJob()

        def retrieve(self, video_id):
            return VideoJob()

    class Client:
        def __init__(self):
            self.videos = Videos()

    provider = OpenAIProvider(
        AuthInfo(provider="openai", api_key="sk-test"),
        client_factory=lambda api_key, base_url=None: Client(),
        sleep_func=lambda seconds: None,
        monotonic_func=iter([0, 10, 20]).__next__,
    )

    with pytest.raises(ProviderError, match="timed out"):
        provider.generate_video(
            "make a demo",
            model="sora-2",
            output_path="video.mp4",
            poll_interval=0,
            timeout=5,
        )


class FakeStreamEvent:
    def __init__(self, event_type, delta=None, response_id=None):
        self.type = event_type
        self.delta = delta
        self.response = type("Resp", (), {"id": response_id})() if response_id else None


def test_openai_provider_streaming(monkeypatch):
    class StreamingResponses(FakeResponses):
        def create(self, **kwargs):
            self.calls.append(kwargs)
            return [
                FakeStreamEvent("response.output_text.delta", "he"),
                FakeStreamEvent("response.output_text.delta", "llo"),
                FakeStreamEvent("response.completed", response_id="resp_stream"),
            ]

    class StreamingOpenAI(FakeOpenAI):
        def __init__(self, api_key):
            self.api_key = api_key
            self.responses = StreamingResponses()

    chunks = []
    provider = OpenAIProvider(
        AuthInfo(provider="openai", api_key="sk-test"),
        client_factory=lambda api_key, base_url=None: StreamingOpenAI(api_key),
    )

    result = provider.complete(
        ProviderRequest(input="hi", model="gpt-test", stream=True),
        on_delta=chunks.append,
    )

    assert chunks == ["he", "llo"]
    assert result.text == "hello"
    assert result.response_id == "resp_stream"


def test_openai_provider_base_url_vision_falls_back_to_chat_completions():
    created = {}

    class Responses:
        def create(self, **kwargs):
            raise RuntimeError("responses endpoint unsupported")

    class ChatCompletions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            message = type("Message", (), {"content": "seen"})()
            choice = type("Choice", (), {"message": message})()
            return type("ChatResponse", (), {"id": "chat_1", "choices": [choice]})()

    class Chat:
        def __init__(self):
            self.completions = ChatCompletions()

    class Client:
        def __init__(self):
            self.responses = Responses()
            self.chat = Chat()

    def factory(api_key, base_url=None):
        created["base_url"] = base_url
        created["client"] = Client()
        return created["client"]

    provider = OpenAIProvider(
        AuthInfo(provider="openai", api_key="gsk-test"),
        client_factory=factory,
        base_url="https://api.groq.com/openai/v1",
    )

    result = provider.complete(
        ProviderRequest(
            input="read this",
            instructions="be concise",
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            attachments=[
                Attachment(
                    kind="image",
                    source="scan.png",
                    filename="scan.png",
                    mime_type="image/png",
                    payload="data:image/png;base64,abc",
                    detail="high",
                )
            ],
        )
    )

    assert result.text == "seen"
    assert result.response_id == "chat_1"
    call = created["client"].chat.completions.calls[0]
    assert call["model"] == "meta-llama/llama-4-scout-17b-16e-instruct"
    assert call["messages"][0] == {"role": "system", "content": "be concise"}
    assert call["messages"][1]["content"] == [
        {"type": "text", "text": "read this"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,abc", "detail": "high"},
        },
    ]


def test_openai_provider_base_url_rasterizes_local_pdf_for_chat_fallback(
    monkeypatch, tmp_path
):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF")
    created = {}

    class FakePixmap:
        def tobytes(self, format_name):
            assert format_name == "png"
            return b"png-page"

    class FakePage:
        def get_pixmap(self, matrix):
            assert matrix.zoom == (2, 2)
            return FakePixmap()

    class FakeDocument:
        page_count = 1

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def __getitem__(self, index):
            assert index == 0
            return FakePage()

    fake_fitz = types.SimpleNamespace(
        Matrix=lambda x, y: types.SimpleNamespace(zoom=(x, y)),
        open=lambda path: FakeDocument(),
    )
    monkeypatch.setitem(sys.modules, "fitz", fake_fitz)

    class Responses:
        def create(self, **kwargs):
            raise RuntimeError("responses endpoint unsupported")

    class ChatCompletions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            message = type("Message", (), {"content": "action items"})()
            choice = type("Choice", (), {"message": message})()
            return type("ChatResponse", (), {"id": "chat_pdf", "choices": [choice]})()

    class Chat:
        def __init__(self):
            self.completions = ChatCompletions()

    class Client:
        def __init__(self):
            self.responses = Responses()
            self.chat = Chat()

    def factory(api_key, base_url=None):
        created["client"] = Client()
        return created["client"]

    provider = OpenAIProvider(
        AuthInfo(provider="openai", api_key="gsk-test"),
        client_factory=factory,
        base_url="https://api.groq.com/openai/v1",
    )

    result = provider.complete(
        ProviderRequest(
            input="Summarize the action items",
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            attachments=[
                Attachment(
                    kind="file",
                    source=str(pdf),
                    filename="report.pdf",
                    mime_type="application/pdf",
                    payload="data:application/pdf;base64,abc",
                )
            ],
        )
    )

    assert result.text == "action items"
    call = created["client"].chat.completions.calls[0]
    assert call["messages"][0]["content"] == [
        {"type": "text", "text": "Summarize the action items"},
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64,cG5nLXBhZ2U=",
                "detail": "high",
            },
        },
    ]


def test_openai_provider_base_url_does_not_chat_fallback_for_remote_files():
    class Responses:
        def create(self, **kwargs):
            raise RuntimeError("responses endpoint unsupported")

    class Client:
        responses = Responses()

    provider = OpenAIProvider(
        AuthInfo(provider="openai", api_key="gsk-test"),
        client_factory=lambda api_key, base_url=None: Client(),
        base_url="https://api.groq.com/openai/v1",
    )

    with pytest.raises(ProviderError, match="Responses API"):
        provider.complete(
            ProviderRequest(
                input="read this",
                model="model",
                attachments=[
                    Attachment(
                        kind="file",
                        source="https://example.test/report.pdf",
                        filename="report.pdf",
                        mime_type="application/pdf",
                    )
                ],
            )
        )


def test_openai_provider_base_url_does_not_fallback_for_auth_error_with_image():
    class Responses:
        def create(self, **kwargs):
            raise RuntimeError("401 unauthorized invalid api key")

    class ChatCompletions:
        def create(self, **kwargs):
            raise AssertionError("auth failures should not use chat fallback")

    class Client:
        responses = Responses()
        chat = type("Chat", (), {"completions": ChatCompletions()})()

    provider = OpenAIProvider(
        AuthInfo(provider="openai", api_key="bad-key"),
        client_factory=lambda api_key, base_url=None: Client(),
        base_url="https://api.groq.com/openai/v1",
    )

    with pytest.raises(ProviderError, match="OpenAI provider request failed"):
        provider.complete(
            ProviderRequest(
                input="read this",
                model="model",
                attachments=[
                    Attachment(
                        kind="image",
                        source="scan.png",
                        filename="scan.png",
                        mime_type="image/png",
                        payload="data:image/png;base64,abc",
                    )
                ],
            )
        )


def test_openai_provider_base_url_fallback_on_supported_http_status_with_image():
    class ResponseError(Exception):
        status_code = 404

    class Responses:
        def create(self, **kwargs):
            raise ResponseError("not found")

    class ChatCompletions:
        def create(self, **kwargs):
            message = type("Message", (), {"content": "seen"})()
            choice = type("Choice", (), {"message": message})()
            return type("ChatResponse", (), {"id": "chat_404", "choices": [choice]})()

    class Client:
        responses = Responses()
        chat = type("Chat", (), {"completions": ChatCompletions()})()

    provider = OpenAIProvider(
        AuthInfo(provider="openai", api_key="provider-key"),
        client_factory=lambda api_key, base_url=None: Client(),
        base_url="https://provider.test/v1",
    )

    result = provider.complete(
        ProviderRequest(
            input="read this",
            model="model",
            attachments=[
                Attachment(
                    kind="image",
                    source="scan.png",
                    filename="scan.png",
                    mime_type="image/png",
                    payload="data:image/png;base64,abc",
                )
            ],
        )
    )

    assert result.text == "seen"
    assert result.response_id == "chat_404"


def test_codex_oauth_provider_posts_to_backend(monkeypatch):
    class FakeHttp:
        def __init__(self):
            self.posts = []

        def post(self, url, **kwargs):
            self.posts.append((url, kwargs))
            return type(
                "Resp",
                (),
                {
                    "status_code": 200,
                    "raise_for_status": lambda self: None,
                    "json": lambda self: {"id": "resp_codex", "output_text": "ok"},
                },
            )()

    http = FakeHttp()
    auth = AuthInfo(
        provider="codex",
        oauth=OAuthTokens(access_token="access", refresh_token="refresh"),
    )
    provider = CodexOAuthProvider(auth, http_client=http)

    result = provider.complete(ProviderRequest(input="hi", model="gpt-test"))

    assert result.text == "ok"
    assert result.response_id == "resp_codex"
    assert http.posts[0][1]["headers"]["Authorization"] == "Bearer access"
    assert http.posts[0][1]["headers"]["Accept"] == "text/event-stream"


def test_codex_oauth_provider_requires_oauth_tokens():
    with pytest.raises(ValueError, match="requires OAuth tokens"):
        CodexOAuthProvider(AuthInfo(provider="codex"))


def test_codex_oauth_provider_uses_env_backend_urls(monkeypatch):
    monkeypatch.setenv("LLMRUN_CODEX_RESPONSES_URL", "https://codex.test/responses")
    monkeypatch.setenv("LLMRUN_OPENAI_REFRESH_URL", "https://codex.test/refresh")

    provider = CodexOAuthProvider(
        AuthInfo(provider="codex", oauth=OAuthTokens(access_token="access")),
        http_client=object(),
    )

    assert provider.responses_url == "https://codex.test/responses"
    assert provider.refresh_url == "https://codex.test/refresh"


def test_codex_oauth_provider_sends_account_id_header_when_available():
    class FakeHttp:
        def __init__(self):
            self.posts = []

        def post(self, url, **kwargs):
            self.posts.append((url, kwargs))
            return type(
                "Resp",
                (),
                {
                    "status_code": 200,
                    "raise_for_status": lambda self: None,
                    "json": lambda self: {"id": "resp_codex", "output_text": "ok"},
                },
            )()

    auth = AuthInfo(
        provider="codex",
        oauth=OAuthTokens(
            access_token="access", refresh_token="refresh", account_id="acct_123"
        ),
    )
    provider = CodexOAuthProvider(auth, http_client=FakeHttp())

    provider.complete(ProviderRequest(input="hi", model="gpt-test"))

    assert provider.http.posts[0][1]["headers"]["ChatGPT-Account-ID"] == "acct_123"


def test_codex_oauth_provider_supplies_default_instructions():
    class FakeHttp:
        def __init__(self):
            self.posts = []

        def post(self, url, **kwargs):
            self.posts.append((url, kwargs))
            return type(
                "Resp",
                (),
                {
                    "status_code": 200,
                    "raise_for_status": lambda self: None,
                    "json": lambda self: {"id": "resp_codex", "output_text": "ok"},
                },
            )()

    auth = AuthInfo(
        provider="codex",
        oauth=OAuthTokens(access_token="access", refresh_token="refresh"),
    )
    provider = CodexOAuthProvider(auth, http_client=FakeHttp())

    provider.complete(ProviderRequest(input="hi", model="gpt-test"))

    payload = provider.http.posts[0][1]["json"]
    assert isinstance(payload["instructions"], str)
    assert payload["instructions"]


def test_codex_oauth_provider_sends_structured_input_list():
    class FakeHttp:
        def __init__(self):
            self.posts = []

        def post(self, url, **kwargs):
            self.posts.append((url, kwargs))
            return type(
                "Resp",
                (),
                {
                    "status_code": 200,
                    "raise_for_status": lambda self: None,
                    "json": lambda self: {"id": "resp_codex", "output_text": "ok"},
                },
            )()

    auth = AuthInfo(
        provider="codex",
        oauth=OAuthTokens(access_token="access", refresh_token="refresh"),
    )
    provider = CodexOAuthProvider(auth, http_client=FakeHttp())

    provider.complete(ProviderRequest(input="hi", model="gpt-test"))

    payload = provider.http.posts[0][1]["json"]
    assert payload["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hi"}],
        }
    ]


def test_codex_oauth_provider_sends_multimodal_content_items():
    class FakeHttp:
        def __init__(self):
            self.posts = []

        def post(self, url, **kwargs):
            self.posts.append((url, kwargs))
            return type(
                "Resp",
                (),
                {
                    "status_code": 200,
                    "raise_for_status": lambda self: None,
                    "json": lambda self: {"id": "resp_codex", "output_text": "ok"},
                },
            )()

    auth = AuthInfo(
        provider="codex",
        oauth=OAuthTokens(access_token="access", refresh_token="refresh"),
    )
    provider = CodexOAuthProvider(auth, http_client=FakeHttp())

    provider.complete(
        ProviderRequest(
            input="describe",
            model="gpt-test",
            attachments=[
                Attachment(
                    kind="image",
                    source="screen.png",
                    filename="screen.png",
                    mime_type="image/png",
                    payload="data:image/png;base64,abc",
                    detail="low",
                ),
                Attachment(
                    kind="file",
                    source="report.pdf",
                    filename="report.pdf",
                    mime_type="application/pdf",
                    payload="data:application/pdf;base64,def",
                ),
            ],
        )
    )

    payload = provider.http.posts[0][1]["json"]
    assert payload["input"][0]["content"] == [
        {"type": "input_text", "text": "describe"},
        {
            "type": "input_image",
            "image_url": "data:image/png;base64,abc",
            "detail": "low",
        },
        {
            "type": "input_file",
            "filename": "report.pdf",
            "file_data": "data:application/pdf;base64,def",
        },
    ]


def test_codex_oauth_provider_omits_previous_response_id():
    class FakeHttp:
        def __init__(self):
            self.posts = []

        def post(self, url, **kwargs):
            self.posts.append((url, kwargs))
            return type(
                "Resp",
                (),
                {
                    "status_code": 200,
                    "raise_for_status": lambda self: None,
                    "json": lambda self: {"id": "resp_codex", "output_text": "ok"},
                },
            )()

    auth = AuthInfo(
        provider="codex",
        oauth=OAuthTokens(access_token="access", refresh_token="refresh"),
    )
    provider = CodexOAuthProvider(auth, http_client=FakeHttp())

    provider.complete(
        ProviderRequest(input="hi", model="gpt-test", previous_response_id="resp_prev")
    )

    payload = provider.http.posts[0][1]["json"]
    assert "previous_response_id" not in payload


def test_codex_oauth_provider_disables_server_side_store():
    class FakeHttp:
        def __init__(self):
            self.posts = []

        def post(self, url, **kwargs):
            self.posts.append((url, kwargs))
            return type(
                "Resp",
                (),
                {
                    "status_code": 200,
                    "raise_for_status": lambda self: None,
                    "json": lambda self: {"id": "resp_codex", "output_text": "ok"},
                },
            )()

    auth = AuthInfo(
        provider="codex",
        oauth=OAuthTokens(access_token="access", refresh_token="refresh"),
    )
    provider = CodexOAuthProvider(auth, http_client=FakeHttp())

    provider.complete(ProviderRequest(input="hi", model="gpt-test"))

    payload = provider.http.posts[0][1]["json"]
    assert payload["store"] is False


def test_codex_oauth_provider_forces_upstream_stream_true_and_strips_rejected_fields():
    class FakeHttp:
        def __init__(self):
            self.posts = []

        def post(self, url, **kwargs):
            self.posts.append((url, kwargs))
            return type(
                "Resp",
                (),
                {
                    "status_code": 200,
                    "headers": {"content-type": "application/json"},
                    "text": '{"id":"resp_codex","output_text":"ok"}',
                    "raise_for_status": lambda self: None,
                    "json": lambda self: {"id": "resp_codex", "output_text": "ok"},
                },
            )()

    auth = AuthInfo(
        provider="codex",
        oauth=OAuthTokens(access_token="access", refresh_token="refresh"),
    )
    provider = CodexOAuthProvider(auth, http_client=FakeHttp())

    provider.complete(
        ProviderRequest(
            input="hi", model="gpt-test", max_output_tokens=20, stream=False
        )
    )

    payload = provider.http.posts[0][1]["json"]
    assert payload["stream"] is True
    assert "max_output_tokens" not in payload


def test_codex_oauth_provider_collects_sse_and_emits_deltas():
    class FakeHttp:
        def post(self, url, **kwargs):
            return type(
                "Resp",
                (),
                {
                    "status_code": 200,
                    "headers": {"content-type": "text/event-stream"},
                    "text": "\n".join(
                        [
                            'data: {"type":"response.output_text.delta","delta":"he"}',
                            "",
                            'data: {"type":"response.output_text.delta","delta":"llo"}',
                            "",
                            'data: {"type":"response.completed","response":{"id":"resp_stream"}}',
                            "",
                            "data: [DONE]",
                            "",
                        ]
                    ),
                    "raise_for_status": lambda self: None,
                    "json": lambda self: {},
                },
            )()

    auth = AuthInfo(
        provider="codex",
        oauth=OAuthTokens(access_token="access", refresh_token="refresh"),
    )
    provider = CodexOAuthProvider(auth, http_client=FakeHttp())
    chunks = []

    result = provider.complete(
        ProviderRequest(input="hi", model="gpt-test", stream=True),
        on_delta=chunks.append,
    )

    assert chunks == ["he", "llo"]
    assert result.text == "hello"
    assert result.response_id == "resp_stream"


def test_codex_oauth_provider_collects_sse_without_event_stream_content_type():
    class FakeHttp:
        def post(self, url, **kwargs):
            return type(
                "Resp",
                (),
                {
                    "status_code": 200,
                    "headers": {"content-type": "text/plain; charset=utf-8"},
                    "text": "\n".join(
                        [
                            "event: response.output_text.delta",
                            'data: {"type":"response.output_text.delta","delta":"ok"}',
                            "",
                            "event: response.completed",
                            'data: {"type":"response.completed","response":{"id":"resp_stream"}}',
                            "",
                        ]
                    ),
                    "raise_for_status": lambda self: None,
                    "json": lambda self: (_ for _ in ()).throw(ValueError("not json")),
                },
            )()

    auth = AuthInfo(
        provider="codex",
        oauth=OAuthTokens(access_token="access", refresh_token="refresh"),
    )
    provider = CodexOAuthProvider(auth, http_client=FakeHttp())

    result = provider.complete(ProviderRequest(input="hi", model="gpt-test"))

    assert result.text == "ok"
    assert result.response_id == "resp_stream"


def test_codex_oauth_provider_translates_unparseable_success_body():
    class FakeHttp:
        def post(self, url, **kwargs):
            return type(
                "Resp",
                (),
                {
                    "status_code": 200,
                    "headers": {"content-type": "text/plain"},
                    "text": "",
                    "raise_for_status": lambda self: None,
                    "json": lambda self: (_ for _ in ()).throw(ValueError("not json")),
                },
            )()

    auth = AuthInfo(
        provider="codex",
        oauth=OAuthTokens(access_token="access", refresh_token="refresh"),
    )
    provider = CodexOAuthProvider(auth, http_client=FakeHttp())

    with pytest.raises(ProviderError, match="empty or unparseable"):
        provider.complete(ProviderRequest(input="hi", model="gpt-test"))


def test_codex_oauth_provider_translates_http_error_without_response():
    class FakeHttp:
        def post(self, url, **kwargs):
            raise httpx.ConnectError("network down")

    provider = CodexOAuthProvider(
        AuthInfo(provider="codex", oauth=OAuthTokens(access_token="access")),
        http_client=FakeHttp(),
    )

    with pytest.raises(ProviderError, match="network down"):
        provider.complete(ProviderRequest(input="hi", model="gpt-test"))


def test_codex_oauth_provider_translates_http_400():
    class FakeHttp:
        def post(self, url, **kwargs):
            request = httpx.Request("POST", url)
            response = httpx.Response(
                400,
                json={
                    "detail": "The model is not supported when using Codex with a ChatGPT account."
                },
                request=request,
            )
            return response

    auth = AuthInfo(
        provider="codex",
        oauth=OAuthTokens(access_token="access", refresh_token="refresh"),
    )
    provider = CodexOAuthProvider(auth, http_client=FakeHttp())

    with pytest.raises(ProviderError) as exc:
        provider.complete(ProviderRequest(input="hi", model="gpt-test"))

    message = str(exc.value)
    assert "HTTP 400" in message
    assert "model is not supported" in message


def test_codex_oauth_provider_refreshes_expired_token():
    expires_at = (
        (datetime.now(UTC) - timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    )

    class FakeHttp:
        def __init__(self):
            self.posts = []

        def post(self, url, **kwargs):
            self.posts.append((url, kwargs))
            if url == "https://refresh.test/token":
                return type(
                    "Resp",
                    (),
                    {
                        "raise_for_status": lambda self: None,
                        "json": lambda self: {
                            "access_token": "new-access",
                            "refresh_token": "new-refresh",
                            "expires_at": "2099-01-01T00:00:00Z",
                        },
                    },
                )()
            return type(
                "Resp",
                (),
                {
                    "status_code": 200,
                    "headers": {"content-type": "application/json"},
                    "text": '{"id":"resp_codex","output_text":"ok"}',
                    "raise_for_status": lambda self: None,
                    "json": lambda self: {"id": "resp_codex", "output_text": "ok"},
                },
            )()

    provider = CodexOAuthProvider(
        AuthInfo(
            provider="codex",
            oauth=OAuthTokens(
                access_token="old-access",
                refresh_token="refresh",
                expires_at=expires_at,
            ),
        ),
        http_client=FakeHttp(),
        refresh_url="https://refresh.test/token",
    )

    result = provider.complete(ProviderRequest(input="hi", model="gpt-test"))

    assert result.text == "ok"
    assert provider.tokens.access_token == "new-access"
    assert provider.http.posts[0][0] == "https://refresh.test/token"
    assert provider.http.posts[1][1]["headers"]["Authorization"] == "Bearer new-access"


def test_codex_oauth_provider_refresh_failure_is_wrapped():
    expires_at = (
        (datetime.now(UTC) - timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    )

    class FakeHttp:
        def post(self, url, **kwargs):
            raise httpx.ConnectError("refresh down")

    provider = CodexOAuthProvider(
        AuthInfo(
            provider="codex",
            oauth=OAuthTokens(
                access_token="old-access",
                refresh_token="refresh",
                expires_at=expires_at,
            ),
        ),
        http_client=FakeHttp(),
    )

    with pytest.raises(ProviderError, match="Re-run Codex login"):
        provider.complete(ProviderRequest(input="hi", model="gpt-test"))


def test_codex_oauth_provider_refresh_missing_access_token_is_wrapped():
    expires_at = (
        (datetime.now(UTC) - timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    )

    class FakeHttp:
        def post(self, url, **kwargs):
            return type(
                "Resp",
                (),
                {"raise_for_status": lambda self: None, "json": lambda self: {}},
            )()

    provider = CodexOAuthProvider(
        AuthInfo(
            provider="codex",
            oauth=OAuthTokens(
                access_token="old-access",
                refresh_token="refresh",
                expires_at=expires_at,
            ),
        ),
        http_client=FakeHttp(),
    )

    with pytest.raises(ProviderError, match="Re-run Codex login"):
        provider.complete(ProviderRequest(input="hi", model="gpt-test"))


def test_codex_oauth_provider_audio_and_video_methods_report_unsupported(tmp_path):
    provider = CodexOAuthProvider(
        AuthInfo(provider="codex", oauth=OAuthTokens(access_token="access")),
        http_client=object(),
    )

    with pytest.raises(ProviderError, match="audio workflows"):
        provider.transcribe_audio("note.mp3", model="transcribe")
    with pytest.raises(ProviderError, match="audio workflows"):
        provider.translate_audio("note.mp3", model="translate")
    with pytest.raises(ProviderError, match="audio workflows"):
        provider.generate_speech(
            "hello",
            model="tts",
            voice="alloy",
            output_path=str(tmp_path / "speech.mp3"),
        )
    with pytest.raises(ProviderError, match="video generation"):
        provider.generate_video(
            "make a demo", model="sora-2", output_path=str(tmp_path / "video.mp4")
        )


def test_codex_oauth_provider_generate_images_uses_responses_image_tool(tmp_path):
    output = tmp_path / "image.png"
    image_b64 = base64.b64encode(b"png").decode("ascii")

    class FakeHttp:
        def __init__(self):
            self.posts = []

        def post(self, url, **kwargs):
            self.posts.append((url, kwargs))
            return type(
                "Resp",
                (),
                {
                    "status_code": 200,
                    "headers": {"content-type": "text/event-stream"},
                    "text": "\n".join(
                        [
                            'data: {"type":"response.output_item.done","item":{"type":"image_generation_call","result":"'
                            + image_b64
                            + '"}}',
                            "",
                            "data: [DONE]",
                            "",
                        ]
                    ),
                    "raise_for_status": lambda self: None,
                },
            )()

    auth = AuthInfo(
        provider="codex",
        oauth=OAuthTokens(access_token="access", refresh_token="refresh"),
    )
    provider = CodexOAuthProvider(auth, http_client=FakeHttp())

    paths = provider.generate_images(
        "draw", model="gpt-image-1", output_path=str(output), count=1
    )

    assert paths == [str(output)]
    assert output.read_bytes() == b"png"
    payload = provider.http.posts[0][1]["json"]
    assert payload["model"] == "gpt-5.4-mini"
    assert payload["tools"] == [{"type": "image_generation"}]
    assert payload["tool_choice"] == {"type": "image_generation"}
    assert payload["input"][0]["content"][0]["text"] == "draw"


def test_codex_oauth_provider_generate_images_writes_numbered_outputs(tmp_path):
    output = tmp_path / "image.png"
    images = [
        base64.b64encode(b"one").decode("ascii"),
        base64.b64encode(b"two").decode("ascii"),
    ]

    class FakeHttp:
        def __init__(self):
            self.posts = []

        def post(self, url, **kwargs):
            image_b64 = images[len(self.posts)]
            self.posts.append((url, kwargs))
            return type(
                "Resp",
                (),
                {
                    "status_code": 200,
                    "headers": {"content-type": "text/event-stream"},
                    "text": f'data: {{"type":"response.output_item.done","item":{{"type":"image_generation_call","result":"{image_b64}"}}}}\n\n',
                    "raise_for_status": lambda self: None,
                },
            )()

    auth = AuthInfo(
        provider="codex",
        oauth=OAuthTokens(access_token="access", refresh_token="refresh"),
    )
    provider = CodexOAuthProvider(auth, http_client=FakeHttp())

    paths = provider.generate_images(
        "draw", model="gpt-5.4", output_path=str(output), count=2
    )

    assert paths == [str(tmp_path / "image-1.png"), str(tmp_path / "image-2.png")]
    assert (tmp_path / "image-1.png").read_bytes() == b"one"
    assert (tmp_path / "image-2.png").read_bytes() == b"two"
    assert [call[1]["json"]["model"] for call in provider.http.posts] == [
        "gpt-5.4",
        "gpt-5.4",
    ]


def test_codex_oauth_provider_generate_images_raises_when_no_image_result():
    class FakeHttp:
        def post(self, url, **kwargs):
            return type(
                "Resp",
                (),
                {
                    "status_code": 200,
                    "headers": {"content-type": "text/event-stream"},
                    "text": 'data: {"type":"response.completed","response":{"id":"resp_1"}}\n\n',
                    "raise_for_status": lambda self: None,
                },
            )()

    auth = AuthInfo(
        provider="codex",
        oauth=OAuthTokens(access_token="access", refresh_token="refresh"),
    )
    provider = CodexOAuthProvider(auth, http_client=FakeHttp())

    with pytest.raises(ProviderError, match="returned no image"):
        provider.generate_images(
            "draw", model="gpt-image-1", output_path="image.png", count=1
        )


def test_codex_oauth_provider_generate_image_returns_single_path(tmp_path):
    output = tmp_path / "image.png"
    image_b64 = base64.b64encode(b"png").decode("ascii")

    class FakeHttp:
        def post(self, url, **kwargs):
            return type(
                "Resp",
                (),
                {
                    "status_code": 200,
                    "headers": {"content-type": "text/event-stream"},
                    "text": f'data: {{"type":"image_generation_call","result":"{image_b64}"}}\n\n',
                    "raise_for_status": lambda self: None,
                },
            )()

    provider = CodexOAuthProvider(
        AuthInfo(provider="codex", oauth=OAuthTokens(access_token="access")),
        http_client=FakeHttp(),
    )

    assert provider.generate_image(
        "draw", model="gpt-5.4", output_path=str(output)
    ) == str(output)
    assert output.read_bytes() == b"png"


def test_codex_oauth_provider_generate_image_http_errors_are_wrapped():
    class StatusHttp:
        def post(self, url, **kwargs):
            request = httpx.Request("POST", url)
            return httpx.Response(500, text="backend failed", request=request)

    provider = CodexOAuthProvider(
        AuthInfo(provider="codex", oauth=OAuthTokens(access_token="access")),
        http_client=StatusHttp(),
    )

    with pytest.raises(ProviderError, match="HTTP 500"):
        provider.generate_images(
            "draw", model="gpt-5.4", output_path="image.png", count=1
        )

    class NetworkHttp:
        def post(self, url, **kwargs):
            raise httpx.ConnectError("network down")

    provider = CodexOAuthProvider(
        AuthInfo(provider="codex", oauth=OAuthTokens(access_token="access")),
        http_client=NetworkHttp(),
    )

    with pytest.raises(ProviderError, match="network down"):
        provider.generate_images(
            "draw", model="gpt-5.4", output_path="image.png", count=1
        )


def test_codex_oauth_provider_edit_image_sends_input_image(tmp_path):
    image = tmp_path / "input.png"
    image.write_bytes(b"source")
    output = tmp_path / "edited.png"
    image_b64 = base64.b64encode(b"edited").decode("ascii")

    class FakeHttp:
        def __init__(self):
            self.posts = []

        def post(self, url, **kwargs):
            self.posts.append((url, kwargs))
            return type(
                "Resp",
                (),
                {
                    "status_code": 200,
                    "headers": {"content-type": "text/event-stream"},
                    "text": f'data: {{"type":"response.output_item.done","item":{{"type":"image_generation_call","result":"{image_b64}"}}}}\n\n',
                    "raise_for_status": lambda self: None,
                },
            )()

    auth = AuthInfo(
        provider="codex",
        oauth=OAuthTokens(access_token="access", refresh_token="refresh"),
    )
    provider = CodexOAuthProvider(auth, http_client=FakeHttp())

    written = provider.edit_image(
        str(image), "make it blue", model="gpt-image-1", output_path=str(output)
    )

    assert written == str(output)
    assert output.read_bytes() == b"edited"
    content = provider.http.posts[0][1]["json"]["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "make it blue"}
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/png;base64,")


def test_provider_response_helpers_cover_fallback_shapes(tmp_path, monkeypatch):
    assert (
        providers._response_text({"output": [{"content": [{"text": "hello"}]}]})
        == "hello"
    )
    assert providers._response_text(object()).startswith("<object object")
    assert (
        providers._chat_response_text({"choices": [{"message": {"content": "chat"}}]})
        == "chat"
    )
    assert providers._chat_response_text({"choices": [{}]}).startswith("{'choices'")
    assert providers._audio_text(object()).startswith("<object object")
    assert (
        providers._dict_response_text(
            {"output": ["ignored", {"content": ["ignored", {"text": "ok"}]}]}
        )
        == "ok"
    )

    content_path = tmp_path / "content.bin"
    providers._write_binary_response(
        type("Resp", (), {"content": b"content"})(), str(content_path)
    )
    assert content_path.read_bytes() == b"content"

    read_path = tmp_path / "read.bin"
    providers._write_binary_response(
        type("Resp", (), {"read": lambda self: b"read"})(), str(read_path)
    )
    assert read_path.read_bytes() == b"read"

    bytes_path = tmp_path / "bytes.bin"
    providers._write_binary_response(b"bytes", str(bytes_path))
    assert bytes_path.read_bytes() == b"bytes"

    with pytest.raises(ProviderError, match="no writable binary"):
        providers._write_binary_response(object(), str(tmp_path / "missing.bin"))

    image_path = tmp_path / "image-url.png"

    class Download:
        content = b"downloaded"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(providers.httpx, "get", lambda url, timeout=60: Download())
    providers._write_image_item({"url": "https://example.test/image.png"}, image_path)
    assert image_path.read_bytes() == b"downloaded"


def test_provider_sse_and_error_helpers_cover_edge_cases():
    result = providers._consume_sse_text(
        "\n".join(
            [
                "event: ignored",
                "data: {not json}",
                'data: {"type":"response.output_item.done","item":{"id":"item_1"}}',
                'data: {"type":"response.content_part.done","part":{"text":"fallback text"}}',
            ]
        ),
        on_delta=None,
    )

    assert result.text == "fallback text"
    assert result.response_id == "item_1"
    assert providers._sse_image_results("data: {not json}\n") == []
    assert providers._responses_unsupported_for_fallback(
        RuntimeError("responses endpoint is not supported")
    )
    assert not providers._responses_unsupported_for_fallback(
        RuntimeError("plain failure")
    )
    assert (
        providers._exception_status_code(
            type("Exc", (), {"response": type("Resp", (), {"status_code": 422})()})()
        )
        == 422
    )
    assert providers._is_expired(None) is False
    assert providers._is_expired("not-a-date") is False


def test_pdf_fallback_import_and_runtime_errors_are_wrapped(monkeypatch):
    attachment = Attachment(
        kind="file",
        source="report.pdf",
        filename="report.pdf",
        mime_type="application/pdf",
        payload="data:application/pdf;base64,abc",
    )
    monkeypatch.setitem(sys.modules, "fitz", None)
    with pytest.raises(ProviderError, match="requires PyMuPDF"):
        providers._pdf_page_attachments(attachment)

    fake_fitz = types.SimpleNamespace(
        Matrix=lambda x, y: object(),
        open=lambda path: (_ for _ in ()).throw(RuntimeError("cannot render")),
    )
    monkeypatch.setitem(sys.modules, "fitz", fake_fitz)
    with pytest.raises(ProviderError, match="PDF vision fallback failed"):
        providers._pdf_page_attachments(attachment)


def test_response_error_detail_handles_text_and_nested_json():
    request = httpx.Request("POST", "https://example.test")
    text_response = httpx.Response(500, text="  plain failure  ", request=request)
    nested_response = httpx.Response(
        500, json={"error": {"message": "nested failure"}}, request=request
    )
    empty_response = httpx.Response(500, json={"error": {}}, request=request)

    assert providers._response_error_detail(text_response) == "plain failure"
    assert providers._response_error_detail(nested_response) == "nested failure"
    assert providers._response_error_detail(empty_response) is None
