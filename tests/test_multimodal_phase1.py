import json

from click.testing import CliRunner

from llmrun.attachments import Attachment
from llmrun.cli import app
from llmrun.providers import CodexOAuthProvider, OpenAIProvider, ProviderRequest
from llmrun.auth import AuthInfo, OAuthTokens
from llmrun.state import StateStore

runner = CliRunner()


class FakeResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("Response", (), {"id": "resp_1", "output_text": "hello"})()


class FakeOpenAI:
    def __init__(self):
        self.responses = FakeResponses()


def test_openai_text_only_request_still_sends_string_input():
    created = {}

    def factory(api_key, base_url=None):
        client = FakeOpenAI()
        created["client"] = client
        return client

    provider = OpenAIProvider(
        AuthInfo(provider="openai", api_key="sk-test"), client_factory=factory
    )

    provider.complete(ProviderRequest(input_text="hi", model="gpt-test"))

    assert created["client"].responses.calls[0]["input"] == "hi"


def test_openai_multimodal_request_sends_structured_input():
    created = {}

    def factory(api_key, base_url=None):
        client = FakeOpenAI()
        created["client"] = client
        return client

    provider = OpenAIProvider(
        AuthInfo(provider="openai", api_key="sk-test"), client_factory=factory
    )

    provider.complete(
        ProviderRequest(
            input_text="describe",
            model="gpt-test",
            attachments=[
                Attachment(
                    kind="image",
                    source="screenshot.png",
                    filename="screenshot.png",
                    mime_type="image/png",
                    payload="data:image/png;base64,abc",
                    detail="low",
                ),
                Attachment(
                    kind="file",
                    source="https://example.test/report.pdf",
                    filename="report.pdf",
                    mime_type="application/pdf",
                ),
            ],
        )
    )

    payload = created["client"].responses.calls[0]["input"]
    assert payload == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "describe"},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,abc",
                    "detail": "low",
                },
                {
                    "type": "input_file",
                    "filename": "report.pdf",
                    "file_url": "https://example.test/report.pdf",
                },
            ],
        }
    ]


def test_codex_oauth_sends_structured_attachment_input():
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

    provider = CodexOAuthProvider(
        AuthInfo(
            provider="codex",
            oauth=OAuthTokens(access_token="access", refresh_token="refresh"),
        ),
        http_client=FakeHttp(),
    )

    provider.complete(
        ProviderRequest(
            input_text="describe",
            model="gpt-test",
            attachments=[
                Attachment(
                    kind="image",
                    source="https://example.test/image.png",
                    filename="image.png",
                    mime_type="image/png",
                    detail="high",
                ),
                Attachment(
                    kind="file",
                    source="https://example.test/report.pdf",
                    filename="report.pdf",
                    mime_type="application/pdf",
                ),
            ],
        )
    )

    payload = provider.http.posts[0][1]["json"]
    assert payload["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "describe"},
                {
                    "type": "input_image",
                    "image_url": "https://example.test/image.png",
                    "detail": "high",
                },
                {
                    "type": "input_file",
                    "filename": "report.pdf",
                    "file_url": "https://example.test/report.pdf",
                },
            ],
        }
    ]


def test_cli_image_flag_reaches_provider_with_attachment(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    image = tmp_path / "screenshot.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    seen = []

    class FakeProvider:
        name = "openai"

        def complete(self, request, on_delta=None):
            seen.append(request)
            return type("Result", (), {"text": "ok", "response_id": "resp_1"})()

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider()
    )

    result = runner.invoke(
        app, ["--image", str(image), "--image-detail", "low", "describe this"]
    )

    assert result.exit_code == 0
    assert seen[0].input_text == "describe this"
    assert seen[0].attachments[0].kind == "image"
    assert seen[0].attachments[0].payload.startswith("data:image/png;base64,")
    assert seen[0].attachments[0].detail == "low"


def test_session_export_includes_attachment_metadata_not_payload(tmp_path):
    store = StateStore(tmp_path)

    store.update_session(
        "work",
        provider="openai",
        model="gpt-test",
        response_id="resp_1",
        prompt="describe",
        output="ok",
        attachments=[
            Attachment(
                kind="image",
                source="screenshot.png",
                filename="screenshot.png",
                mime_type="image/png",
                payload="data:image/png;base64,secret",
                detail="high",
            )
        ],
    )

    exported = json.loads(store.export_session("work", "json"))

    assert exported["turns"][0]["attachments"] == [
        {
            "kind": "image",
            "source": "screenshot.png",
            "filename": "screenshot.png",
            "mime_type": "image/png",
            "detail": "high",
        }
    ]
    assert "secret" not in store._session_path("work").read_text(encoding="utf-8")
