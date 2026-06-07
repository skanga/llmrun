import json

import pytest
from click.testing import CliRunner

from llmrun import cli
from llmrun.cli import app
from llmrun.providers import DEFAULT_MODEL, ProviderError
from llmrun.state import StateStore

runner = CliRunner()


def test_cli_one_shot_prompt(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    class FakeProvider:
        def complete(self, request, on_delta=None):
            return type(
                "Result",
                (),
                {"text": f"answer:{request.input}", "response_id": "resp_1"},
            )()

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider()
    )

    result = runner.invoke(app, ["hello"])

    assert result.exit_code == 0
    assert "answer:hello" in result.stdout


def test_cli_no_args_shows_help():
    result = runner.invoke(app, [])

    assert result.exit_code == 0
    assert "Run one prompt directly" in result.stdout


def test_cli_stdin_prompt(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    class FakeProvider:
        def complete(self, request, on_delta=None):
            return type(
                "Result", (), {"text": request.input.upper(), "response_id": "resp_1"}
            )()

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider()
    )

    result = runner.invoke(app, ["-"], input="from stdin")

    assert result.exit_code == 0
    assert "FROM STDIN" in result.stdout


def test_cli_named_session_continuation(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    seen_previous = []

    class FakeProvider:
        def complete(self, request, on_delta=None):
            seen_previous.append(request.previous_response_id)
            response_id = "resp_1" if request.input == "first" else "resp_2"
            return type("Result", (), {"text": "ok", "response_id": response_id})()

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider()
    )

    first = runner.invoke(app, ["--session", "work", "first"])
    second = runner.invoke(app, ["--session", "work", "second"])

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert seen_previous == [None, "resp_1"]


def test_cli_codex_session_replays_transcript_instead_of_previous_response_id(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path))
    codex_auth = tmp_path / "auth.json"
    codex_auth.write_text(
        json.dumps({"tokens": {"access_token": "access", "refresh_token": "refresh"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_AUTH_JSON_PATH", str(codex_auth))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    seen = []

    class FakeProvider:
        name = "codex"

        def complete(self, request, on_delta=None):
            seen.append((request.input, request.previous_response_id))
            response_id = "resp_1" if len(seen) == 1 else "resp_2"
            text = "first answer" if len(seen) == 1 else "second answer"
            return type("Result", (), {"text": text, "response_id": response_id})()

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider()
    )

    first = runner.invoke(app, ["--session", "work", "first question"])
    second = runner.invoke(app, ["--session", "work", "second question"])

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert seen[0] == ("first question", None)
    assert seen[1][1] is None
    assert "first question" in seen[1][0]
    assert "first answer" in seen[1][0]
    assert "second question" in seen[1][0]


def test_cli_base_url_session_replays_transcript_instead_of_previous_response_id(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "provider-key")
    seen = []

    class FakeProvider:
        name = "openai"

        def complete(self, request, on_delta=None):
            seen.append((request.input, request.previous_response_id, self.base_url))
            response_id = "resp_1" if len(seen) == 1 else "resp_2"
            text = "first answer" if len(seen) == 1 else "second answer"
            return type("Result", (), {"text": text, "response_id": response_id})()

        def __init__(self, base_url):
            self.base_url = base_url

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider(base_url)
    )

    first = runner.invoke(
        app,
        [
            "--base-url",
            "https://provider.test/v1",
            "--session",
            "work",
            "first question",
        ],
    )
    second = runner.invoke(
        app,
        [
            "--base-url",
            "https://provider.test/v1",
            "--session",
            "work",
            "second question",
        ],
    )

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert seen[0] == ("first question", None, "https://provider.test/v1")
    assert seen[1][1] is None
    assert "first question" in seen[1][0]
    assert "first answer" in seen[1][0]
    assert "second question" in seen[1][0]


def test_cli_session_provider_change_replays_transcript(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    seen = []

    class FakeProvider:
        name = "openai"

        def complete(self, request, on_delta=None):
            seen.append((request.input, request.previous_response_id, self.base_url))
            response_id = "resp_1" if len(seen) == 1 else "resp_2"
            text = "first answer" if len(seen) == 1 else "second answer"
            return type("Result", (), {"text": text, "response_id": response_id})()

        def __init__(self, base_url):
            self.base_url = base_url

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider(base_url)
    )

    first = runner.invoke(app, ["--session", "work", "first question"])
    second = runner.invoke(
        app, ["--provider", "groq", "--session", "work", "second question"]
    )

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert seen[0] == ("first question", None, None)
    assert seen[1][1] is None
    assert seen[1][2] == "https://api.groq.com/openai/v1"
    assert "first question" in seen[1][0]
    assert "first answer" in seen[1][0]
    assert "second question" in seen[1][0]


def test_cli_config_templates_fragments_and_export(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    class FakeProvider:
        def complete(self, request, on_delta=None):
            return type("Result", (), {"text": "ok", "response_id": "resp_1"})()

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider()
    )

    assert (
        runner.invoke(app, ["config", "set", "default_model", "gpt-test"]).exit_code
        == 0
    )
    assert "gpt-test" in runner.invoke(app, ["config", "get", "default_model"]).stdout

    assert runner.invoke(app, ["templates", "run", "bug", "ignored"]).exit_code != 0
    assert (
        runner.invoke(app, ["templates", "add", "bug", "Bug: {prompt}"]).exit_code == 0
    )
    assert "bug" in runner.invoke(app, ["templates", "list"]).stdout
    assert runner.invoke(app, ["templates", "delete", "bug"]).exit_code == 0
    assert "bug" not in runner.invoke(app, ["templates", "list"]).stdout
    assert (
        runner.invoke(app, ["templates", "add", "bug", "Bug: {prompt}"]).exit_code == 0
    )

    assert runner.invoke(app, ["fragments", "add", "style", "Be terse."]).exit_code == 0
    assert "style" in runner.invoke(app, ["fragments", "list"]).stdout

    run = runner.invoke(app, ["--session", "work", "--fragment", "style", "hello"])
    assert run.exit_code == 0

    exported = runner.invoke(app, ["sessions", "export", "work", "--format", "json"])
    assert exported.exit_code == 0
    assert json.loads(exported.stdout)["name"] == "work"


def test_cli_sessions_export_format_is_long_only(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path))

    help_result = runner.invoke(app, ["sessions", "export", "-h"])
    short_result = runner.invoke(app, ["sessions", "export", "work", "-f", "json"])

    assert help_result.exit_code == 0
    assert "--format TEXT" in help_result.stdout
    assert "-f, --format" not in help_result.stdout
    assert short_result.exit_code != 0
    assert "No such option: -f" in short_result.stderr


def test_cli_state_commands_cover_show_list_delete_and_errors(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    class FakeProvider:
        def complete(self, request, on_delta=None):
            return type("Result", (), {"text": "ok", "response_id": "resp_1"})()

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider()
    )

    assert runner.invoke(app, ["--session", "work", "hello"]).exit_code == 0
    listed = runner.invoke(app, ["sessions", "list"])
    shown = runner.invoke(app, ["sessions", "show", "work"])
    invalid_export = runner.invoke(
        app, ["sessions", "export", "work", "--format", "html"]
    )
    missing_show = runner.invoke(app, ["sessions", "show", "missing"])
    missing_delete = runner.invoke(app, ["sessions", "delete", "missing"])
    deleted = runner.invoke(app, ["sessions", "delete", "work"])

    assert listed.exit_code == 0
    assert "work" in listed.stdout
    assert shown.exit_code == 0
    assert "# Session: work" in shown.stdout
    assert invalid_export.exit_code == 1
    assert "json or markdown" in invalid_export.stderr
    assert missing_show.exit_code == 1
    assert "Unknown session" in missing_show.stderr
    assert missing_delete.exit_code == 1
    assert deleted.exit_code == 0


def test_cli_template_fragment_and_config_error_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path))

    assert (
        runner.invoke(app, ["templates", "add", "bug", "Bug: {prompt}"]).exit_code == 0
    )
    assert runner.invoke(app, ["templates", "show", "bug"]).stdout == "Bug: {prompt}\n"
    assert runner.invoke(app, ["templates", "show", "missing"]).exit_code == 1
    assert runner.invoke(app, ["templates", "delete", "missing"]).exit_code == 1

    assert runner.invoke(app, ["fragments", "add", "style", "Be terse."]).exit_code == 0
    assert runner.invoke(app, ["fragments", "show", "style"]).stdout == "Be terse.\n"
    assert runner.invoke(app, ["fragments", "show", "missing"]).exit_code == 1
    assert runner.invoke(app, ["fragments", "delete", "missing"]).exit_code == 1

    assert runner.invoke(app, ["config", "get", "missing"]).exit_code == 1
    assert runner.invoke(app, ["config", "unset", "missing"]).exit_code == 0


def test_cli_provider_error_is_reported_without_traceback(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    class FakeProvider:
        def complete(self, request, on_delta=None):
            raise ProviderError(
                "Codex OAuth provider request failed: HTTP 400: model rejected"
            )

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider()
    )

    result = runner.invoke(app, ["hello"])

    assert result.exit_code == 1
    assert "HTTP 400" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_builtin_default_model_is_auth_scheme_neutral(monkeypatch, tmp_path):
    seen = []

    class FakeProvider:
        def __init__(self, provider_name):
            self.name = provider_name

        def complete(self, request, on_delta=None):
            seen.append((self.name, request.model))
            return type("Result", (), {"text": "ok", "response_id": "resp_1"})()

    monkeypatch.setattr(
        "llmrun.cli.make_provider",
        lambda auth, base_url=None: FakeProvider(auth.provider),
    )

    openai_state = tmp_path / "openai-state"
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(openai_state))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert runner.invoke(app, ["hello"]).exit_code == 0

    codex_state = tmp_path / "codex-state"
    codex_auth = tmp_path / "auth.json"
    codex_auth.write_text(
        json.dumps({"tokens": {"access_token": "access", "refresh_token": "refresh"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(codex_state))
    monkeypatch.setenv("CODEX_AUTH_JSON_PATH", str(codex_auth))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert runner.invoke(app, ["hello"]).exit_code == 0

    assert seen == [("openai", DEFAULT_MODEL), ("codex", DEFAULT_MODEL)]


def test_cli_groq_base_url_uses_groq_default_model(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "gsk-test")
    monkeypatch.setenv("LLMRUN_BASE_URL", "https://api.groq.com/openai/v1")
    seen = []

    class FakeProvider:
        name = "openai"

        def complete(self, request, on_delta=None):
            seen.append(request.model)
            return type("Result", (), {"text": "ok", "response_id": "resp_1"})()

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider()
    )

    result = runner.invoke(app, ["hello"])

    assert result.exit_code == 0
    assert seen == ["openai/gpt-oss-120b"]


def test_cli_groq_base_url_uses_groq_vision_default_for_image(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "gsk-test")
    monkeypatch.setenv("LLMRUN_BASE_URL", "https://api.groq.com/openai/v1")
    image = tmp_path / "receipt.png"
    image.write_bytes(b"png")
    seen = []

    class FakeProvider:
        name = "openai"

        def complete(self, request, on_delta=None):
            seen.append(request.model)
            return type("Result", (), {"text": "ok", "response_id": "resp_1"})()

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider()
    )

    result = runner.invoke(app, ["--image", str(image), "extract text"])

    assert result.exit_code == 0
    assert seen == ["meta-llama/llama-4-scout-17b-16e-instruct"]


def test_cli_groq_base_url_uses_groq_vision_default_for_local_pdf(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "gsk-test")
    monkeypatch.setenv("LLMRUN_BASE_URL", "https://api.groq.com/openai/v1")
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF")
    seen = []

    class FakeProvider:
        name = "openai"

        def complete(self, request, on_delta=None):
            seen.append(request.model)
            return type("Result", (), {"text": "ok", "response_id": "resp_1"})()

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider()
    )

    result = runner.invoke(app, ["--file", str(pdf), "summarize"])

    assert result.exit_code == 0
    assert seen == ["meta-llama/llama-4-scout-17b-16e-instruct"]


def test_cli_provider_groq_sets_base_url_and_default_model(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "gsk-test")
    monkeypatch.delenv("LLMRUN_BASE_URL", raising=False)
    seen = []

    class FakeProvider:
        name = "openai"

        def complete(self, request, on_delta=None):
            seen.append((request.model, self.base_url))
            return type("Result", (), {"text": "ok", "response_id": "resp_1"})()

        def __init__(self, base_url):
            self.base_url = base_url

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider(base_url)
    )

    result = runner.invoke(app, ["--provider", "groq", "hello"])

    assert result.exit_code == 0
    assert seen == [
        ("openai/gpt-oss-120b", "https://api.groq.com/openai/v1")
    ]


def test_cli_provider_groq_missing_key_mentions_groq_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    result = runner.invoke(app, ["--provider", "groq", "hello"])

    assert result.exit_code == 1
    assert "GROQ_API_KEY" in result.stderr
    assert "OPENAI_API_KEY" not in result.stderr


@pytest.mark.parametrize(
    ("provider_name", "env_name", "base_url", "default_model"),
    [
        (
            "nvidia-nim",
            "NVIDIA_API_KEY",
            "https://integrate.api.nvidia.com/v1",
            "openai/gpt-oss-120b",
        ),
        (
            "sambanova",
            "SAMBANOVA_API_KEY",
            "https://api.sambanova.ai/v1",
            "gpt-oss-120b",
        ),
        ("cerebras", "CEREBRAS_API_KEY", "https://api.cerebras.ai/v1", "gpt-oss-120b"),
        (
            "openrouter",
            "OPENROUTER_API_KEY",
            "https://openrouter.ai/api/v1",
            "openai/gpt-oss-120b:free",
        ),
        (
            "together",
            "TOGETHER_API_KEY",
            "https://api.together.xyz/v1",
            "openai/gpt-oss-120b",
        ),
        (
            "deepinfra",
            "DEEPINFRA_API_KEY",
            "https://api.deepinfra.com/v1/openai",
            "deepseek-ai/DeepSeek-V3",
        ),
    ],
)
def test_cli_provider_preset_sets_base_url_key_and_default_model(
    monkeypatch, tmp_path, provider_name, env_name, base_url, default_model
):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv(env_name, "provider-key")
    seen = []

    class FakeProvider:
        name = "openai"

        def __init__(self, api_key, base_url):
            self.api_key = api_key
            self.base_url = base_url

        def complete(self, request, on_delta=None):
            seen.append((request.model, self.api_key, self.base_url))
            return type("Result", (), {"text": "ok", "response_id": "resp_1"})()

    monkeypatch.setattr(
        "llmrun.cli.make_provider",
        lambda auth, base_url=None: FakeProvider(auth.api_key, base_url),
    )

    result = runner.invoke(app, ["--provider", provider_name, "hello"])

    assert result.exit_code == 0
    assert seen == [(default_model, "provider-key", base_url)]


def test_cli_provider_preset_allows_explicit_model_and_base_url(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("NVIDIA_API_KEY", "nv-key")
    seen = []

    class FakeProvider:
        name = "openai"

        def complete(self, request, on_delta=None):
            seen.append((request.model, self.base_url))
            return type("Result", (), {"text": "ok", "response_id": "resp_1"})()

        def __init__(self, base_url):
            self.base_url = base_url

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider(base_url)
    )

    result = runner.invoke(
        app,
        [
            "--provider",
            "nvidia-nim",
            "--base-url",
            "http://localhost:8000/v1",
            "--model",
            "local-model",
            "hello",
        ],
    )

    assert result.exit_code == 0
    assert seen == [("local-model", "http://localhost:8000/v1")]


def test_cli_provider_openai_ignores_env_base_url(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLMRUN_BASE_URL", "https://api.groq.com/openai/v1")
    seen = []

    class FakeProvider:
        name = "openai"

        def complete(self, request, on_delta=None):
            seen.append((request.model, self.base_url))
            return type("Result", (), {"text": "ok", "response_id": "resp_1"})()

        def __init__(self, base_url):
            self.base_url = base_url

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider(base_url)
    )

    result = runner.invoke(app, ["--provider", "openai", "hello"])

    assert result.exit_code == 0
    assert seen == [(DEFAULT_MODEL, None)]


def test_cli_provider_codex_uses_oauth_even_with_openai_key(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    codex_auth = tmp_path / "auth.json"
    codex_auth.write_text(
        json.dumps({"tokens": {"access_token": "access", "refresh_token": "refresh"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_AUTH_JSON_PATH", str(codex_auth))
    seen = []

    class FakeProvider:
        name = "codex"

        def complete(self, request, on_delta=None):
            seen.append(self.provider_name)
            return type("Result", (), {"text": "ok", "response_id": "resp_1"})()

        def __init__(self, provider_name):
            self.provider_name = provider_name

    monkeypatch.setattr(
        "llmrun.cli.make_provider",
        lambda auth, base_url=None: FakeProvider(auth.provider),
    )

    result = runner.invoke(app, ["--provider", "codex", "hello"])

    assert result.exit_code == 0
    assert seen == ["codex"]


def test_cli_base_url_option_and_config(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    seen = []

    class FakeProvider:
        name = "openai"

        def complete(self, request, on_delta=None):
            return type("Result", (), {"text": "ok", "response_id": "resp_1"})()

    def fake_make_provider(auth, base_url=None):
        seen.append(base_url)
        return FakeProvider()

    monkeypatch.setattr("llmrun.cli.make_provider", fake_make_provider)

    assert (
        runner.invoke(
            app, ["config", "set", "base_url", "https://configured.test/v1"]
        ).exit_code
        == 0
    )
    assert runner.invoke(app, ["hello"]).exit_code == 0
    assert (
        runner.invoke(app, ["--base-url", "https://flag.test/v1", "hello"]).exit_code
        == 0
    )

    assert seen == ["https://configured.test/v1", "https://flag.test/v1"]


def test_cli_session_export_includes_base_url_for_new_sessions(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "provider-key")

    class FakeProvider:
        name = "openai"

        def complete(self, request, on_delta=None):
            return type("Result", (), {"text": "ok", "response_id": "resp_1"})()

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider()
    )

    assert (
        runner.invoke(
            app,
            ["--base-url", "https://provider.test/v1", "--session", "work", "hello"],
        ).exit_code
        == 0
    )

    exported = runner.invoke(app, ["sessions", "export", "work", "--format", "json"])

    assert exported.exit_code == 0
    assert json.loads(exported.stdout)["base_url"] == "https://provider.test/v1"


def test_cli_base_url_env_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLMRUN_BASE_URL", "https://env.test/v1")
    seen = []

    class FakeProvider:
        name = "openai"

        def complete(self, request, on_delta=None):
            return type("Result", (), {"text": "ok", "response_id": "resp_1"})()

    monkeypatch.setattr(
        "llmrun.cli.make_provider",
        lambda auth, base_url=None: seen.append(base_url) or FakeProvider(),
    )

    assert runner.invoke(app, ["hello"]).exit_code == 0

    assert seen == ["https://env.test/v1"]


def test_cli_help_explains_common_root_options():
    result = runner.invoke(app, ["-h"])

    assert result.exit_code == 0
    assert "Run one prompt directly" in result.stdout
    assert "-m, --model TEXT" in result.stdout
    assert "Model name for prompt, audio, image, or" in result.stdout
    assert "-p, --provider [openai|codex|groq" in result.stdout
    assert "-i, --image TEXT" in result.stdout
    assert "Attach an image path or HTTPS URL. Repeat" in result.stdout
    assert "-f, --file TEXT" in result.stdout
    assert "Attach a file path or HTTPS URL. Repeat" in result.stdout
    assert "--max-output-tokens INTEGER" in result.stdout
    assert "-n, --max-output-tokens" not in result.stdout
    assert "Transcribe, translate, or synthesize audio." in result.stdout


def test_cli_common_short_options_route_to_prompt(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    image = tmp_path / "receipt.png"
    file = tmp_path / "report.pdf"
    image.write_bytes(b"png")
    file.write_bytes(b"pdf")
    seen = []

    class FakeProvider:
        name = "openai"

        def complete(self, request, on_delta=None):
            seen.append(
                (
                    request.model,
                    request.temperature,
                    request.max_output_tokens,
                    request.json_mode,
                    [
                        (attachment.kind, attachment.source)
                        for attachment in request.attachments
                    ],
                )
            )
            return type("Result", (), {"text": "ok", "response_id": "resp_1"})()

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider()
    )

    result = runner.invoke(
        app,
        [
            "-m",
            "gpt-test",
            "-s",
            "work",
            "-i",
            str(image),
            "-f",
            str(file),
            "-t",
            "0.2",
            "--max-output-tokens",
            "128",
            "-j",
            "hello",
        ],
    )

    assert result.exit_code == 0
    assert seen == [
        (
            "gpt-test",
            0.2,
            128,
            True,
            [("image", str(image)), ("file", str(file))],
        )
    ]


def test_cli_root_n_short_option_is_not_used_for_tokens(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    result = runner.invoke(app, ["-n", "128", "hello"])

    assert result.exit_code != 0
    assert "No such option: -n" in result.stderr


def test_cli_streaming_not_implemented_retry_prints_once(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    class FakeProvider:
        def complete(self, request, on_delta=None):
            if request.stream:
                raise NotImplementedError("streaming unsupported")
            return type("Result", (), {"text": "ok", "response_id": "resp_1"})()

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider()
    )

    result = runner.invoke(app, ["hello"])

    assert result.exit_code == 0
    assert result.stdout == "ok\n"


def test_cli_chat_loop_runs_until_quit(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMRUN_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    seen = []

    class FakeProvider:
        def complete(self, request, on_delta=None):
            seen.append(request.input)
            return type("Result", (), {"text": "ok", "response_id": "resp_1"})()

    monkeypatch.setattr(
        "llmrun.cli.make_provider", lambda auth, base_url=None: FakeProvider()
    )

    result = runner.invoke(app, ["chat", "--session", "work"], input="hello\nquit\n")

    assert result.exit_code == 0
    assert seen == ["hello"]


def test_cli_private_helper_wrappers(monkeypatch, tmp_path):
    store = StateStore(tmp_path / "state")
    store.set_fragment("style", "Be terse.")
    store.set_config("default_model", "gpt-test")

    assert cli._require_fragment(store, "style") == "Be terse."
    with pytest.raises(Exception, match="Unknown fragment"):
        cli._require_fragment(store, "missing")

    assert cli._session_replay_input([], "hello").endswith("User:\nhello")
    assert cli._should_replay_session(
        type("S", (), {"provider": "openai", "base_url": None})(),
        provider_name="codex",
        base_url=None,
    )
    assert (
        cli._normalize_base_url("HTTPS://EXAMPLE.TEST/v1/") == "https://example.test/v1"
    )
    assert (
        cli._audio_model_default(
            None,
            store=store,
            provider_choice=None,
            openai_default="openai-audio",
            groq_default="groq-audio",
        )
        == "openai-audio"
    )
    assert (
        cli._prompt_model_default(None, existing_model=None, store=store, base_url=None)
        == "gpt-test"
    )
    assert cli._configured_base_url(store=store, provider_choice="openai") is None
    assert cli._provider_preset(None) is None
    assert cli._provider_preset("groq")["api_key_env"] == "GROQ_API_KEY"
    assert cli._default_model_for_base_url("https://api.groq.com/openai/v1")
    assert cli._is_groq_base_url("https://api.groq.com/openai/v1")

    monkeypatch.chdir(tmp_path)
    assert cli._timestamped_output("speech", ".mp3").startswith(
        str(tmp_path / "speech-")
    )


def test_cli_provider_from_auth_wrapper_reports_auth_errors(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CODEX_AUTH_JSON_PATH", str(tmp_path / "missing.json"))

    with pytest.raises(Exception, match="No auth found"):
        cli._provider_from_auth(store=StateStore(tmp_path / "state"))
