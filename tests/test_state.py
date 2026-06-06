import json
from pathlib import Path

from llmrun import state
from llmrun.state import StateStore


def test_session_create_continue_delete_and_export(tmp_path):
    store = StateStore(tmp_path)
    session = store.update_session(
        "work",
        provider="openai",
        model="gpt-5.4-mini",
        response_id="resp_1",
        prompt="hello",
        output="hi",
    )

    assert session.response_id == "resp_1"
    assert store.get_session("work").turns[0].prompt == "hello"

    store.update_session(
        "work",
        provider="openai",
        model="gpt-5.4-mini",
        response_id="resp_2",
        prompt="next",
        output="done",
    )

    exported_json = store.export_session("work", "json")
    exported_md = store.export_session("work", "markdown")

    assert '"response_id": "resp_2"' in exported_json
    assert "## User" in exported_md
    assert "next" in exported_md

    assert store.delete_session("work") is True
    assert store.get_session("work") is None
    assert store.delete_session("work") is False


def test_export_unknown_session_and_format_errors(tmp_path):
    store = StateStore(tmp_path)

    try:
        store.export_session("missing", "json")
    except KeyError as exc:
        assert "Unknown session" in str(exc)
    else:
        raise AssertionError("expected missing session to raise")

    store.update_session(
        "work",
        provider="openai",
        model="gpt-test",
        response_id=None,
        prompt="hello",
        output="hi",
    )

    try:
        store.export_session("work", "html")
    except ValueError as exc:
        assert "json or markdown" in str(exc)
    else:
        raise AssertionError("expected invalid export format to raise")


def test_session_reads_legacy_json_without_base_url(tmp_path):
    store = StateStore(tmp_path)
    path = store.sessions_dir / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "name": "legacy",
                "provider": "openai",
                "model": "gpt-test",
                "response_id": "resp_1",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
                "turns": [],
            }
        ),
        encoding="utf-8",
    )

    session = store.get_session("legacy")

    assert session is not None
    assert session.base_url is None


def test_session_names_do_not_collide_after_safe_encoding(tmp_path):
    store = StateStore(tmp_path)

    store.update_session(
        "a/b",
        provider="openai",
        model="gpt-test",
        response_id="resp_slash",
        prompt="slash",
        output="one",
    )
    store.update_session(
        "a?b",
        provider="openai",
        model="gpt-test",
        response_id="resp_question",
        prompt="question",
        output="two",
    )

    assert store.get_session("a/b").response_id == "resp_slash"
    assert store.get_session("a?b").response_id == "resp_question"
    assert sorted(store.list_sessions()) == ["a/b", "a?b"]


def test_templates_fragments_and_config(tmp_path):
    store = StateStore(tmp_path)

    store.set_template("bug", "System: $system\nPrompt: $prompt")
    store.set_fragment("style", "Be concise.")
    store.set_config("default_model", "gpt-test")

    assert store.get_template("bug") == "System: $system\nPrompt: $prompt"
    assert store.get_fragment("style") == "Be concise."
    assert store.get_config("default_model") == "gpt-test"
    assert "bug" in store.list_templates()
    assert "style" in store.list_fragments()

    store.unset_config("default_model")
    store.delete_fragment("style")

    assert store.get_config("default_model") is None
    assert "style" not in store.list_fragments()
    assert store.delete_template("bug") is True
    assert store.delete_template("bug") is False


def test_template_and_fragment_names_do_not_collide_after_safe_encoding(tmp_path):
    store = StateStore(tmp_path)

    store.set_template("a/b", "slash template")
    store.set_template("a?b", "question template")
    store.set_fragment("a/b", "slash fragment")
    store.set_fragment("a?b", "question fragment")

    assert store.get_template("a/b") == "slash template"
    assert store.get_template("a?b") == "question template"
    assert store.get_fragment("a/b") == "slash fragment"
    assert store.get_fragment("a?b") == "question fragment"
    assert sorted(store.list_templates()) == ["a/b", "a?b"]
    assert sorted(store.list_fragments()) == ["a/b", "a?b"]


def test_legacy_normalized_template_fragment_and_session_files_still_work(tmp_path):
    store = StateStore(tmp_path)
    legacy_session = store.sessions_dir / "a-b.json"
    legacy_template = store.templates_dir / "a-b.txt"
    legacy_fragment = store.fragments_dir / "a-b.txt"
    legacy_session.write_text(
        json.dumps(
            {
                "name": "a/b",
                "provider": "openai",
                "model": "gpt-test",
                "response_id": "resp_1",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
                "turns": [],
            }
        ),
        encoding="utf-8",
    )
    legacy_template.write_text("legacy template", encoding="utf-8")
    legacy_fragment.write_text("legacy fragment", encoding="utf-8")

    assert store.get_session("a/b").response_id == "resp_1"
    assert store.get_template("a/b") == "legacy template"
    assert store.get_fragment("a/b") == "legacy fragment"
    assert store.delete_session("a/b") is True
    assert store.delete_template("a/b") is True
    assert store.delete_fragment("a/b") is True
    assert not legacy_session.exists()
    assert not legacy_template.exists()
    assert not legacy_fragment.exists()


def test_default_state_dir_disables_windows_appauthor_duplication(monkeypatch):
    calls = []

    def fake_user_data_dir(appname, appauthor=None):
        calls.append((appname, appauthor))
        return "C:/Users/example/AppData/Local/llmrun"

    monkeypatch.delenv("LLMRUN_STATE_DIR", raising=False)
    monkeypatch.setattr(state, "user_data_dir", fake_user_data_dir)

    assert state.default_state_dir() == Path("C:/Users/example/AppData/Local/llmrun")
    assert calls == [("llmrun", False)]


def test_old_state_dir_env_var_is_ignored_after_rename(monkeypatch):
    def fake_user_data_dir(appname, appauthor=None):
        return "C:/Users/example/AppData/Local/llmrun"

    monkeypatch.setenv("ANYLLM_STATE_DIR", "C:/old/state")
    monkeypatch.delenv("LLMRUN_STATE_DIR", raising=False)
    monkeypatch.setattr(state, "user_data_dir", fake_user_data_dir)

    assert state.default_state_dir() == Path("C:/Users/example/AppData/Local/llmrun")


def test_json_writes_are_atomic_and_leave_no_temp_files(tmp_path):
    store = StateStore(tmp_path)
    store.set_config("default_model", "old-model")

    store.set_config("default_model", "new-model")
    store.update_session(
        "work",
        provider="openai",
        model="gpt-test",
        response_id="resp_1",
        prompt="hello",
        output="hi",
    )

    assert json.loads(store.config_path.read_text(encoding="utf-8")) == {
        "default_model": "new-model"
    }
    assert store.get_session("work").response_id == "resp_1"
    assert list(tmp_path.rglob("*.tmp")) == []


def test_invalid_json_and_names_raise_clear_errors(tmp_path):
    store = StateStore(tmp_path)
    store.config_path.write_text("[]", encoding="utf-8")

    try:
        store.get_config("default_model")
    except ValueError as exc:
        assert "Expected JSON object" in str(exc)
    else:
        raise AssertionError("expected invalid config JSON to raise")

    try:
        state._safe_name("   ")
    except ValueError as exc:
        assert "safe character" in str(exc)
    else:
        raise AssertionError("expected unsafe name to raise")


def test_malformed_encoded_names_are_returned_as_stems(tmp_path):
    store = StateStore(tmp_path)
    (store.templates_dir / "~not-valid-base64.txt").write_text("text", encoding="utf-8")

    assert "~not-valid-base64" in store.list_templates()
