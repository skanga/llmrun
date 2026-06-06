from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

RUNTIME_DIR = Path(__file__).resolve().parent.parent / ".pytest-runtime"
RUNTIME_DIR.mkdir(exist_ok=True)
TMP_DIR = RUNTIME_DIR / f"tmp-{os.getpid()}"
TMP_DIR.mkdir(exist_ok=True)

for name in ("TMP", "TEMP", "TMPDIR"):
    os.environ[name] = str(TMP_DIR)

tempfile.tempdir = (
    os.environ.get("TMPDIR") or os.environ.get("TEMP") or os.environ.get("TMP")
)


@pytest.fixture(autouse=True)
def clean_provider_environment(monkeypatch):
    for name in (
        "OPENAI_API_KEY",
        "GROQ_API_KEY",
        "NVIDIA_API_KEY",
        "SAMBANOVA_API_KEY",
        "CEREBRAS_API_KEY",
        "OPENROUTER_API_KEY",
        "TOGETHER_API_KEY",
        "DEEPINFRA_API_KEY",
        "LLMRUN_BASE_URL",
        "LLMRUN_PROVIDER",
        "LLMRUN_STATE_DIR",
        "CODEX_AUTH_JSON_PATH",
        "CODEX_HOME",
    ):
        monkeypatch.delenv(name, raising=False)
