from pathlib import Path


WORKFLOW = Path(".github/workflows/release-artifacts.yml")


def test_release_artifact_workflow_builds_pyinstaller_matrix():
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "name: Release artifacts" in text
    assert "contents: write" in text
    assert "strategy:" in text
    assert "matrix:" in text
    for runner in (
        "windows-latest",
        "ubuntu-22.04",
        "ubuntu-22.04-arm",
        "macos-15-intel",
        "macos-15",
    ):
        assert f"os: {runner}" in text

    assert "executable: llmrun.exe" in text
    assert text.count("executable: llmrun\n") == 4
    assert "executable: llmrun-windows-x64.exe" not in text
    assert "executable: llmrun-linux-x64" not in text
    assert "executable: llmrun-linux-arm64" not in text
    assert "executable: llmrun-macos-x64" not in text
    assert "executable: llmrun-macos-arm64" not in text

    for package in (
        "llmrun-windows-x64.zip",
        "llmrun-linux-x64.tar.gz",
        "llmrun-linux-arm64.tar.gz",
        "llmrun-macos-x64.tar.gz",
        "llmrun-macos-arm64.tar.gz",
    ):
        assert package in text

    assert 'python -m pip install -e ".[release,dev]"' in text
    assert "llmrun release pyinstaller --output" in text
    assert "chmod +x dist/${{ matrix.executable }}" in text
    assert "tar -czf dist/${{ matrix.package }}" in text
    assert "Compress-Archive" in text
    assert "actions/upload-artifact" in text
    assert "path: dist/${{ matrix.package }}" in text
    assert "path: dist/${{ matrix.executable }}" not in text
    assert "publish-release:" in text
    assert "softprops/action-gh-release@v2" in text
