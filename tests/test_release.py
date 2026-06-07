import subprocess
import sys
import zipapp
from pathlib import Path

import pytest
from click.testing import CliRunner

from llmrun import release
from llmrun.cli import app


runner = CliRunner()


def test_release_pyz_command_builds_default_zipapp(monkeypatch, tmp_path):
    calls = []
    output = tmp_path / "llmrun.pyz"

    def fake_build_zipapp_release(**kwargs):
        calls.append(kwargs)
        return output

    monkeypatch.setattr("llmrun.cli.build_zipapp_release", fake_build_zipapp_release)

    result = runner.invoke(app, ["release", "pyz", "--output", str(output)])

    assert result.exit_code == 0
    assert str(output) in result.stdout
    assert calls == [
        {
            "output_path": output,
            "source_dir": Path("."),
            "include_pdf": False,
            "python_executable": None,
        }
    ]


def test_release_pyz_command_can_include_pdf_extra(monkeypatch, tmp_path):
    calls = []
    output = tmp_path / "llmrun.pyz"

    def fake_build_zipapp_release(**kwargs):
        calls.append(kwargs)
        return output

    monkeypatch.setattr("llmrun.cli.build_zipapp_release", fake_build_zipapp_release)

    result = runner.invoke(
        app, ["release", "pyz", "--include-pdf", "--output", str(output)]
    )

    assert result.exit_code == 0
    assert calls[0]["include_pdf"] is True


def test_release_scie_command_builds_default_executable(monkeypatch, tmp_path):
    calls = []
    output = tmp_path / "llmrun.exe"

    def fake_build_scie_release(**kwargs):
        calls.append(kwargs)
        return output

    monkeypatch.setattr("llmrun.cli.build_scie_release", fake_build_scie_release)

    result = runner.invoke(app, ["release", "scie", "--output", str(output)])

    assert result.exit_code == 0
    assert str(output) in result.stdout
    assert calls == [
        {
            "output_path": output,
            "source_dir": Path("."),
            "include_pdf": False,
            "scie_mode": "eager",
            "python_executable": None,
        }
    ]


def test_release_scie_command_can_include_pdf_extra(monkeypatch, tmp_path):
    calls = []
    output = tmp_path / "llmrun.exe"

    def fake_build_scie_release(**kwargs):
        calls.append(kwargs)
        return output

    monkeypatch.setattr("llmrun.cli.build_scie_release", fake_build_scie_release)

    result = runner.invoke(
        app, ["release", "scie", "--include-pdf", "--output", str(output)]
    )

    assert result.exit_code == 0
    assert calls[0]["include_pdf"] is True


def test_release_pyinstaller_command_builds_default_executable(monkeypatch, tmp_path):
    calls = []
    output = tmp_path / "llmrun.exe"

    def fake_build_pyinstaller_release(**kwargs):
        calls.append(kwargs)
        return output

    monkeypatch.setattr(
        "llmrun.cli.build_pyinstaller_release", fake_build_pyinstaller_release
    )

    result = runner.invoke(app, ["release", "pyinstaller", "--output", str(output)])

    assert result.exit_code == 0
    assert str(output) in result.stdout
    assert calls == [
        {
            "output_path": output,
            "source_dir": Path("."),
            "include_pdf": False,
            "python_executable": None,
        }
    ]


def test_release_pyinstaller_command_can_include_pdf_extra(monkeypatch, tmp_path):
    calls = []
    output = tmp_path / "llmrun.exe"

    def fake_build_pyinstaller_release(**kwargs):
        calls.append(kwargs)
        return output

    monkeypatch.setattr(
        "llmrun.cli.build_pyinstaller_release", fake_build_pyinstaller_release
    )

    result = runner.invoke(
        app, ["release", "pyinstaller", "--include-pdf", "--output", str(output)]
    )

    assert result.exit_code == 0
    assert calls[0]["include_pdf"] is True


def test_build_zipapp_release_installs_project_without_pdf_by_default(
    monkeypatch, tmp_path
):
    commands = []
    archives = []
    main_modules = []

    def fake_run(command, check):
        commands.append((command, check))

    def fake_create_archive(source, *, target, interpreter, compressed):
        main_modules.append((source / "__main__.py").read_text(encoding="utf-8"))
        archives.append((source, target, interpreter, compressed))
        Path(target).write_bytes(b"pyz")

    monkeypatch.setattr(release.subprocess, "run", fake_run)
    monkeypatch.setattr(release.zipapp, "create_archive", fake_create_archive)

    output = release.build_zipapp_release(
        output_path=tmp_path / "dist" / "llmrun.pyz",
        source_dir=tmp_path,
        python_executable="python-test",
    )

    assert output == tmp_path / "dist" / "llmrun.pyz"
    assert commands == [
        (
            [
                "python-test",
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--target",
                str(archives[0][0]),
                str(tmp_path),
            ],
            True,
        )
    ]
    assert main_modules == [release.MAIN_MODULE]
    assert archives[0][1] == output
    assert archives[0][2] == "/usr/bin/env python3"
    assert archives[0][3] is True


def test_zipapp_main_imports_cli_from_extracted_directory(tmp_path):
    app_dir = tmp_path / "app"
    package_dir = app_dir / "llmrun"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "cli.py").write_text(
        "from pathlib import Path\n\n"
        "def main():\n"
        "    if not Path(__file__).is_file():\n"
        "        raise SystemExit(f'not extracted: {__file__}')\n"
        "    print('extracted')\n",
        encoding="utf-8",
    )
    (app_dir / "__main__.py").write_text(release.MAIN_MODULE, encoding="utf-8")
    archive = tmp_path / "llmrun.pyz"
    zipapp.create_archive(app_dir, target=archive)

    result = subprocess.run(
        [sys.executable, str(archive)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "extracted"


def test_build_scie_release_invokes_pex_with_embedded_python(monkeypatch, tmp_path):
    commands = []
    output = tmp_path / "dist" / "llmrun.exe"

    def fake_run(command, check):
        commands.append((command, check))

    monkeypatch.setattr(release.subprocess, "run", fake_run)

    result = release.build_scie_release(
        output_path=output,
        source_dir=tmp_path,
        python_executable="python-test",
    )

    assert result == output.resolve()
    assert commands == [
        (
            [
                "python-test",
                "-m",
                "pex",
                str(tmp_path),
                "-c",
                "llmrun",
                "--scie",
                "eager",
                "-o",
                str(output.resolve()),
            ],
            True,
        )
    ]


def test_build_scie_release_can_include_pdf_extra(monkeypatch, tmp_path):
    commands = []

    def fake_run(command, check):
        commands.append((command, check))

    monkeypatch.setattr(release.subprocess, "run", fake_run)

    release.build_scie_release(
        output_path=tmp_path / "llmrun.exe",
        source_dir=tmp_path,
        include_pdf=True,
        python_executable="python-test",
    )

    assert commands[0][0][3] == f"{tmp_path.resolve()}[pdf]"


def test_build_pyinstaller_release_installs_project_and_runs_pyinstaller(
    monkeypatch, tmp_path
):
    commands = []
    output = tmp_path / "dist" / "llmrun.exe"

    def fake_run(command, check):
        commands.append((command, check))
        if command[:3] == ["python-test", "-m", "PyInstaller"]:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"exe")

    monkeypatch.setattr(release.subprocess, "run", fake_run)

    result = release.build_pyinstaller_release(
        output_path=output,
        source_dir=tmp_path,
        python_executable="python-test",
    )

    assert result == output.resolve()
    install_command, build_command = commands
    assert install_command == (
        [
            "python-test",
            "-m",
            "pip",
            "install",
            "--upgrade",
            str(tmp_path),
        ],
        True,
    )
    assert build_command[0][:5] == [
        "python-test",
        "-m",
        "PyInstaller",
        "--onefile",
        "--noupx",
    ]
    assert build_command[0][5:7] == ["--name", output.resolve().stem]
    assert "--distpath" in build_command[0]
    launcher = Path(build_command[0][-1])
    assert launcher.name == "llmrun_pyinstaller_launcher.py"


def test_build_pyinstaller_release_raises_when_expected_output_is_missing(
    monkeypatch, tmp_path
):
    def fake_run(command, check):
        return None

    monkeypatch.setattr(release.subprocess, "run", fake_run)
    output = tmp_path / "dist" / "missing.exe"

    with pytest.raises(FileNotFoundError, match="PyInstaller did not create"):
        release.build_pyinstaller_release(
            output_path=output,
            source_dir=tmp_path,
            python_executable="python-test",
        )


def test_build_pyinstaller_release_returns_requested_output_when_created(
    monkeypatch, tmp_path
):
    output = tmp_path / "dist" / "created.exe"

    def fake_run(command, check):
        if command[:3] == ["python-test", "-m", "PyInstaller"]:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"exe")

    monkeypatch.setattr(release.subprocess, "run", fake_run)

    result = release.build_pyinstaller_release(
        output_path=output,
        source_dir=tmp_path,
        python_executable="python-test",
    )

    assert result == output.resolve()
    assert output.read_bytes() == b"exe"


def test_build_pyinstaller_release_can_include_pdf_extra(monkeypatch, tmp_path):
    commands = []

    def fake_run(command, check):
        commands.append((command, check))
        if command[:3] == ["python-test", "-m", "PyInstaller"]:
            (tmp_path / "llmrun.exe").write_bytes(b"exe")

    monkeypatch.setattr(release.subprocess, "run", fake_run)

    release.build_pyinstaller_release(
        output_path=tmp_path / "llmrun.exe",
        source_dir=tmp_path,
        include_pdf=True,
        python_executable="python-test",
    )

    assert commands[0][0][-1] == f"{tmp_path.resolve()}[pdf]"


def test_build_zipapp_release_can_install_pdf_extra(monkeypatch, tmp_path):
    commands = []

    def fake_run(command, check):
        commands.append((command, check))

    def fake_create_archive(source, *, target, interpreter, compressed):
        Path(target).write_bytes(b"pyz")

    monkeypatch.setattr(release.subprocess, "run", fake_run)
    monkeypatch.setattr(release.zipapp, "create_archive", fake_create_archive)

    release.build_zipapp_release(
        output_path=tmp_path / "llmrun.pyz",
        source_dir=tmp_path,
        include_pdf=True,
        python_executable="python-test",
    )

    assert commands[0][0][-1] == f"{tmp_path}[pdf]"
