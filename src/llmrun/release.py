from __future__ import annotations

import subprocess
import sys
import tempfile
import zipapp
from pathlib import Path


MAIN_MODULE = r'''
import hashlib
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


def _extracted_app_dir() -> Path:
    archive = Path(sys.argv[0]).resolve()
    stat = archive.stat()
    cache_key = hashlib.sha256(
        f"{archive}|{stat.st_mtime_ns}|{stat.st_size}".encode("utf-8")
    ).hexdigest()[:24]
    app_dir = Path(tempfile.gettempdir()) / "llmrun-pyz-cache" / cache_key
    marker = app_dir / ".llmrun-extracted"
    if marker.exists():
        return app_dir
    if app_dir.exists():
        shutil.rmtree(app_dir)
    app_dir.mkdir(parents=True)
    with zipfile.ZipFile(archive) as source:
        source.extractall(app_dir)
    marker.write_text("ok", encoding="utf-8")
    return app_dir


sys.path.insert(0, str(_extracted_app_dir()))

from llmrun.cli import main


if __name__ == '__main__':
    main()
'''.lstrip()

PYINSTALLER_LAUNCHER = (
    "from llmrun.cli import main\n\n"
    "if __name__ == '__main__':\n"
    "    main()\n"
)


def build_zipapp_release(
    *,
    output_path: str | Path,
    source_dir: str | Path = ".",
    include_pdf: bool = False,
    python_executable: str | None = None,
) -> Path:
    output = Path(output_path).resolve()
    source = Path(source_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    python = python_executable or sys.executable

    with tempfile.TemporaryDirectory(prefix="llmrun-pyz-") as temp_dir:
        app_dir = Path(temp_dir) / "app"
        app_dir.mkdir(parents=True, exist_ok=True)
        requirement = f"{source}[pdf]" if include_pdf else str(source)
        subprocess.run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--target",
                str(app_dir),
                requirement,
            ],
            check=True,
        )
        (app_dir / "__main__.py").write_text(MAIN_MODULE, encoding="utf-8")
        zipapp.create_archive(
            app_dir,
            target=output,
            interpreter="/usr/bin/env python3",
            compressed=True,
        )
    return output


def build_scie_release(
    *,
    output_path: str | Path,
    source_dir: str | Path = ".",
    include_pdf: bool = False,
    scie_mode: str = "eager",
    python_executable: str | None = None,
) -> Path:
    output = Path(output_path).resolve()
    source = Path(source_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    python = python_executable or sys.executable
    requirement = f"{source}[pdf]" if include_pdf else str(source)

    subprocess.run(
        [
            python,
            "-m",
            "pex",
            requirement,
            "-c",
            "llmrun",
            "--scie",
            scie_mode,
            "-o",
            str(output),
        ],
        check=True,
    )
    return output


def build_pyinstaller_release(
    *,
    output_path: str | Path,
    source_dir: str | Path = ".",
    include_pdf: bool = False,
    python_executable: str | None = None,
) -> Path:
    output = Path(output_path).resolve()
    source = Path(source_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    python = python_executable or sys.executable
    requirement = f"{source}[pdf]" if include_pdf else str(source)

    subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--upgrade",
            requirement,
        ],
        check=True,
    )
    with tempfile.TemporaryDirectory(prefix="llmrun-pyinstaller-") as temp_dir:
        launcher = Path(temp_dir) / "llmrun_pyinstaller_launcher.py"
        launcher.write_text(PYINSTALLER_LAUNCHER, encoding="utf-8")
        subprocess.run(
            [
                python,
                "-m",
                "PyInstaller",
                "--onefile",
                "--noupx",
                "--name",
                output.stem,
                "--distpath",
                str(output.parent),
                "--workpath",
                str(Path(temp_dir) / "build"),
                "--specpath",
                str(Path(temp_dir) / "spec"),
                str(launcher),
            ],
            check=True,
        )
    if not output.exists():
        raise FileNotFoundError(
            f"PyInstaller did not create the expected output: {output}"
        )
    return output
