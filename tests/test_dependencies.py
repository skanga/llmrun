import tomllib


def test_runtime_dependencies_do_not_include_unused_packages():
    with open("pyproject.toml", "rb") as handle:
        data = tomllib.load(handle)

    names = {
        dependency.split(">=", 1)[0].split("==", 1)[0].lower()
        for dependency in data["project"]["dependencies"]
    }

    assert "pydantic" not in names
    assert "typer" not in names
    assert "pymupdf" not in names


def test_pdf_dependency_is_optional():
    with open("pyproject.toml", "rb") as handle:
        data = tomllib.load(handle)

    pdf_dependencies = {
        dependency.split(">=", 1)[0].split("==", 1)[0].lower()
        for dependency in data["project"]["optional-dependencies"]["pdf"]
    }

    assert "pymupdf" in pdf_dependencies


def test_pex_dependency_is_release_only():
    with open("pyproject.toml", "rb") as handle:
        data = tomllib.load(handle)

    runtime_dependencies = {
        dependency.split(">=", 1)[0].split("==", 1)[0].lower()
        for dependency in data["project"]["dependencies"]
    }
    release_dependencies = {
        dependency.split(">=", 1)[0].split("==", 1)[0].lower()
        for dependency in data["project"]["optional-dependencies"]["release"]
    }

    assert "pex" not in runtime_dependencies
    assert "pex" in release_dependencies
    assert "pyinstaller" not in runtime_dependencies
    assert "pyinstaller" in release_dependencies
