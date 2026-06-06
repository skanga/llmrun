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
