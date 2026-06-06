from llmrun.prompts import build_prompt, render_template


def test_build_prompt_from_argument_system_and_fragments():
    request = build_prompt(
        prompt="Explain this",
        stdin_text=None,
        system="You are terse.",
        fragments=["Use bullets."],
    )

    assert request.instructions == "You are terse.\n\nUse bullets."
    assert request.input == "Explain this"


def test_build_prompt_from_stdin_marker():
    request = build_prompt(
        prompt="-", stdin_text="stdin body", system=None, fragments=[]
    )

    assert request.input == "stdin body"


def test_render_template_substitutes_prompt_and_variables():
    rendered = render_template(
        "Bug: $prompt\nLang: $lang",
        prompt="crash on boot",
        variables={"lang": "python"},
    )

    assert rendered == "Bug: crash on boot\nLang: python"


def test_render_template_supports_shell_safe_brace_placeholders():
    rendered = render_template(
        "Bug: {prompt}\nLang: {lang}",
        prompt="crash on boot",
        variables={"lang": "python"},
    )

    assert rendered == "Bug: crash on boot\nLang: python"
