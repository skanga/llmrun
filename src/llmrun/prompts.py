from __future__ import annotations

from dataclasses import dataclass
from string import Template
import re


@dataclass(frozen=True)
class PromptRequest:
    input: str
    instructions: str | None = None


def build_prompt(
    *,
    prompt: str | None,
    stdin_text: str | None,
    system: str | None,
    fragments: list[str],
) -> PromptRequest:
    if prompt == "-":
        input_text = stdin_text or ""
    else:
        input_text = prompt or stdin_text or ""
    instructions_parts = [part for part in [system, *fragments] if part]
    instructions = "\n\n".join(instructions_parts) if instructions_parts else None
    return PromptRequest(input=input_text, instructions=instructions)


def render_template(
    template_text: str, *, prompt: str, variables: dict[str, str] | None = None
) -> str:
    values = {"prompt": prompt}
    if variables:
        values.update(variables)
    brace_rendered = _safe_brace_substitute(template_text, values)
    return Template(brace_rendered).safe_substitute(values)


def _safe_brace_substitute(template_text: str, values: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        return str(values.get(name, match.group(0)))

    return re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", replace, template_text)
