# llmrun

`llmrun` is a small command-line client for OpenAI-style LLM prompts. It focuses on the parts that are useful in daily terminal work: one-shot prompts, image and file inputs, stdin prompts, named sessions, templates, reusable fragments, JSON output mode, and transcript export.

The primary provider uses OpenAI's Responses API. A secondary Codex OAuth provider can use tokens from Codex's local auth file when no OpenAI API key is available.

## Features

- Run a one-shot prompt from the shell.
- Attach images and files to root prompts with `--image` and `--file`.
- Transcribe audio, generate speech, and generate or edit images with explicit subcommands.
- Read a prompt body from stdin with `-`.
- Add system instructions with `--system`.
- Continue named sessions with `--session`.
- Export session transcripts as Markdown or JSON.
- Store and run prompt templates.
- Store reusable instruction fragments.
- Configure defaults such as `default_model`.
- Use `OPENAI_API_KEY`, an API key from Codex auth, or Codex OAuth tokens.

## Installation

From this repository:

```console
python -m pip install -e .
```

For development dependencies:

```console
python -m pip install -e ".[dev]"
```

After installation, the `llmrun` command is available:

```console
llmrun --help
```

You can also run the module directly from a source checkout:

macOS/Linux:

```sh
export PYTHONPATH=src
python -m llmrun.cli --help
```

Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m llmrun.cli --help
```

## Authentication

`llmrun` resolves auth in this order:

1. `OPENAI_API_KEY`
2. `OPENAI_API_KEY` inside Codex `auth.json`
3. Codex OAuth tokens inside Codex `auth.json`

The default Codex auth file is:

```text
~/.codex/auth.json
```

On Windows this is usually:

```text
%USERPROFILE%\.codex\auth.json
```

On macOS and Linux this is usually:

```text
~/.codex/auth.json
```

Supported auth and state environment variables:

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Use the official OpenAI provider. Highest priority. |
| `GROQ_API_KEY` | API key used by `--provider groq` when `OPENAI_API_KEY` is not set. |
| `NVIDIA_API_KEY` | API key used by `--provider nvidia` or `--provider nvidia-nim` when `OPENAI_API_KEY` is not set. |
| `SAMBANOVA_API_KEY` | API key used by `--provider sambanova` when `OPENAI_API_KEY` is not set. |
| `CEREBRAS_API_KEY` | API key used by `--provider cerebras` when `OPENAI_API_KEY` is not set. |
| `OPENROUTER_API_KEY` | API key used by `--provider openrouter` when `OPENAI_API_KEY` is not set. |
| `TOGETHER_API_KEY` | API key used by `--provider together` when `OPENAI_API_KEY` is not set. |
| `DEEPINFRA_API_KEY` | API key used by `--provider deepinfra` when `OPENAI_API_KEY` is not set. |
| `CODEX_HOME` | Directory containing `auth.json`; defaults to `~/.codex`. |
| `CODEX_AUTH_JSON_PATH` | Explicit path to Codex `auth.json`. |
| `LLMRUN_PROVIDER` | Force `openai` or `codex` when compatible with available auth. |
| `LLMRUN_STATE_DIR` | Override where config, sessions, templates, and fragments are stored. |
| `LLMRUN_BASE_URL` | Base URL for OpenAI-compatible providers when using API-key auth. |
| `LLMRUN_CODEX_RESPONSES_URL` | Override the Codex Responses backend URL. Mostly for testing. |
| `LLMRUN_OPENAI_REFRESH_URL` | Override the OAuth refresh URL. Mostly for testing. |

Secrets are not printed by the CLI or included in error messages.

## Quick Start

Run a prompt:

```console
llmrun "Explain Python decorators in three bullets"
```

Read the prompt from stdin:

macOS/Linux:

```sh
cat notes.txt | llmrun -
```

Windows PowerShell:

```powershell
Get-Content .\notes.txt | llmrun -
```

Use a system instruction:

```console
llmrun --system "You are a pirate." "Why is the sky blue?"
```

Use a specific model:

```console
llmrun --model gpt-5.4-mini "Write a regex for ISO dates"
```

Use an OpenAI-compatible provider:

```console
llmrun --base-url https://provider.example/v1 --model provider-model "Explain Coulombs laws"
```

Choose a provider for one command:

```console
llmrun --provider openai "Give me the value of PI to 10 digits"
llmrun --provider groq "Give me the value of PI to 10 digits"
llmrun --provider nvidia "Give me the value of PI to 10 digits"
llmrun --provider sambanova "Give me the value of PI to 10 digits"
llmrun --provider cerebras "Give me the value of PI to 10 digits"
llmrun --provider openrouter "Give me the value of PI to 10 digits"
llmrun --provider together "Give me the value of PI to 10 digits"
llmrun --provider deepinfra "Give me the value of PI to 10 digits"
llmrun --provider codex "Give me the value of PI to 10 digits"
```

`--provider openai` ignores `LLMRUN_BASE_URL` and configured `base_url` unless `--base-url` is also passed. OpenAI-compatible presets set a base URL and provider default model, and prefer the provider-specific API-key environment variable listed above before falling back to `OPENAI_API_KEY`. NVIDIA/NIM presets send prompt requests to Chat Completions because NVIDIA exposes those hosted model endpoints at `/v1/chat/completions`. `--provider codex` uses Codex OAuth and ignores `OPENAI_API_KEY` and base-url settings.

Provider preset defaults:

| Provider | Base URL | Default model |
| --- | --- | --- |
| `groq` | `https://api.groq.com/openai/v1` | `meta-llama/llama-4-scout-17b-16e-instruct` |
| `nvidia`, `nvidia-nim` | `https://integrate.api.nvidia.com/v1` | `openai/gpt-oss-120b` |
| `sambanova` | `https://api.sambanova.ai/v1` | `gpt-oss-120b` |
| `cerebras` | `https://api.cerebras.ai/v1` | `gpt-oss-120b` |
| `openrouter` | `https://openrouter.ai/api/v1` | `openai/gpt-oss-120b:free` |
| `together` | `https://api.together.xyz/v1` | `openai/gpt-oss-120b` |
| `deepinfra` | `https://api.deepinfra.com/v1/openai` | `deepseek-ai/DeepSeek-V3` |

For example, with Groq:

macOS/Linux:

```sh
export OPENAI_API_KEY="gsk_..."
export LLMRUN_BASE_URL="https://api.groq.com/openai/v1"
llmrun --model openai/gpt-oss-120b "Explain Coulombs law"
llmrun --image receipt.webp --model meta-llama/llama-4-scout-17b-16e-instruct "Extract the text"
llmrun --provider groq --file report.pdf "Summarize the action items"
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY = "gsk_..."
$env:LLMRUN_BASE_URL = "https://api.groq.com/openai/v1"
llmrun --model openai/gpt-oss-120b "Explain Coulombs law"
llmrun --image receipt.webp --model meta-llama/llama-4-scout-17b-16e-instruct "Extract the text"
llmrun --provider groq --file report.pdf "Summarize the action items"
```

When the configured base URL is Groq and no prompt model is set, `llmrun` defaults prompt requests to `meta-llama/llama-4-scout-17b-16e-instruct`. Pass `--model` or set `config default_model` to use a different Groq model. Groq supports image recognition/OCR through prompt image inputs, but it does not expose OpenAI-compatible image generation or image editing endpoints.

Disable streaming:

```console
llmrun --no-stream "Give me a summary Shakespeares Merchant of Venice in 1 page"
```

Ask for JSON output:

```console
llmrun --json "Return a prettified JSON object with name, risks, and next_steps"
```

Attach an image or file to a normal prompt:

```console
llmrun --image screenshot.png "Describe what you see"
llmrun --file report.pdf "Summarize the action items"
llmrun --image https://example.com/screenshot.png "What is visible here?"
```

Local attachments are sent as base64 data URLs. HTTPS URLs are passed through without reading local files. Session transcripts store attachment metadata only, not base64 payloads. OpenAI API-key auth and Codex OAuth auth both send attachments as structured Responses input; Codex OAuth support depends on the Codex backend accepting that payload shape for the selected model/account. For OpenAI-compatible `--base-url` providers that do not support Responses multimodal input, image prompts fall back to Chat Completions vision format. Local PDFs also fall back by rendering pages as images for vision-capable models; remote PDF URLs still require a provider that supports Responses `input_file`.

## Multimodal Commands

Audio transcription prints text to stdout:

```console
llmrun --provider groq audio transcribe meeting.wav
```

Audio translation prints English text to stdout:

```console
llmrun --provider groq audio translate french.mp3
```

Speech generation writes an audio file:

```console
llmrun --provider groq audio speech "Build complete" --voice troy --output done.wav
```

OpenAI defaults are `gpt-4o-transcribe` for transcription, `whisper-1` for translation, `gpt-4o-mini-tts` for speech, `alloy` for speech voice, and `.mp3` for speech output. With OpenAI-compatible providers, pass the provider's audio model names to `--model`. The configured `--base-url`, `config base_url`, or `LLMRUN_BASE_URL` is used for these audio SDK calls too. When the base URL is Groq (`https://api.groq.com/openai/v1`), omitted audio models default to `whisper-large-v3-turbo` for transcription, `whisper-large-v3` for translation, and `canopylabs/orpheus-v1-english` for speech. Groq speech defaults to voice `troy`, requires one of `autumn`, `diana`, `hannah`, `austin`, `daniel`, or `troy`, and writes `.wav` because Groq requires `response_format=wav`. Groq audio commands use `/audio/transcriptions`, `/audio/translations`, and `/audio/speech` through the configured base URL.

Image generation and editing write image files:

```console
llmrun images generate "A compact architecture diagram" --output diagram.png
llmrun images generate "Generate a logo direction for an ice cream brand" --count 3 --output logo.png
llmrun images edit --image diagram.png "Use blue labels" --output diagram-edited.png
```

The image default model is `gpt-image-1`; override it with `--model` or `llmrun config set image_model MODEL`. OpenAI API-key auth uses the Images API with GPT Image models. Codex OAuth uses the Responses `image_generation` hosted tool instead, so `gpt-image-*` image model names are mapped to the normal default Responses model for that provider. When `--count` is greater than 1, files are written with numbered suffixes such as `logo-1.png`, `logo-2.png`, and `logo-3.png`.

Video generation submits a Sora render job, polls until completion, then downloads the MP4:

```console
llmrun videos generate "A short product demo" --model sora-2-pro --output demo.mp4
llmrun videos generate "Animate this first frame" --model sora-2 --image frame.png --seconds 8 --size 1280x720 --output demo.mp4
```

Set `config video_model` if you do not want to pass `--model` each time. Video generation does not run inside prompt sessions.

## Sessions

Named sessions store the provider, model, effective base URL, latest response id, timestamps, and a compact transcript. The CLI interface is the same for all providers, but adapters handle continuation differently:

- Official OpenAI API-key auth without a base URL sends `previous_response_id` on the next turn.
- Codex OAuth, OpenAI-compatible base-url providers, and provider/base-url changes replay the compact local transcript into the next request.

Start or continue a session:

```console
llmrun --session work "Draft a plan for a Hello World app in Java"
llmrun --session work "Now turn that into a checklist"
```

Start an interactive chat loop:

```console
llmrun chat --session work
```

Exit chat with `exit`, `quit`, Ctrl+D on macOS/Linux, or Ctrl+Z followed by Enter on Windows.

List sessions:

```console
llmrun sessions list
```

Show a session as Markdown:

```console
llmrun sessions show work
```

Export a transcript:

```console
llmrun sessions export work --format markdown
llmrun sessions export work --format json
```

Delete a session:

```console
llmrun sessions delete work
```

## Templates

Templates are saved prompt text with `{prompt}` substitution. `$prompt` is also supported for compatibility, but `{prompt}` is safer because many shells can expand `$prompt` before `llmrun` receives it.

Add a template:

```console
llmrun templates add bug "Find the likely bug in this code: {prompt}"
```

List templates:

```console
llmrun templates list
```

Show a template:

```console
llmrun templates show bug
```

Delete a template:

```console
llmrun templates delete bug
```

Run a template:

```console
llmrun templates run bug "function returns None for valid input"
```

Run a template in a session:

```console
llmrun templates run bug "crashes during import" --session work
```

## Fragments

Fragments are reusable instruction snippets. They are appended to the system instructions for a prompt.

Add a fragment:

```console
llmrun fragments add terse "Be concise. Prefer concrete recommendations."
llmrun fragments add security "Flag any potential security issues"
```

Use a fragment:

```console
llmrun --fragment terse "Review this README"
```

Use multiple fragments:

```console
llmrun --fragment terse --fragment security "Review this auth code"
```

List, show, or delete fragments:

```console
llmrun fragments list
llmrun fragments show terse
llmrun fragments delete terse
llmrun fragments delete security
```

## Config

Config values are stored locally in the `llmrun` state directory.

The built-in default model is `gpt-5.4-mini`. It is intentionally shared by the OpenAI API-key provider and the Codex OAuth provider so the same command behaves the same way regardless of auth scheme. Override it only when you know the target provider and account support the model you choose.

Set the default model:

```console
llmrun config set default_model gpt-5.4-mini
```

Set a base URL for an OpenAI-compatible provider:

```console
llmrun config set base_url https://provider.example/v1
llmrun config set default_model provider-model
```

Read a config value:

```console
llmrun config get default_model
```

Remove a config value:

```console
llmrun config unset default_model
```

Run `llmrun -h` or `llmrun COMMAND -h` for command-specific help.

Model selection order is:

1. `--model`
2. existing session model
3. `config default_model`
4. built-in default, `gpt-5.4-mini`

Base URL selection order is:

1. `--base-url`
2. `config base_url`
3. `LLMRUN_BASE_URL`
4. OpenAI SDK default

## Common Options

These options apply to root prompts:

| Option | Description |
| --- | --- |
| `-S, --system TEXT` | System instruction text. |
| `-m, --model TEXT` | Model override for this request. |
| `-b, --base-url URL` | Base URL for OpenAI-compatible providers when using API-key auth. |
| `-p, --provider NAME` | Provider preset or auth mode for one command. |
| `-s, --session NAME` | Continue or create a named session. |
| `--fragment NAME` | Add a stored fragment. Can be repeated. |
| `-i, --image PATH_OR_URL` | Attach an image to a root prompt. Can be repeated. |
| `-f, --file PATH_OR_URL` | Attach a file to a root prompt. Can be repeated. |
| `--image-detail auto|low|high` | Detail level for attached images. Defaults to `auto`. |
| `-t, --temperature FLOAT` | Sampling temperature passed to the provider. |
| `-n, --max-output-tokens INTEGER` | Maximum output tokens passed when supported. The Codex OAuth backend currently rejects this field, so that adapter strips it. |
| `--reasoning-effort TEXT` | Reasoning effort value passed as `reasoning.effort`. |
| `--stream / --no-stream` | Stream output when supported. Streaming is on by default. |
| `-j, --json` | Request JSON-object output mode. |

Common subcommand aliases include `-o, --output` for generated file paths, `-m, --model` for model selection, `-i, --image` for image inputs, `-v, --voice` for speech voices, and `-n, --count` for image counts.

## State Files

By default, state is stored in the OS-specific user data directory for `llmrun`. Typically this is:

```text
Windows: C:\Users\<Username>\AppData\Local\llmrun
macOS: /Users/<Username>/Library/Application Support/llmrun
Linux: /home/<Username>/.local/share/llmrun (adhering to the XDG Base Directory Specification)
```
The implementation calls `platformdirs.user_data_dir("llmrun", appauthor=False)` so Windows does not create a duplicated `llmrun\llmrun` path.

You can also set `LLMRUN_STATE_DIR` to use a specific directory like this:

macOS/Linux:

```sh
export LLMRUN_STATE_DIR=/tmp/llmrun-state
```

Windows PowerShell:

```powershell
$env:LLMRUN_STATE_DIR = "C:\tmp\llmrun-state"
```

State includes:

```text
config.json
sessions/*.json
templates/*.txt
fragments/*.txt
```

Session names, template names, and fragment names are stored with collision-safe encoded file names. The default state directory and environment variable names use `llmrun`; old project-name state directories and environment variables are not migrated or read.

## Providers

### OpenAI

The OpenAI provider is used when an API key is available. It calls:

```python
client.responses.create(...)
```

For OpenAI-compatible providers, set `OPENAI_API_KEY` to the provider key and pass `--base-url`, set `config base_url`, or set `LLMRUN_BASE_URL`. The base URL is passed to the OpenAI SDK client and is not used by the Codex OAuth provider. Use `--provider openai`, an OpenAI-compatible preset such as `groq`, `nvidia`, `nvidia-nim`, `sambanova`, `cerebras`, `openrouter`, `together`, or `deepinfra`, or `--provider codex` to override provider selection for one command. Presets set the provider base URL, prefer provider-specific API-key environment variables, and choose a provider-specific prompt default model. Text, audio, image generation, image editing, and video commands use the same configured SDK client when the provider implements those endpoints, and root `--base-url` applies to those subcommands. Text and image input prompts first try Responses format, then fall back to Chat Completions when a base-url provider rejects Responses with an unsupported or not-found response. NVIDIA/NIM prompt requests use Chat Completions directly. Local PDF file prompts also fall back by rendering pages as images for vision-capable models. Groq supports image recognition/OCR, but not OpenAI-compatible image generation or editing endpoints.

Named sessions store each response id and effective base URL. Official OpenAI sessions without a base URL pass `previous_response_id` on the next turn. Codex OAuth, OpenAI-compatible base-url providers, and sessions continued after a provider/base-url change replay the local transcript instead of sending a stale response id.

### Codex OAuth

The Codex OAuth provider is a best-effort fallback for machines already logged in with Codex. It reads local OAuth tokens from Codex `auth.json`, refreshes an expired access token when a refresh token is available, then posts to the Codex Responses backend.

To keep CLI arguments consistent with the OpenAI provider, the Codex adapter normalizes requests before sending them upstream:

- Adds default `instructions` when no `--system` value is supplied.
- Converts prompt text into the structured Responses `input` list shape.
- Forces upstream `stream: true`, then parses SSE output back into normal CLI output.
- Sets `store: false`.
- Removes unsupported fields such as `previous_response_id` and `max_output_tokens`.
- Sends `ChatGPT-Account-ID` when it can be read from Codex auth.
- Continues named sessions by replaying the local transcript instead of using server-side response state.

Current limitations:

- Attachment input uses the structured Responses payload shape for Codex OAuth as well as OpenAI API-key auth.
- Codex OAuth image generation and image editing use the Responses `image_generation` hosted tool and write the returned base64 artifact locally.
- Audio workflows and video generation require API-key auth. Codex OAuth returns a clean unsupported-provider error for those artifact-generation paths.
- The Codex backend requires streaming upstream. `llmrun` handles that inside the provider; `--stream` and `--no-stream` still control what the CLI prints.
- Codex backend compatibility is less stable than the official OpenAI provider.
- If token refresh fails, re-run `codex login`.

## Development

Run tests:

```console
pytest
```

Run formatting, lint, and type checks:

```console
python -m ruff check src tests
python -m black --check src tests
python -m mypy src tests
```

Measure test coverage:

```console
python -m coverage run --source=src/llmrun -m pytest
python -m coverage report
```

In restricted sandboxes where the default temp directory is not writable, point pytest at a workspace-local temp directory.

macOS/Linux:

```sh
mkdir -p .pytest-runtime
TMPDIR="$(pwd)/.pytest-runtime" pytest --basetemp .pytest-runtime/basetemp
```

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path .pytest-runtime
$env:TMP = (Resolve-Path ".pytest-runtime").Path
$env:TEMP = $env:TMP
pytest --basetemp .pytest-runtime\basetemp
```

The automated tests mock providers and HTTP clients. They do not make real OpenAI or Codex network calls.

## Scope

This is v1. It does not implement tools, embeddings, evals, plugins, or video remix/list/delete commands.
