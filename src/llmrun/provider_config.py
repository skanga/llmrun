from __future__ import annotations

from dataclasses import dataclass

GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"
GROQ_VISION_DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_SPEECH_DEFAULT_VOICE = "troy"
GROQ_SPEECH_VOICES = ("autumn", "diana", "hannah", "austin", "daniel", "troy")


@dataclass(frozen=True)
class ProviderPreset:
    name: str
    base_url: str
    api_key_env: str
    default_model: str
    capabilities: "ProviderCapabilities"
    vision_default_model: str | None = None


@dataclass(frozen=True)
class ProviderCapabilities:
    prompt_endpoint: str = "responses"
    is_groq: bool = False
    speech_default_voice: str | None = None
    speech_voices: tuple[str, ...] = ()
    speech_output_suffix: str | None = None
    supports_images_endpoint: bool = True


GROQ_CAPABILITIES = ProviderCapabilities(
    is_groq=True,
    speech_default_voice=GROQ_SPEECH_DEFAULT_VOICE,
    speech_voices=GROQ_SPEECH_VOICES,
    speech_output_suffix=".wav",
    supports_images_endpoint=False,
)
NVIDIA_CAPABILITIES = ProviderCapabilities(prompt_endpoint="chat")
DEFAULT_CAPABILITIES = ProviderCapabilities()


PROVIDER_PRESETS: dict[str, ProviderPreset] = {
    "groq": ProviderPreset(
        name="groq",
        base_url=GROQ_BASE_URL,
        api_key_env="GROQ_API_KEY",
        default_model=GROQ_DEFAULT_MODEL,
        capabilities=GROQ_CAPABILITIES,
        vision_default_model=GROQ_VISION_DEFAULT_MODEL,
    ),
    "nvidia": ProviderPreset(
        name="nvidia",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key_env="NVIDIA_API_KEY",
        default_model="openai/gpt-oss-120b",
        capabilities=NVIDIA_CAPABILITIES,
    ),
    "nvidia-nim": ProviderPreset(
        name="nvidia-nim",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key_env="NVIDIA_API_KEY",
        default_model="openai/gpt-oss-120b",
        capabilities=NVIDIA_CAPABILITIES,
    ),
    "sambanova": ProviderPreset(
        name="sambanova",
        base_url="https://api.sambanova.ai/v1",
        api_key_env="SAMBANOVA_API_KEY",
        default_model="gpt-oss-120b",
        capabilities=DEFAULT_CAPABILITIES,
    ),
    "cerebras": ProviderPreset(
        name="cerebras",
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        default_model="gpt-oss-120b",
        capabilities=DEFAULT_CAPABILITIES,
    ),
    "openrouter": ProviderPreset(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        default_model="openai/gpt-oss-120b:free",
        capabilities=DEFAULT_CAPABILITIES,
    ),
    "together": ProviderPreset(
        name="together",
        base_url="https://api.together.xyz/v1",
        api_key_env="TOGETHER_API_KEY",
        default_model="openai/gpt-oss-120b",
        capabilities=DEFAULT_CAPABILITIES,
    ),
    "deepinfra": ProviderPreset(
        name="deepinfra",
        base_url="https://api.deepinfra.com/v1/openai",
        api_key_env="DEEPINFRA_API_KEY",
        default_model="deepseek-ai/DeepSeek-V3",
        capabilities=DEFAULT_CAPABILITIES,
    ),
}

OPENAI_COMPATIBLE_PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    name: {
        "base_url": preset.base_url,
        "api_key_env": preset.api_key_env,
        "default_model": preset.default_model,
    }
    for name, preset in PROVIDER_PRESETS.items()
}


def provider_preset(provider_choice: str | None) -> ProviderPreset | None:
    if not provider_choice:
        return None
    return PROVIDER_PRESETS.get(provider_choice)


def provider_names() -> tuple[str, ...]:
    return tuple(PROVIDER_PRESETS)


def normalize_base_url(base_url: str | None) -> str | None:
    return base_url.rstrip("/").lower() if base_url else None


def preset_for_base_url(base_url: str | None) -> ProviderPreset | None:
    normalized = normalize_base_url(base_url)
    if not normalized:
        return None
    for preset in PROVIDER_PRESETS.values():
        if normalized == normalize_base_url(preset.base_url):
            return preset
    if "api.groq.com" in normalized:
        return PROVIDER_PRESETS["groq"]
    return None


def default_model_for_base_url(
    base_url: str | None, *, vision: bool = False
) -> str | None:
    preset = preset_for_base_url(base_url)
    if preset and vision and preset.vision_default_model:
        return preset.vision_default_model
    return preset.default_model if preset else None


def capabilities_for_base_url(base_url: str | None) -> ProviderCapabilities:
    preset = preset_for_base_url(base_url)
    return preset.capabilities if preset else DEFAULT_CAPABILITIES
