from llmrun.provider_config import (
    GROQ_BASE_URL,
    GROQ_DEFAULT_MODEL,
    GROQ_SPEECH_DEFAULT_VOICE,
    GROQ_SPEECH_VOICES,
    OPENAI_COMPATIBLE_PROVIDER_PRESETS,
    capabilities_for_base_url,
    default_model_for_base_url,
    provider_names,
    provider_preset,
)


def test_provider_presets_preserve_existing_values():
    assert provider_preset("groq").base_url == GROQ_BASE_URL
    assert provider_preset("groq").api_key_env == "GROQ_API_KEY"
    assert provider_preset("groq").default_model == GROQ_DEFAULT_MODEL
    assert provider_preset("nvidia").base_url == "https://integrate.api.nvidia.com/v1"
    assert provider_preset("nvidia").api_key_env == "NVIDIA_API_KEY"
    assert provider_preset("nvidia").default_model == "openai/gpt-oss-120b"
    assert (
        provider_preset("nvidia-nim").base_url == "https://integrate.api.nvidia.com/v1"
    )
    assert provider_preset("sambanova").api_key_env == "SAMBANOVA_API_KEY"
    assert provider_preset("cerebras").default_model == "gpt-oss-120b"
    assert provider_preset("openrouter").default_model == "openai/gpt-5.2"
    assert provider_preset("together").api_key_env == "TOGETHER_API_KEY"
    assert (
        provider_preset("deepinfra").base_url == "https://api.deepinfra.com/v1/openai"
    )
    assert sorted(OPENAI_COMPATIBLE_PROVIDER_PRESETS) == [
        "cerebras",
        "deepinfra",
        "groq",
        "nvidia",
        "nvidia-nim",
        "openrouter",
        "sambanova",
        "together",
    ]


def test_base_url_capabilities_preserve_groq_speech_policy():
    caps = capabilities_for_base_url("https://api.groq.com/openai/v1/")

    assert caps.is_groq is True
    assert caps.speech_default_voice == GROQ_SPEECH_DEFAULT_VOICE
    assert caps.speech_voices == GROQ_SPEECH_VOICES
    assert caps.speech_output_suffix == ".wav"
    assert (
        default_model_for_base_url("https://api.groq.com/openai/v1/")
        == GROQ_DEFAULT_MODEL
    )
    assert (
        default_model_for_base_url("https://gateway.api.groq.com/custom")
        == GROQ_DEFAULT_MODEL
    )


def test_unknown_base_url_has_openai_like_capabilities():
    caps = capabilities_for_base_url("https://provider.test/v1")

    assert caps.is_groq is False
    assert caps.speech_default_voice is None
    assert caps.speech_voices == ()
    assert caps.speech_output_suffix is None
    assert default_model_for_base_url("https://provider.test/v1") is None


def test_provider_names_and_empty_choices_are_stable():
    assert "groq" in provider_names()
    assert provider_preset(None) is None
    assert capabilities_for_base_url(None).is_groq is False
