from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderConfig:
    """Provider configuration shared across agents and evaluation judges.

    Supported providers:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Normalize provider name and handle common aliases."""
    val = (value or "").strip().lower()
    if val in {"openai", "open-ai"}:
        return "openai"
    if val in {"custom", "openai-compatible", "local-openai", "vllm", "local"}:
        return "custom"
    if val in {"gemini", "google", "google-genai", "google_genai"}:
        return "gemini"
    if val in {"anthropic", "anthorpic", "claude"}:
        return "anthropic"
    if val in {"ollama", "local-ollama"}:
        return "ollama"
    if val in {"openrouter", "open-router"}:
        return "openrouter"
    return val


def build_chat_model(config: ProviderConfig) -> Any:
    """Instantiate a chat model for the selected provider."""
    provider = normalize_provider(config.provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key or "dummy-key",
            base_url=config.base_url,
        )

    if provider == "custom":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key or "custom-key",
            base_url=config.base_url or "http://localhost:8000/v1",
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=config.api_key or "dummy-key",
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model_name=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key or "dummy-key",
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            base_url=config.base_url or "http://localhost:11434",
        )

    if provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        return ChatOpenRouter(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key or "dummy-key",
        )

    raise ValueError(f"Unsupported provider: {config.provider}")
