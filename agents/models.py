"""Model provider factory: Ollama (local, dynamic), OpenAI-compatible, Agnes
AI, and Google Gemini. Embeddings are always local (Ollama) per the app's
local-first mandate.
"""

from __future__ import annotations

import ollama
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI

import config


class LocalOnlyModeError(PermissionError):
    """Raised when a remote provider is requested while local-only mode is on."""


def list_ollama_models() -> list[str]:
    """Dynamically list locally-pulled Ollama models. Returns [] if Ollama
    isn't running rather than raising, since this feeds a UI dropdown."""
    try:
        client = ollama.Client(host=config.OLLAMA_BASE_URL)
        return sorted(m.model for m in client.list().models)
    except Exception:
        return []


def build_embeddings(model: str) -> Embeddings:
    return OllamaEmbeddings(model=model, base_url=config.OLLAMA_BASE_URL)


def build_chat_model(provider: str, model: str, local_only: bool = False) -> BaseChatModel:
    if provider == "ollama":
        return ChatOllama(model=model, base_url=config.OLLAMA_BASE_URL)

    if local_only:
        raise LocalOnlyModeError(f"Local-only mode is on; refusing to call remote provider '{provider}'.")

    if provider == "openai_compatible":
        kwargs = {"model": model, "api_key": config.OPENAI_API_KEY, "base_url": config.OPENAI_BASE_URL}
        if model in config.REASONING_EFFORT_MODELS:
            kwargs["reasoning_effort"] = config.REASONING_EFFORT
        return ChatOpenAI(**kwargs)

    if provider == "agnes":
        return ChatOpenAI(model=config.AGNES_MODEL, api_key=config.AGNES_API_KEY, base_url=config.AGNES_BASE_URL)

    if provider == "gemini":
        return ChatGoogleGenerativeAI(model=model, google_api_key=config.GOOGLE_API_KEY)

    raise ValueError(f"Unknown provider: {provider}")


def models_for_provider(provider: str) -> list[str]:
    if provider == "ollama":
        return list_ollama_models()
    if provider == "openai_compatible":
        return config.OPENAI_COMPAT_MODELS
    if provider == "agnes":
        return [config.AGNES_MODEL]
    if provider == "gemini":
        return config.GEMINI_MODELS
    return []


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Rough $ estimate from config.PRICING_USD_PER_1M. 0.0 for local/unpriced models."""
    rates = config.PRICING_USD_PER_1M.get(model)
    if not rates:
        return 0.0
    return (input_tokens / 1_000_000) * rates["input"] + (output_tokens / 1_000_000) * rates["output"]
