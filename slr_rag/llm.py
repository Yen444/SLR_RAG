"""Create chat / embedding models from the named entries in settings.yaml (any LangChain provider)."""

import os

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from .config import Settings

# Env vars that hold the API key, per provider (any one is enough). Providers not listed need no key.
API_KEY_ENV = {
    "openai": ["OPENAI_API_KEY"],
    "google_genai": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "mistralai": ["MISTRAL_API_KEY"],
    "groq": ["GROQ_API_KEY"],
}


def chat_model(settings: Settings, name: str | None = None) -> BaseChatModel:
    spec = settings.chat_models[name or settings.chat_model]
    return init_chat_model(spec.model, model_provider=spec.provider, **spec.params)


def embeddings(settings: Settings) -> Embeddings:
    spec = settings.embedding_models[settings.embedding_model]
    return init_embeddings(spec.model, provider=spec.provider, **spec.params)


def missing_key(provider: str) -> str | None:
    """Name of the env var to set if the provider's API key is missing, else None."""
    names = API_KEY_ENV.get(provider, [])
    if names and not any(os.environ.get(n) for n in names):
        return names[0]
    return None
