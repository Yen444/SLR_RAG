import pytest
from pydantic import ValidationError

from slr_rag import index, llm
from slr_rag.config import Settings

from .conftest import MODELS


def test_builds_configured_providers(settings, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    assert type(llm.chat_model(settings)).__name__ == "ChatGoogleGenerativeAI"
    assert type(llm.chat_model(settings, "gpt")).__name__ == "ChatOpenAI"
    assert type(llm.embeddings(settings)).__name__ == "GoogleGenerativeAIEmbeddings"


def test_params_are_passed(settings, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    settings.chat_models["gem"].params = {"temperature": 0.2}
    assert llm.chat_model(settings, "gem").temperature == 0.2


def test_unknown_default_model_is_rejected(tmp_path):
    with pytest.raises(ValidationError, match="not in chat_models"):
        Settings(**{**MODELS, "chat_model": "nope"})


def test_missing_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    assert llm.missing_key("google_genai") is None
    monkeypatch.delenv("GEMINI_API_KEY")
    assert llm.missing_key("google_genai") == "GOOGLE_API_KEY"
    assert llm.missing_key("ollama") is None


def test_one_index_per_embedding_model(settings):
    assert index.collection_name(settings) == "papers-emb"
