import pymupdf
import pytest

from slr_rag.config import ReviewConfig, Settings


def make_pdf(path, pages: list[str], title: str = ""):
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=11)
    if title:
        doc.set_metadata({"title": title})
    doc.save(path)
    doc.close()


MODELS = {
    "chat_model": "gem",
    "embedding_model": "emb",
    "chat_models": {
        "gem": {"provider": "google_genai", "model": "gemini-2.5-flash"},
        "gpt": {"provider": "openai", "model": "gpt-5-mini"},
    },
    "embedding_models": {"emb": {"provider": "google_genai", "model": "gemini-embedding-001"}},
}


@pytest.fixture
def settings(tmp_path):
    s = Settings(papers_dir=tmp_path / "papers", data_dir=tmp_path / "data", **MODELS)
    s.papers_dir.mkdir()
    return s


@pytest.fixture
def review():
    return ReviewConfig(
        statuses=["todo", "done"],
        decisions=["undecided", "include", "exclude"],
        exclusion_reasons=["off topic"],
        fields=[
            {"name": "design", "type": "select", "options": ["RCT", "cohort"]},
            {"name": "n", "type": "number"},
            {"name": "methods", "type": "multiselect", "options": ["survey", "interview"]},
            {"name": "findings", "type": "longtext"},
        ],
    )
