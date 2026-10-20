"""Index + similar-paper search with fake embeddings (no OpenAI calls)."""

import pytest
from langchain_core.embeddings import Embeddings

from slr_rag import index
from slr_rag.catalog import scan_papers
from slr_rag.extraction import build_model

from .conftest import make_pdf

VOCAB = ["cat", "dog", "pet", "rocket", "orbit", "space", "soil", "crop"]


class BagOfWords(Embeddings):
    """Tiny deterministic embedding so similar texts get similar vectors."""

    def __init__(self, **_):
        pass

    def _embed(self, text):
        words = text.lower().split()
        return [words.count(w) + 0.01 for w in VOCAB]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


@pytest.fixture
def papers(settings, monkeypatch):
    monkeypatch.setattr(index.llm, "embeddings", lambda s: BagOfWords())
    make_pdf(settings.papers_dir / "pets1.pdf", ["cat dog pet cat", "dog pet"])
    make_pdf(settings.papers_dir / "pets2.pdf", ["pet cat dog dog"])
    make_pdf(settings.papers_dir / "space.pdf", ["rocket orbit space rocket"])
    make_pdf(settings.papers_dir / "empty.pdf", [""])
    return {p.filename: p for p in scan_papers(settings)}


def test_index_is_incremental(settings, papers):
    n, no_text = index.index_papers(list(papers.values()), settings)
    assert n == 3 and [p.filename for p in no_text] == ["empty.pdf"]
    n, _ = index.index_papers(list(papers.values()), settings)
    assert n == 0


def test_index_adds_missing_paper_vectors(settings, papers):
    """Papers indexed before paper-level vectors existed get them on the next run."""
    pets = [papers["pets1.pdf"]]
    index.index_papers(pets, settings)
    index.get_paper_store(settings).delete(ids=[pets[0].paper_id])
    assert index.index_papers(pets, settings)[0] == 1
    assert index.get_paper_store(settings).get(ids=[pets[0].paper_id])["ids"] == [pets[0].paper_id]


def test_extract_abstract():
    pages = ["Journal X\nA Title\nAbstract: We study cats. " + "Cats are great. " * 20
             + "\n1. Introduction\nLong intro text"]
    abstract = index.extract_abstract(pages)
    assert abstract.startswith("We study cats.") and "intro" not in abstract.lower()
    assert index.extract_abstract(["No abstract heading here"]) == "No abstract heading here"


def test_similar_and_search(settings, papers):
    index.index_papers(list(papers.values()), settings)

    similar = index.similar_to_paper(papers["pets1.pdf"], settings)
    assert similar[0]["paper_id"] == papers["pets2.pdf"].paper_id
    assert papers["pets1.pdf"].paper_id not in {r["paper_id"] for r in similar}
    assert [r["score"] for r in similar] == sorted((r["score"] for r in similar), reverse=True)
    assert len(similar) == 2  # every other paper with text, one entry each

    hits = index.search("rocket space", settings)
    assert hits[0]["paper_id"] == papers["space.pdf"].paper_id
    assert len({h["paper_id"] for h in hits}) == len(hits)  # one row per paper

    with pytest.raises(LookupError):
        index.similar_to_paper(papers["empty.pdf"], settings)


def test_extraction_schema_is_openai_strict_compatible(review):
    # The converter the OpenAI SDK applies to pydantic response formats; raises if not strict-compatible.
    from openai.lib._pydantic import to_strict_json_schema

    schema = to_strict_json_schema(build_model(review))
    design = schema["$defs"]["Answer_design"]
    assert set(design["required"]) == {"value", "evidence", "page"}
