"""Vector indexes (Chroma + configured embeddings), used only to find related papers.

Two collections per embedding model:
- paper level: one vector per paper (title + the words after "Abstract" in the PDF) -> "similar to this paper"
- chunk level: one vector per ~1500-character chunk  -> "search by topic" (finds specific passages)
"""

import re
from collections.abc import Callable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import llm
from .catalog import Paper
from .config import Settings
from .pdf_text import get_pages

# "Abstract" heading at a line start, also spaced ("A B S T R A C T", Elsevier) or followed by ":" "." "—" (IEEE)
_ABSTRACT_START = re.compile(r"(?:^|\n)\s*a\s?b\s?s\s?t\s?r\s?a\s?c\s?t\b\s*[:.\-—–]?\s*", re.IGNORECASE)


def _model_tag(settings: Settings) -> str:
    """Vectors from different embedding models can't be compared, so each model gets its own collections."""
    return re.sub(r"[^A-Za-z0-9._-]", "-", settings.embedding_model)


def collection_name(settings: Settings) -> str:
    return "papers-" + _model_tag(settings)


def paper_collection_name(settings: Settings) -> str:
    return "paperlevel-" + _model_tag(settings)


def _open(settings: Settings, name: str) -> Chroma:
    return Chroma(
        collection_name=name,
        embedding_function=llm.embeddings(settings),
        persist_directory=str(settings.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def get_store(settings: Settings) -> Chroma:
    return _open(settings, collection_name(settings))


def get_paper_store(settings: Settings) -> Chroma:
    return _open(settings, paper_collection_name(settings))


def indexed_ids(store: Chroma) -> set[str]:
    return {m["paper_id"] for m in store.get(include=["metadatas"])["metadatas"]}


def extract_abstract(pages: list[str], words: int = 300) -> str:
    """The `words` words after the 'Abstract' heading on the first three pages (cover pages happen);
    without such a heading, the first `words` words of the paper."""
    head = "\n".join(pages[:3])
    start = _ABSTRACT_START.search(head)
    text = head[start.end():] if start else "\n".join(pages)
    return " ".join(text.split()[:words])


def paper_text(paper: Paper, settings: Settings) -> str:
    """What the paper-level vector represents: title + the words after 'Abstract' in the PDF."""
    return f"{paper.title}\n\n{extract_abstract(get_pages(paper, settings), settings.paper_level_words)}"


def _paper_text_version(settings: Settings) -> str:
    """Stored with each paper-level vector; vectors made another way (or with another length) are re-made."""
    return f"abstract-{settings.paper_level_words}w"


def index_papers(
    papers: list[Paper],
    settings: Settings,
    progress: Callable[[int, int, Paper], None] | None = None,
) -> tuple[int, list[Paper]]:
    """Embed papers missing from either index. Returns (number indexed, papers without extractable text)."""
    chunks, paper_level = get_store(settings), get_paper_store(settings)
    have_chunks = indexed_ids(chunks)
    stored = paper_level.get(include=["metadatas"])
    version = _paper_text_version(settings)
    have_paper = {i for i, m in zip(stored["ids"], stored["metadatas"]) if (m or {}).get("text") == version}
    outdated = [i for i in stored["ids"] if i not in have_paper]
    todo = [p for p in papers if p.paper_id not in have_chunks or p.paper_id not in have_paper]
    splitter = RecursiveCharacterTextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)

    indexed, no_text = 0, []
    for i, paper in enumerate(todo, 1):
        if progress:
            progress(i, len(todo), paper)
        pages = get_pages(paper, settings)
        if not any(p.strip() for p in pages):  # scanned PDF without a text layer
            no_text.append(paper)
            continue
        if paper.paper_id not in have_chunks:
            docs = [
                Document(page_content=chunk, metadata={"paper_id": paper.paper_id, "title": paper.title, "page": n})
                for n, page in enumerate(pages, 1)
                for chunk in splitter.split_text(page)
            ]
            chunks.add_documents(docs, ids=[f"{paper.paper_id}-{k}" for k in range(len(docs))])
        if paper.paper_id not in have_paper:
            if paper.paper_id in outdated:
                paper_level.delete(ids=[paper.paper_id])  # Chroma ignores add() for an existing id
            doc = Document(page_content=paper_text(paper, settings),
                           metadata={"paper_id": paper.paper_id, "title": paper.title, "text": version})
            paper_level.add_documents([doc], ids=[paper.paper_id])
        indexed += 1
    return indexed, no_text


def _group_by_paper(hits: list[tuple[Document, float]], exclude: str | None, k: int) -> list[dict]:
    """Keep the best-scoring chunk per paper. `hits` carry cosine distances; score = 1 - distance."""
    best: dict[str, dict] = {}
    for doc, distance in hits:
        score = 1.0 - distance
        pid = doc.metadata["paper_id"]
        if pid == exclude:
            continue
        if pid not in best or score > best[pid]["score"]:
            best[pid] = {
                "paper_id": pid,
                "title": doc.metadata.get("title", pid),
                "score": round(float(score), 3),
                "page": doc.metadata.get("page"),
                "snippet": doc.page_content[:400],
            }
    return sorted(best.values(), key=lambda r: -r["score"])[:k]


def search(query: str, settings: Settings, exclude: str | None = None, k: int | None = None) -> list[dict]:
    """Papers with a passage that best matches a free-text query (chunk level)."""
    k = k or settings.similar_top_k
    hits = get_store(settings).similarity_search_with_score(query, k=k * 5)
    return _group_by_paper(hits, exclude, k)


def similar_to_paper(paper: Paper, settings: Settings, k: int | None = None) -> list[dict]:
    """Top-k papers by similarity of their title + abstract vectors, highest score first (no API call)."""
    k = k or settings.similar_top_k
    store = get_paper_store(settings)
    vectors = store.get(ids=[paper.paper_id], include=["embeddings"])["embeddings"]
    if vectors is None or len(vectors) == 0:
        raise LookupError("This paper is not indexed yet. Click 'Build / update index' in the sidebar.")
    # Despite its name, this returns cosine distances (lower = closer), like similarity_search_with_score.
    hits = store.similarity_search_by_vector_with_relevance_scores(list(vectors[0]), k=k + 1)
    results = [
        {
            "paper_id": doc.metadata["paper_id"],
            "title": doc.metadata.get("title", doc.metadata["paper_id"]),
            "score": round(1.0 - float(distance), 3),
            "page": None,
            "snippet": doc.page_content.split("\n\n", 1)[-1][:400],  # the abstract
        }
        for doc, distance in hits
        if doc.metadata["paper_id"] != paper.paper_id
    ]
    return sorted(results, key=lambda r: -r["score"])[:k]
