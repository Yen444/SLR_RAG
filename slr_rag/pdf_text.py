"""Extract text per page from a PDF, cached to data/text/<paper_id>.json."""

import json

import pymupdf

from .catalog import Paper
from .config import Settings


def get_pages(paper: Paper, settings: Settings) -> list[str]:
    cache = settings.text_dir / f"{paper.paper_id}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))["pages"]
    with pymupdf.open(paper.path) as doc:
        pages = [page.get_text() for page in doc]
    settings.text_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"paper_id": paper.paper_id, "pages": pages}), encoding="utf-8")
    return pages


def full_text(pages: list[str], max_chars: int) -> str:
    """Join pages with [Page N] markers so the model can cite pages."""
    text = "\n\n".join(f"[Page {i}]\n{p.strip()}" for i, p in enumerate(pages, 1))
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[... text truncated ...]"
    return text
