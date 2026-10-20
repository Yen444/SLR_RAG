"""Scan the papers folder and index new papers for similar-paper search.

Usage: uv run python scripts/ingest.py
"""

from slr_rag.catalog import scan_papers
from slr_rag.config import load_settings
from slr_rag.index import index_papers


def main():
    settings = load_settings()
    papers = scan_papers(settings)
    print(f"Found {len(papers)} papers in {settings.papers_dir}")
    n, no_text = index_papers(papers, settings, progress=lambda i, t, p: print(f"  [{i}/{t}] {p.filename}"))
    print(f"Indexed {n} new paper(s).")
    for p in no_text:
        print(f"  WARNING no extractable text (scanned PDF?): {p.filename}")


if __name__ == "__main__":
    main()
