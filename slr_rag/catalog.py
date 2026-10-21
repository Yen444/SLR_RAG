"""Find the PDFs in papers_dir and give each a stable id (hash of the file content)."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pymupdf

from .config import Settings


@dataclass(frozen=True)
class Paper:
    paper_id: str
    filename: str  # path relative to papers_dir, with forward slashes
    path: Path
    title: str
    abstract: str = ""  # from the screening CSV, if any

    @property
    def folder(self) -> str:
        """Sub-folder inside papers_dir ('' for the top level)."""
        parent = PurePosixPath(self.filename).parent.as_posix()
        return "" if parent == "." else parent


def paper_id_for(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:12]


def guess_title(path: Path) -> str:
    """PDF metadata title if it looks real, else the largest-font text on page 1, else the file name."""
    fallback = path.stem
    try:
        with pymupdf.open(path) as doc:
            meta = ((doc.metadata or {}).get("title") or "").strip()
            junk = meta.lower().startswith(("microsoft word", "untitled")) or meta.lower().endswith((".doc", ".docx", ".pdf"))
            if len(meta) > 10 and not junk:
                return meta
            if doc.page_count:
                spans = [
                    s
                    for b in doc[0].get_text("dict")["blocks"]
                    for line in b.get("lines", [])
                    for s in line["spans"]
                    if s["text"].strip()
                ]
                if spans:
                    size = max(s["size"] for s in spans)
                    text = " ".join(s["text"].strip() for s in spans if abs(s["size"] - size) < 0.5)
                    if 10 <= len(text) <= 300:
                        return text
    except Exception:
        pass
    return fallback


def scan_papers(settings: Settings) -> list[Paper]:
    """Scan papers_dir. Hashes and titles are cached in data/catalog.json (keyed by path, size, mtime)."""
    settings.papers_dir.mkdir(parents=True, exist_ok=True)
    cache_path = settings.data_dir / "catalog.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    new_cache, papers, seen = {}, [], set()
    for path in sorted(settings.papers_dir.rglob("*.pdf"), key=lambda p: str(p).lower()):
        rel = path.relative_to(settings.papers_dir).as_posix()
        stat = path.stat()
        entry = cache.get(rel)
        if not entry or entry["size"] != stat.st_size or entry["mtime_ns"] != stat.st_mtime_ns:
            entry = {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "paper_id": paper_id_for(path),
                "title": guess_title(path),
            }
        new_cache[rel] = entry
        if entry["paper_id"] in seen:  # identical file stored twice
            continue
        seen.add(entry["paper_id"])
        papers.append(Paper(entry["paper_id"], rel, path, entry["title"]))

    if new_cache != cache:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(new_cache, indent=1), encoding="utf-8")
    return papers
