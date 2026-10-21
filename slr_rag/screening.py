"""Screening results from an earlier review step: CSV files with one row per included paper.

The CSVs live inside papers_dir, usually one sub-folder per database with its CSV and PDFs:
    papers/science_direct/results.csv, papers/science_direct/science_direct_4_included.pdf, ...
A row is linked to a PDF in the CSV's folder (or below it) by its `pdf_filename` column, or else by
`item_id` (file name <item_id>.pdf, <item_id>_included.pdf or <item_id>_excluded.pdf). When there are
CSV files, only linked papers are reviewed. The CSVs are never modified; `export` writes a new table.
"""

import dataclasses
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import pandas as pd

from . import store
from .catalog import Paper
from .config import ReviewConfig, Settings

ID, PDF, TITLE, ABSTRACT, DECISION = "item_id", "pdf_filename", "title", "abstract", "final_decision"
FOLDER = "source_folder"  # added column: the CSV's folder relative to papers_dir ('' for the top level)
_DECISION_SUFFIX = re.compile(r"_(included|excluded)$", re.IGNORECASE)


@dataclass
class Scope:
    papers: list[Paper]  # PDFs listed in a CSV, with title/abstract from the CSV
    rows: dict[str, int] = field(default_factory=dict)  # paper_id -> row index in the records table
    missing: list[dict] = field(default_factory=list)  # CSV rows without a PDF
    hidden: list[Paper] = field(default_factory=list)  # PDFs not listed in any CSV


def csv_files(settings: Settings) -> list[Path]:
    d = settings.papers_dir
    return sorted(d.rglob("*.csv"), key=lambda p: str(p).lower()) if d.is_dir() else []


def _rel_folder(path: Path, settings: Settings) -> str:
    folder = path.parent.relative_to(settings.papers_dir).as_posix()
    return "" if folder == "." else folder


def load_records(settings: Settings) -> pd.DataFrame:
    """All CSV rows as text ('' for empty cells), plus source_csv and source_folder columns.

    An item_id repeated within one folder is dropped (the first row is kept)."""
    frames = [
        pd.read_csv(f, encoding="utf-8-sig", dtype=str, keep_default_na=False).assign(
            source_csv=f.relative_to(settings.papers_dir).as_posix(), **{FOLDER: _rel_folder(f, settings)})
        for f in csv_files(settings)
    ]
    if not frames:
        return pd.DataFrame()
    records = pd.concat(frames, ignore_index=True).fillna("")
    if ID in records:
        repeated = records[ID].ne("") & records.duplicated([FOLDER, ID])
        records = records[~repeated].reset_index(drop=True)
    return records


def _name(path: str) -> str:
    return PurePosixPath(path.replace("\\", "/")).name.lower()


def _folder_and_parents(folder: str) -> list[str]:
    """'a/b' -> ['a/b', 'a', '']: a PDF can be matched by a CSV in its own folder or any folder above it."""
    parts = folder.split("/") if folder else []
    return ["/".join(parts[:n]) for n in range(len(parts), -1, -1)]


def match(papers: list[Paper], records: pd.DataFrame) -> dict[str, int]:
    """paper_id -> row index, by pdf_filename first, then by item_id. Each row is used at most once."""
    by_name, by_id = {}, {}
    for i, row in records.iterrows():
        if row.get(PDF):
            by_name.setdefault((row[FOLDER], _name(row[PDF])), i)
        if row.get(ID):
            by_id.setdefault((row[FOLDER], row[ID].lower()), i)
    out, used = {}, set()
    for paper in papers:
        name = _name(paper.filename)
        stem = _DECISION_SUFFIX.sub("", PurePosixPath(paper.filename).stem).lower()
        for folder in _folder_and_parents(paper.folder):
            i = by_name.get((folder, name), by_id.get((folder, stem)))
            if i is not None:
                break
        if i is not None and i not in used:
            out[paper.paper_id] = i
            used.add(i)
    return out


def scope(papers: list[Paper], records: pd.DataFrame) -> Scope:
    rows = match(papers, records)
    in_scope = []
    for paper in papers:
        if paper.paper_id in rows:
            row = records.loc[rows[paper.paper_id]]
            in_scope.append(dataclasses.replace(
                paper, title=row.get(TITLE) or paper.title, abstract=row.get(ABSTRACT) or ""))
    linked = set(rows.values())
    return Scope(
        papers=in_scope,
        rows=rows,
        missing=[records.loc[i].to_dict() for i in records.index if i not in linked],
        hidden=[p for p in papers if p.paper_id not in rows],
    )


def citation(row: dict) -> str:
    """'Nai et al. 2025 · Journal' from BibTeX-style author/year/journal columns."""
    def surname(author: str) -> str:  # "Nai, Roberto" or "Roberto Nai"
        return author.split(",")[0] if "," in author else author.split()[-1]

    authors = [a.strip() for a in (row.get("author") or "").split(" and ") if a.strip()]
    names = [surname(a) for a in authors]
    first = names[0] + " et al." if len(names) > 2 else " & ".join(names)
    who = " ".join(x for x in (first, row.get("year") or "") if x)
    venue = row.get("journal") or row.get("booktitle") or ""
    return " · ".join(x for x in (who, venue) if x)


def doi_url(row: dict) -> str:
    doi = (row.get("doi") or "").strip()
    return doi if not doi or doi.startswith("http") else f"https://doi.org/{doi}"


def export(records: pd.DataFrame, sc: Scope, papers: list[Paper], settings: Settings,
           review: ReviewConfig) -> pd.DataFrame:
    """The original rows and columns, plus the review and extraction done in this app."""
    saved = store.load_all(settings)
    by_id = {p.paper_id: p for p in papers}
    pid_of = {i: pid for pid, i in sc.rows.items()}
    item_of = {pid: records.loc[i].get(ID, "") for pid, i in sc.rows.items()}
    rows = []
    for i in records.index:
        pid = pid_of.get(i)
        original = records.loc[i].get(DECISION, "")
        if pid not in by_id:  # no PDF
            rows.append({"final_decision_updated": original})
            continue
        ann = saved.get(pid) or store.default_annotation(by_id[pid], review)
        rows.append({
            "final_decision_updated": "excluded" if ann["decision"] == review.exclude_decision else original,
            "review_status": ann["status"],
            "review_decision": ann["decision"],
            "review_exclusion_reason": ann.get("exclusion_reason"),
            "review_reason_text": ann.get("reason_text", ""),
            "review_reason_note": ann.get("reason_note", ""),
            "review_notes": ann.get("notes", ""),
            "linked_item_ids": ";".join(item_of.get(l) or l for l in ann.get("linked_papers", [])),
        } | store.field_columns(ann["fields"], review))
    columns = ["final_decision_updated", "review_status", "review_decision", "review_exclusion_reason",
               "review_reason_text", "review_reason_note", "review_notes", "linked_item_ids",
               *store.field_columns({}, review)]
    added = pd.DataFrame(rows, index=records.index, columns=columns).astype(object).fillna("")
    out = records.copy()
    for name in columns:
        out[name if name not in out else f"extract_{name}"] = added[name]  # never overwrite an original column
    return out
