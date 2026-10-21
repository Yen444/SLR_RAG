"""Screening results from an earlier review step: CSV files with one row per included paper.

The CSVs live inside papers_dir, usually one sub-folder per database with its CSV and PDFs:
    papers/science_direct/results.csv, papers/science_direct/science_direct_4_included.pdf, ...
A row is linked to a PDF in the CSV's folder (or below it) by its `pdf_filename` column, or else by
`item_id` (file name <item_id>.pdf, <item_id>_included.pdf or <item_id>_excluded.pdf). When there are
CSV files, only linked papers are reviewed. The CSVs are never modified; `export` writes a new table.
"""

import dataclasses
import html
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import pandas as pd

from . import store
from .catalog import Paper
from .config import ReviewConfig, Settings

# Databases name their columns differently (title / Document Title / Item Title, ...). Each piece of
# information is copied into an internal "_<name>" column, found via settings.csv_columns.
ID, PDF, TITLE, NOTE, DECISION = "_item_id", "_pdf_filename", "_title", "_note", "_final_decision"
AUTHORS, YEAR, VENUE, DOI = "_authors", "_year", "_venue", "_doi"
FOLDER = "source_folder"  # added column: the CSV's folder relative to papers_dir ('' for the top level)
_DECISION_SUFFIX = re.compile(r"_(included|excluded)$", re.IGNORECASE)


@dataclass
class Scope:
    papers: list[Paper]  # PDFs listed in a CSV, with the title from the CSV
    rows: dict[str, int] = field(default_factory=dict)  # paper_id -> row index in the records table
    missing: list[dict] = field(default_factory=list)  # CSV rows without a PDF
    hidden: list[Paper] = field(default_factory=list)  # PDFs not listed in any CSV


def csv_files(settings: Settings) -> list[Path]:
    d = settings.papers_dir
    return sorted(d.rglob("*.csv"), key=lambda p: str(p).lower()) if d.is_dir() else []


def _rel_folder(path: Path, settings: Settings) -> str:
    folder = path.parent.relative_to(settings.papers_dir).as_posix()
    return "" if folder == "." else folder


def _key(column: str) -> str:
    return re.sub(r"[\s_]+", "", column).casefold()


def _find_column(columns, candidates: list[str]) -> str | None:
    """The first of `candidates` that is one of `columns` (ignoring case, spaces and underscores)."""
    by_key = {}
    for column in columns:
        by_key.setdefault(_key(column), column)
    return next((by_key[_key(c)] for c in candidates if _key(c) in by_key), None)


def _internal_columns(frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Add the "_<name>" columns, each copied from the first matching column of this CSV ('' if none)."""
    for name, candidates in settings.csv_columns.items():
        column = _find_column(frame.columns, candidates)
        frame[f"_{name}"] = frame[column] if column else ""
    frame[TITLE] = frame[TITLE].map(lambda t: " ".join(html.unescape(t).split()))  # "from&#xa0;Hybrid"
    frame[YEAR] = frame[YEAR].map(lambda y: (re.search(r"\d{4}", y) or [""])[0])  # "2016.0" -> "2016"
    return frame


def internal_columns(settings: Settings) -> list[str]:
    return [f"_{name}" for name in settings.csv_columns]


def load_records(settings: Settings) -> pd.DataFrame:
    """All CSV rows as text ('' for empty cells), plus source_csv, source_folder and the "_<name>" columns.

    An item_id repeated within one folder is dropped (the first row is kept)."""
    frames = [
        _internal_columns(pd.read_csv(f, encoding="utf-8-sig", dtype=str, keep_default_na=False).assign(
            source_csv=f.relative_to(settings.papers_dir).as_posix(), **{FOLDER: _rel_folder(f, settings)}), settings)
        for f in csv_files(settings)
    ]
    if not frames:
        return pd.DataFrame()
    records = pd.concat(frames, ignore_index=True).fillna("")
    repeated = records[ID].ne("") & records.duplicated([FOLDER, ID])
    return records[~repeated].reset_index(drop=True)


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
            in_scope.append(dataclasses.replace(paper, title=row.get(TITLE) or paper.title))
    linked = set(rows.values())
    return Scope(
        papers=in_scope,
        rows=rows,
        missing=[records.loc[i].to_dict() for i in records.index if i not in linked],
        hidden=[p for p in papers if p.paper_id not in rows],
    )


def split_authors(text: str) -> list[str]:
    """Author lists as the databases write them:
    'Nai, Roberto and Sulis, Emilio' (BibTeX), 'S. Lee; I. Choi' (IEEE), 'O Vasilecas, T Savickas…'
    (Google Scholar) and 'Anti AlmanFabrizio Maria Maggi' (Springer, no separator)."""
    text = text.replace("…", "").strip()
    if " and " in text:
        parts = text.split(" and ")
    elif ";" in text:
        parts = text.split(";")
    elif "," in text and all(" " in p.strip() for p in text.split(",")):  # not one BibTeX "Nai, Roberto"
        parts = text.split(",")
    elif "," in text:
        parts = [text]
    else:
        parts = re.split(r"(?<=[a-z])(?=[A-Z])", text)
    return [p.strip() for p in parts if p.strip()]


def citation(row: dict) -> str:
    """'Nai et al. 2025 · Journal'."""
    def surname(author: str) -> str:  # "Nai, Roberto" or "Roberto Nai"
        return author.split(",")[0] if "," in author else author.split()[-1]

    names = [surname(a) for a in split_authors(row.get(AUTHORS) or "")]
    first = names[0] + " et al." if len(names) > 2 else " & ".join(names)
    who = " ".join(x for x in (first, row.get(YEAR) or "") if x)
    return " · ".join(x for x in (who, row.get(VENUE) or "") if x)


def doi_url(row: dict) -> str:
    doi = (row.get(DOI) or "").strip()
    return doi if not doi or doi.startswith("http") else f"https://doi.org/{doi}"


def export(records: pd.DataFrame, sc: Scope, papers: list[Paper], settings: Settings,
           review: ReviewConfig) -> pd.DataFrame:
    """The original rows and columns, plus the review and extraction done in this app."""
    saved = store.load_all(settings)
    by_id = {p.paper_id: p for p in papers}
    pid_of = {i: pid for pid, i in sc.rows.items()}
    filename_of = {p.paper_id: p.filename for p in papers}
    rows = []
    for i in records.index:
        pid = pid_of.get(i)
        original = records.loc[i].get(DECISION, "")
        if pid not in by_id:  # no PDF
            rows.append({"final_decision_updated": original})
            continue
        ann = store.default_annotation(by_id[pid], review) | saved.get(pid, {})
        rows.append({
            "final_decision_updated": "excluded" if ann["decision"] == review.exclude_decision else original,
            "review_status": ann["status"],
            "review_decision": ann["decision"],
            "review_exclusion_reason": ann.get("exclusion_reason"),
            "review_exclusion_note": ann.get("exclusion_note") or ann.get("reason_note", ""),
            "linked_papers": store.linked_names(ann, filename_of),
            "review_relation_note": ann.get("relation_note", ""),
        } | store.field_columns(ann["fields"], review))
    columns = ["final_decision_updated", "review_status", "review_decision", "review_exclusion_reason",
               "review_exclusion_note", "linked_papers", "review_relation_note",
               *store.field_columns({}, review)]
    added = pd.DataFrame(rows, index=records.index, columns=columns).astype(object).fillna("")
    out = records.drop(columns=internal_columns(settings), errors="ignore")
    for name in columns:
        out[name if name not in out else f"extract_{name}"] = added[name]  # never overwrite an original column
    return out


def reviewed_path(source_csv: str, settings: Settings) -> Path:
    """data/reviewed/<database>/<name>_reviewed.csv for papers/<database>/<name>.csv"""
    rel = PurePosixPath(source_csv)
    return settings.data_dir / "reviewed" / rel.parent / f"{rel.stem}_reviewed.csv"


def write_reviewed(records: pd.DataFrame, sc: Scope, papers: list[Paper], settings: Settings,
                   review: ReviewConfig) -> list[str]:
    """Write one up-to-date copy of each screening CSV (the originals are never touched).

    Each copy has the CSV's own columns plus the review and extraction columns. Its decision column
    holds the decision after this review (e.g. 'excluded'); the screening decision is kept in
    `screening_final_decision`. Returns error messages (e.g. a file open in Excel), empty if all went well."""
    table = export(records, sc, papers, settings, review)
    added = [c for c in table.columns if c not in records.columns]
    errors = []
    for source, rows in table.groupby("source_csv", sort=False):
        original = list(pd.read_csv(settings.papers_dir / source, encoding="utf-8-sig", nrows=0).columns)
        rows = rows[original + added].copy()
        updated = rows.pop("final_decision_updated")
        column = _find_column(original, settings.csv_columns["final_decision"]) or "final_decision"
        if column in rows:
            rows.insert(rows.columns.get_loc(column) + 1, "screening_final_decision", rows[column])
        rows[column] = updated
        path = reviewed_path(source, settings)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as f:
                rows.to_csv(f, index=False)
            os.replace(tmp, path)
        except OSError as e:  # PermissionError while the file is open in Excel
            Path(tmp).unlink(missing_ok=True)
            errors.append(f"{path.name}: {e.strerror or e}")
    return errors
