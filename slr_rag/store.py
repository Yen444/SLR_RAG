"""Per-paper annotations saved as data/annotations/<paper_id>.json."""

import json
import os
import tempfile
from datetime import datetime
from pathlib import PurePosixPath

import pandas as pd

from .catalog import Paper
from .config import ReviewConfig, Settings


def default_annotation(paper: Paper, review: ReviewConfig) -> dict:
    return {
        "paper_id": paper.paper_id,
        "filename": paper.filename,
        "title": paper.title,
        "status": review.statuses[0],
        "decision": review.decisions[0],
        "exclusion_reason": None,  # one of review.exclusion_reasons
        "exclusion_note": "",  # explanation of the decision, optional
        "relation_note": "",  # how this paper relates to its linked papers, optional
        "fields": {},
        "ai_suggestions": {},  # field -> {"value", "evidence", "page"} from the last "Fill with AI"
        "ai_model": None,  # chat_models key that produced ai_suggestions
        "linked_papers": [],  # paper_ids (stay valid if a PDF is renamed)
        "linked_files": [],  # the same papers by file name without .pdf, e.g. google_scholar_62_included
        "updated_at": None,
    }


def _path(paper_id: str, settings: Settings):
    return settings.annotations_dir / f"{paper_id}.json"


def get(paper: Paper, settings: Settings, review: ReviewConfig) -> dict:
    """Saved annotation merged over defaults (so new keys appear for old files)."""
    ann = default_annotation(paper, review)
    path = _path(paper.paper_id, settings)
    if path.exists():
        ann.update(json.loads(path.read_text(encoding="utf-8")))
        if "reason_note" in ann:  # older files
            ann["exclusion_note"] = ann["exclusion_note"] or ann.pop("reason_note")
        ann.pop("chat_history", None)  # chats are no longer saved
        ann["filename"], ann["title"] = paper.filename, paper.title
    return ann


def save(ann: dict, settings: Settings) -> None:
    """Write to a temp file then rename, so a crash never leaves a half-written file."""
    ann["updated_at"] = datetime.now().isoformat(timespec="seconds")
    settings.annotations_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=settings.annotations_dir, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(ann, f, indent=2, ensure_ascii=False)
    os.replace(tmp, _path(ann["paper_id"], settings))


def file_stem(filename: str) -> str:
    """'google_scholar/google_scholar_62_included.pdf' -> 'google_scholar_62_included'"""
    return PurePosixPath(filename).stem


def link(a: dict, b: dict, settings: Settings) -> None:
    """Link two papers in both directions."""
    if a["paper_id"] == b["paper_id"]:
        return
    for x, y in ((a, b), (b, a)):
        if y["paper_id"] not in x["linked_papers"]:
            x["linked_papers"].append(y["paper_id"])
            x["linked_files"] = [*x.get("linked_files", []), file_stem(y["filename"])]
            save(x, settings)


def unlink(a: dict, b: dict, settings: Settings) -> None:
    for x, y in ((a, b), (b, a)):
        if y["paper_id"] in x["linked_papers"]:
            x["linked_papers"].remove(y["paper_id"])
            x["linked_files"] = [f for f in x.get("linked_files", []) if f != file_stem(y["filename"])]
            save(x, settings)


def linked_names(ann: dict, filename_of: dict[str, str]) -> str:
    """Linked papers as file names without .pdf, ';'-separated (current names; else the saved ones)."""
    saved = dict(zip(ann.get("linked_papers", []), ann.get("linked_files", [])))
    return ";".join(file_stem(filename_of[pid]) if pid in filename_of else saved.get(pid, pid)
                    for pid in ann.get("linked_papers", []))


def load_all(settings: Settings) -> dict[str, dict]:
    if not settings.annotations_dir.exists():
        return {}
    out = {}
    for path in settings.annotations_dir.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        out[data["paper_id"]] = data
    return out


def field_columns(fields: dict, review: ReviewConfig) -> dict:
    """Export columns for the configured fields: lists joined with ';', plus <name>_specify boxes."""
    out = {}
    for f in review.fields:
        value = fields.get(f.name)
        out[f.name] = ";".join(value) if isinstance(value, list) else value
        if f.specify:
            out[f.specify_name] = fields.get(f.specify_name)
    return out


def export_table(papers: list[Paper], settings: Settings, review: ReviewConfig) -> pd.DataFrame:
    """One row per paper, one column per configured field."""
    saved = load_all(settings)
    filename_of = {p.paper_id: p.filename for p in papers}
    rows = []
    for paper in papers:
        ann = default_annotation(paper, review) | saved.get(paper.paper_id, {})
        row = {
            "paper_id": paper.paper_id,
            "folder": paper.folder,
            "filename": paper.filename,
            "title": paper.title,
            "status": ann["status"],
            "decision": ann["decision"],
            "exclusion_reason": ann.get("exclusion_reason"),
            "exclusion_note": ann.get("exclusion_note") or ann.get("reason_note", ""),
            "relation_note": ann.get("relation_note", ""),
            "linked_papers": linked_names(ann, filename_of),
        }
        rows.append(row | field_columns(ann["fields"], review))
    return pd.DataFrame(rows)
