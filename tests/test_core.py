import json

import pytest
from pydantic import ValidationError

from slr_rag import store
from slr_rag.catalog import scan_papers
from slr_rag.config import ReviewConfig, load_review, load_settings
from slr_rag.extraction import _normalize, build_model, field_guide
from slr_rag.pdf_text import full_text, get_pages

from .conftest import make_pdf


def test_shipped_config_is_valid():
    load_settings()
    review = load_review()
    assert review.fields


def test_select_without_options_is_rejected():
    with pytest.raises(ValidationError, match="no options"):
        ReviewConfig(statuses=["a"], decisions=["b"], fields=[{"name": "x", "type": "select"}])


def test_unknown_field_type_is_rejected():
    with pytest.raises(ValidationError):
        ReviewConfig(statuses=["a"], decisions=["b"], fields=[{"name": "x", "type": "date"}])


def test_sections_and_specify_boxes():
    review = ReviewConfig(statuses=["a"], decisions=["b"], sections=[
        {"title": "A. One", "fields": [{"name": "x", "type": "select", "options": ["p", "Other"], "specify": "Other"}]},
        {"title": "B. Two", "fields": [{"name": "y", "type": "longtext", "height": 300}]},
    ])
    assert [(f.name, f.section) for f in review.fields] == [("x", "A. One"), ("y", "B. Two")]
    assert review.fields[0].specify_name == "x_specify"
    assert set(build_model(review).model_json_schema()["$defs"]["Answer_x"]["properties"]) == {
        "value", "specify", "evidence", "page"}
    assert store.field_columns({"x": "Other", "x_specify": "mine", "y": None}, review) == {
        "x": "Other", "x_specify": "mine", "y": None}
    with pytest.raises(ValidationError, match="duplicate"):
        ReviewConfig(statuses=["a"], decisions=["b"], fields=[
            {"name": "x", "specify": "Other"}, {"name": "x_specify"}])


def test_done_status_defaults_to_last(review):
    assert review.done_status == "done"


def test_paper_id_is_stable_across_rename(settings):
    make_pdf(settings.papers_dir / "a.pdf", ["hello world"], title="A Study of Something Important")
    (first,) = scan_papers(settings)
    assert first.title == "A Study of Something Important"
    (settings.papers_dir / "a.pdf").rename(settings.papers_dir / "renamed.pdf")
    (second,) = scan_papers(settings)
    assert second.paper_id == first.paper_id
    assert second.filename == "renamed.pdf"


def test_folder(settings):
    (settings.papers_dir / "sub" / "deeper").mkdir(parents=True)
    make_pdf(settings.papers_dir / "top.pdf", ["a"])
    make_pdf(settings.papers_dir / "sub" / "deeper" / "x.pdf", ["b"])
    folders = {p.filename: p.folder for p in scan_papers(settings)}
    assert folders == {"top.pdf": "", "sub/deeper/x.pdf": "sub/deeper"}


def test_duplicate_files_listed_once(settings):
    make_pdf(settings.papers_dir / "a.pdf", ["same"])
    (settings.papers_dir / "copy.pdf").write_bytes((settings.papers_dir / "a.pdf").read_bytes())
    assert len(scan_papers(settings)) == 1


def test_page_text_and_markers(settings):
    make_pdf(settings.papers_dir / "a.pdf", ["first page", "second page"])
    (paper,) = scan_papers(settings)
    pages = get_pages(paper, settings)
    assert "second page" in pages[1]
    assert (settings.text_dir / f"{paper.paper_id}.json").exists()
    text = full_text(pages, 10_000)
    assert "[Page 2]" in text
    assert full_text(pages, 10).endswith("[... text truncated ...]")


def test_annotation_round_trip_and_links(settings, review):
    make_pdf(settings.papers_dir / "a.pdf", ["paper a"])
    make_pdf(settings.papers_dir / "b.pdf", ["paper b"])
    a, b = scan_papers(settings)

    ann_a = store.get(a, settings, review)
    assert ann_a["status"] == "todo" and ann_a["decision"] == "undecided"
    ann_a["fields"]["design"] = "RCT"
    ann_a["status"] = "done"
    store.save(ann_a, settings)

    reloaded = store.get(a, settings, review)
    assert reloaded["fields"]["design"] == "RCT" and reloaded["updated_at"]

    store.link(reloaded, store.get(b, settings, review), settings)
    assert store.get(a, settings, review)["linked_papers"] == [b.paper_id]
    assert store.get(b, settings, review)["linked_papers"] == [a.paper_id]

    store.unlink(store.get(a, settings, review), store.get(b, settings, review), settings)
    assert store.get(b, settings, review)["linked_papers"] == []

    df = store.export_table([a, b], settings, review)
    assert df["design"][0] == "RCT" and df["design"].isna()[1]


def test_fields_removed_from_config_are_kept(settings, review):
    make_pdf(settings.papers_dir / "a.pdf", ["x"])
    (paper,) = scan_papers(settings)
    ann = store.get(paper, settings, review)
    ann["fields"]["old_field"] = "keep me"
    store.save(ann, settings)
    data = json.loads((settings.annotations_dir / f"{paper.paper_id}.json").read_text(encoding="utf-8"))
    assert data["fields"]["old_field"] == "keep me"


def test_extraction_schema(review):
    model = build_model(review)
    schema = model.model_json_schema()
    assert set(schema["properties"]) == {"design", "n", "methods", "findings"}
    parsed = model.model_validate({
        "design": {"value": "RCT", "evidence": "a randomised trial", "page": 2},
        "n": {"value": None, "evidence": None, "page": None},
        "methods": {"value": ["survey"], "evidence": "we surveyed", "page": 3},
        "findings": {"value": "it works", "evidence": "results show", "page": 5},
    })
    assert parsed.model_dump()["design"]["value"] == "RCT"
    assert "enum" not in json.dumps(schema)  # options are in the prompt; Gemini rejects big enum schemas


def test_extraction_prompt_and_option_normalizing(review):
    guide = field_guide(review)
    assert "- design (Design, select) Options (one): RCT | cohort." in guide
    assert "Options (any number): survey | interview." in guide
    methods = next(f for f in review.fields if f.name == "methods")
    assert _normalize(methods, ["Survey ", "survey", "focus group"]) == ["survey", "focus group"]
    assert _normalize(next(f for f in review.fields if f.name == "design"), "rct") == "RCT"


def test_old_reason_note_is_read_as_exclusion_note(settings, review):
    make_pdf(settings.papers_dir / "a.pdf", ["x"])
    (paper,) = scan_papers(settings)
    settings.annotations_dir.mkdir(parents=True)
    (settings.annotations_dir / f"{paper.paper_id}.json").write_text(
        json.dumps({"paper_id": paper.paper_id, "reason_note": "old note", "reason_text": "old"}), encoding="utf-8")
    ann = store.get(paper, settings, review)
    assert ann["exclusion_note"] == "old note" and "reason_note" not in ann
    assert store.export_table([paper], settings, review)["exclusion_note"][0] == "old note"
