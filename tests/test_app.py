"""Smoke test of the Streamlit UI (no OpenAI/Gemini calls, no PDF.js download)."""

import json

import pandas as pd
import pytest
import yaml
from streamlit.testing.v1 import AppTest

from slr_rag import viewer
from slr_rag.config import ROOT

from .conftest import MODELS, make_pdf


@pytest.fixture(autouse=True)
def _isolate_viewer(tmp_path, monkeypatch):
    monkeypatch.setattr(viewer, "ensure_pdfjs", lambda: None)
    monkeypatch.setattr(viewer, "PDF_DIR", tmp_path / "static_pdf")


def _app(tmp_path, monkeypatch, files=("p0.pdf", "p1.pdf"), csvs=None):
    """csvs: {"science_direct/results.csv": [row dicts]}, written inside the papers folder."""
    papers = tmp_path / "papers"
    for i, name in enumerate(files):
        (papers / name).parent.mkdir(parents=True, exist_ok=True)
        make_pdf(papers / name, [f"content of paper {i}", "second page"], title=f"Paper number {i} about things")
    papers.mkdir(exist_ok=True)
    settings_file = tmp_path / "settings.yaml"
    for rel, rows in (csvs or {}).items():
        pd.DataFrame(rows).to_csv(papers / rel, index=False, encoding="utf-8-sig")
    settings_file.write_text(yaml.safe_dump({"papers_dir": str(papers), "data_dir": str(tmp_path / "data"), **MODELS}))
    monkeypatch.setenv("SLR_SETTINGS", str(settings_file))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def _annotations(tmp_path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in (tmp_path / "data" / "annotations").glob("*.json")]


def _header(at, slot=1) -> tuple[str, str]:
    """(title, file name) shown in the header bar above the PDF."""
    import re

    value = next(h.proto.body for h in at.html if 'class="paper-head"' in h.proto.body)
    title, file = re.search(r'<b title="[^"]*">(.*?)</b><span>(.*?)</span>', value).groups()
    return title, file


def _header_title(at, slot=1) -> str:
    return f"**{_header(at, slot)[0]}**"


def test_app_renders_and_saves_status(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    next(s for s in at.selectbox if s.label == "Status").set_value("done").run()
    assert not at.exception
    assert _annotations(tmp_path)[0]["status"] == "done"


def test_decision_reason_fields_save(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    reason = next(s for s in at.selectbox if s.label == "Exclusion reason")
    assert reason.disabled  # only for the exclude decision
    next(s for s in at.selectbox if s.label == "Decision").set_value("exclude").run()
    next(s for s in at.selectbox if s.label == "Exclusion reason").set_value("other").run()
    next(t for t in at.text_input if t.label == "Exclusion note").set_value("same data as another paper").run()
    assert not any(t.label == "Reason (optional)" for t in at.text_input)
    assert not at.exception
    (ann,) = _annotations(tmp_path)
    assert ann["decision"] == "exclude" and ann["exclusion_reason"] == "other"
    assert ann["exclusion_note"] == "same data as another paper"
    next(s for s in at.selectbox if s.label == "Decision").set_value("include").run()  # changed back: saved again
    (ann,) = _annotations(tmp_path)
    assert ann["decision"] == "include" and ann["exclusion_reason"] is None


def test_related_paper_from_another_database(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch, files=("gs/a.pdf", "sd/b.pdf", "sd/c.pdf"))
    assert any(c.value == "**Related papers (0)**" for c in at.caption)  # the list is shown even when empty
    assert any(c.value == "None yet." for c in at.caption)
    assert not any(s.label == "Paper" for s in at.selectbox)  # pickers only after "Add related paper"
    next(b for b in at.button if b.label == "Add related paper").click().run()
    paper = next(s for s in at.selectbox if s.label == "Paper")
    assert paper.disabled  # choose the database first
    next(s for s in at.selectbox if s.label == "Database").set_value("sd").run()
    paper = next(s for s in at.selectbox if s.label == "Paper")
    assert not paper.disabled and len(paper.options) == 2
    paper.set_value(paper.options[0]).run()  # the option is the paper's title
    assert not at.exception
    links = {a["filename"]: a["linked_papers"] for a in _annotations(tmp_path)}
    assert len(links["gs/a.pdf"]) == 1 and len(links["sd/b.pdf"]) == 1  # both directions
    assert any(c.value == "**Related papers (1)**" for c in at.caption)  # the chosen paper appears in the list
    assert not any(c.value == "None yet." for c in at.caption)
    assert not any(s.label == "Paper" for s in at.selectbox)  # pickers closed again
    assert any(c.value in ("b", "c") for c in at.caption)  # the file name, next to the title
    next(b for b in at.button if b.label == "Add related paper").click().run()
    next(s for s in at.selectbox if s.label == "Database").set_value("sd").run()
    assert len(next(s for s in at.selectbox if s.label == "Paper").options) == 1  # linked one is not offered again
    assert not any(t.label == "Relation note" for t in at.text_input)  # box only after the button
    next(b for b in at.button if b.label == "Add relation note").click().run()
    next(t for t in at.text_input if t.label == "Relation note").set_value("extends the same simulator").run()
    ann = next(a for a in _annotations(tmp_path) if a["filename"] == "gs/a.pdf")
    assert ann["relation_note"] == "extends the same simulator"
    assert not any(t.label == "Relation note" for t in at.text_input)  # closed after saving
    assert any(c.value == "extends the same simulator" for c in at.caption)  # shown as text
    assert any(b.label == "Edit relation note" for b in at.button)


def test_extraction_form_saves(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    labels = [e.label for e in at.expander]
    assert any(l.startswith("**A. General information** · 0/5") for l in labels)
    sections = [l.split("**")[1] for l in labels if l.startswith("**")]
    assert sections[:5] == ["A. General information", "B. Input", "C. Output", "D. Methodology", "E. Configurability"]
    next(s for s in at.selectbox if s.label == "A1. Application domain").set_value("Domain-specific")
    next(t for t in at.text_input if t.label == "Domain (if domain-specific)").set_value("healthcare")
    next(m for m in at.multiselect if m.label == "A3. Target application").set_value(["Process discovery"])
    next(t for t in at.text_area if t.label == "D1. Core mechanism").set_value("Discrete-event simulation.")
    next(b for b in at.button if b.label == "💾 Save").click().run()
    assert not at.exception
    fields = _annotations(tmp_path)[0]["fields"]
    assert fields["application_domain"] == "Domain-specific" and fields["application_domain_specify"] == "healthcare"
    assert fields["target_application"] == ["Process discovery"]
    assert fields["core_mechanism"] == "Discrete-event simulation."
    assert any(l.startswith("**A. General information** · 2/5") for l in (e.label for e in at.expander))


def _list_rows(at) -> list:
    return [b for b in at.button if (b.key or "").startswith("list_")]


def _list_titles(at) -> list[str]:
    return [b.label.split(". ", 1)[1] for b in _list_rows(at)]


def _open_db(at, db="_top"):
    """Click a database row ("_top" = PDFs directly in papers/)."""
    at.button(key=f"db_{db}").click().run()
    return at


def _db_rows(at) -> list[str]:
    return [b.label for b in at.button if (b.key or "").startswith("db_")]


def test_databases_listed_and_opened_on_click(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch, files=("sd/a.pdf", "gs/b.pdf", "gs/c.pdf"))
    assert _db_rows(at) == ["gs · 2", "sd · 1"] and not _list_rows(at)
    assert _header_title(at)  # a paper is open anyway
    _open_db(at, "gs")
    assert len(_list_rows(at)) == 2
    assert at.segmented_control(key="status_view").options[0] == "All · 2"  # counts for the open database
    _open_db(at, "gs")  # click again: closed
    assert not _list_rows(at)
    at.text_input(key="list_query").set_value("number").run()  # search with no database open: all databases
    assert len(_list_rows(at)) == 3


def test_paper_list_and_prev_next(tmp_path, monkeypatch):
    at = _open_db(_app(tmp_path, monkeypatch, files=("a.pdf", "b.pdf", "c.pdf")))
    assert _list_titles(at) == [f"Paper number {i} about things" for i in range(3)]
    assert _header_title(at) == "**Paper number 0 about things**"
    _list_rows(at)[2].click().run()
    assert _header_title(at) == "**Paper number 2 about things**"
    assert [b.proto.type for b in _list_rows(at)] == ["tertiary", "tertiary", "primary"]  # open paper highlighted
    at.button(key="next_1").click().run()  # wraps around
    assert _header_title(at) == "**Paper number 0 about things**"
    at.button(key="prev_1").click().run()
    assert _header_title(at) == "**Paper number 2 about things**"


def test_subfolders_belong_to_their_database(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch, files=("top.pdf", "sd/x.pdf", "sd/pdfs/y.pdf"))
    assert _db_rows(at) == ["(top level) · 1", "sd · 2"]
    _open_db(at, "sd")
    assert [c.value for c in at.caption if c.value.startswith("📁")] == ["📁 sd", "📁 sd/pdfs"]
    assert not at.exception


def test_status_counts_filter_list(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch, files=("a.pdf", "b.pdf", "c.pdf"))
    view = at.segmented_control(key="status_view")
    assert view.options == ["All · 3", "In progress · 0", "Done · 0", "Excluded · 0"]
    next(s for s in at.selectbox if s.label == "Status").set_value("done").run()
    view = at.segmented_control(key="status_view")
    assert view.options == ["All · 3", "In progress · 0", "Done · 1", "Excluded · 0"]
    view.set_value("done").run()
    assert _db_rows(at) == ["(top level) · 1"]
    _open_db(at)
    assert _list_titles(at) == ["Paper number 0 about things"]
    at.segmented_control(key="status_view").set_value("in_progress").run()
    assert not _list_rows(at) and any("No papers match" in c.value for c in at.caption)
    at.segmented_control(key="status_view").set_value(None).run()
    at.multiselect(key="decision_filter").set_value(["include"]).run()
    assert len(_list_rows(at)) == 3


def test_compare_mode(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    at.toggle(key="compare").set_value(True).run()
    assert not at.exception
    assert at.button(key="prev_2") and at.radio(key="open_side")


def test_model_picker_and_key_warning(tmp_path, monkeypatch):
    for env in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr("slr_rag.config.load_dotenv", lambda *a, **k: None)  # ignore the real .env
    at = _app(tmp_path, monkeypatch)
    picker = at.selectbox(key="chat_model_name")
    assert picker.value == "gem" and picker.options == ["gem (gemini-2.5-flash)", "gpt (gpt-5-mini)"]
    assert [w.value for w in at.warning] == ["`GOOGLE_API_KEY` is not set. Add it to `.env` to use AI features."]
    picker.set_value("gpt").run()
    assert len(at.warning) == 2  # OpenAI for chat + Google for embeddings


def test_empty_folder(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch, files=())
    assert any("No PDFs found" in i.value for i in at.info)


def test_viewer_url(settings):
    make_pdf(settings.papers_dir / "a.pdf", ["x"])
    from slr_rag.catalog import scan_papers

    (paper,) = scan_papers(settings)
    url = viewer.viewer_url(paper)
    assert url == f"/app/static/pdfjs/web/viewer.html?v=nolabels&file=%2Fapp%2Fstatic%2Fpdf%2F{paper.paper_id}.pdf"
    assert (viewer.PDF_DIR / f"{paper.paper_id}.pdf").exists()
    jump = viewer.viewer_url(paper, page=3, search="randomised trial", nonce=7)
    assert jump.endswith("#page=3&search=randomised%20trial&phrase=true&n=7")


def test_pdfjs_shows_physical_page_numbers(tmp_path, monkeypatch):
    web = tmp_path / "pdfjs" / "web"
    web.mkdir(parents=True)
    (web / "viewer.html").write_text('<script src="viewer.mjs" type="module"></script>')
    (web / "viewer.mjs").write_text("  disablePageLabels: {\n    value: false,\n  },\n    disablePageLabels: false,\n")
    monkeypatch.setattr(viewer, "PDFJS_DIR", tmp_path / "pdfjs")
    viewer._disable_page_labels()  # ensure_pdfjs itself is stubbed in this module
    js = (web / "viewer.mjs").read_text()
    assert "false" not in js and js.count("true") == 2
    assert 'src="viewer.mjs?v=nolabels"' in (web / "viewer.html").read_text()  # bypass the browser cache


def test_csv_mode_lists_only_screened_papers_and_excluded_view(tmp_path, monkeypatch):
    sd = [
        {"item_id": "sd_1", "title": "Screened paper one", "author": "Nai, Roberto", "year": "2025",
         "doi": "10.1/x", "pdf_filename": "sd_1_included.pdf", "note": "check method"},
        {"item_id": "sd_2", "title": "Screened paper two", "author": "", "year": "", "doi": "",
         "pdf_filename": "", "note": ""},
        {"item_id": "sd_9", "title": "Not downloaded", "author": "", "year": "", "doi": "",
         "pdf_filename": "sd_9_included.pdf", "note": ""},
    ]
    gs = [{"item_id": "gs_1", "title": "Scholar paper", "pdf_filename": "gs_1_included.pdf"}]
    at = _app(tmp_path, monkeypatch, csvs={"sd/results.csv": sd, "gs/results.csv": gs},
              files=("sd/sd_1_included.pdf", "sd/sd_2.pdf", "sd/unrelated.pdf", "gs/gs_1_included.pdf"))
    assert _db_rows(at) == ["gs · 1", "sd · 2"]  # unrelated.pdf is not in the CSV
    _open_db(at, "sd")
    assert _list_titles(at) == ["Screened paper one", "Screened paper two"]  # titles from the CSV
    _list_rows(at)[0].click().run()
    assert _header(at) == ("Screened paper one", "sd_1_included.pdf")  # title from the CSV + file name only
    assert any("1 CSV row(s) without a PDF" in e.label for e in at.expander)

    next(s for s in at.selectbox if s.label == "Decision").set_value("exclude").run()
    assert at.segmented_control(key="status_view").options[0] == "All · 1"  # in the open database
    assert at.segmented_control(key="status_view").options[-1] == "Excluded · 1"
    at.segmented_control(key="status_view").set_value("__excluded__").run()
    assert _list_titles(at) == ["Screened paper one · exclude"]
    reviewed = pd.read_csv(tmp_path / "data" / "reviewed" / "sd" / "results_reviewed.csv", dtype=str)
    assert reviewed.set_index("item_id").loc["sd_1", "final_decision"] == "excluded"  # updated copy kept current
    assert not at.exception


def _fake_ai(monkeypatch):
    result = {"main_goal": {"value": "Simulate logs.", "evidence": "we simulate logs", "page": 2},
              "target_application": {"value": ["Other (specify)"], "specify": "teaching", "evidence": "x", "page": 1}}
    monkeypatch.setattr("slr_rag.extraction.suggest", lambda *a, **k: result)


def _ai_saved(tmp_path) -> list[dict]:
    return [a.get("ai_suggestions") for a in _annotations(tmp_path)] if (tmp_path / "data" / "annotations").exists() else []


def test_fill_with_ai_is_only_saved_with_save(tmp_path, monkeypatch):
    _fake_ai(monkeypatch)
    at = _app(tmp_path, monkeypatch)
    next(b for b in at.button if b.label == "✨ Fill with AI").click().run()
    assert not at.exception
    assert not any(_ai_saved(tmp_path))  # nothing on disk yet
    assert next(t for t in at.text_area if t.label == "A2. Main goal").value == "Simulate logs."
    assert next(t for t in at.text_input if t.label == "Other (specify)").value == "teaching"
    assert any(e.label == "🔍 Show evidence in the PDF" for e in at.expander)  # quotes usable before saving
    next(b for b in at.button if b.label == "💾 Save").click().run()
    (ann,) = _annotations(tmp_path)
    assert ann["fields"]["main_goal"] == "Simulate logs."
    assert ann["ai_suggestions"]["main_goal"]["page"] == 2 and ann["ai_model"] == "gem"


def test_fill_with_ai_discard_saves_nothing(tmp_path, monkeypatch):
    _fake_ai(monkeypatch)
    at = _app(tmp_path, monkeypatch)
    next(b for b in at.button if b.label == "✨ Fill with AI").click().run()
    next(b for b in at.button if b.label == "Discard unsaved").click().run()
    assert not any(_ai_saved(tmp_path))
    assert next(t for t in at.text_area if t.label == "A2. Main goal").value == ""
    assert not any(e.label == "🔍 Show evidence in the PDF" for e in at.expander)


def test_similar_papers_listed_with_title_and_file_name(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch, files=("gs/google_scholar_5_included.pdf", "sd/science_direct_4_included.pdf"))
    pid = at.session_state["slot1"]
    catalog = json.loads((tmp_path / "data" / "catalog.json").read_text(encoding="utf-8"))
    other = next(e["paper_id"] for e in catalog.values() if e["paper_id"] != pid)
    at.session_state[f"similar_{pid}"] = [{"paper_id": other, "score": 0.83, "page": None, "snippet": "abstract text"}]
    at.run()
    assert not at.exception
    assert any(c.value == "science_direct_4_included" for c in at.caption)  # file name next to the title
    assert any(m.value == "Paper number 1 about things" for m in at.markdown)
    assert any(c.value == "0.83" for c in at.caption)
    at.button(key=f"link_sim_1_{pid}_{other}").click().run()
    assert any(c.value == "🔗" for c in at.caption)
    assert any(m.value == "🔗 Paper number 1 about things" for m in at.markdown)  # listed in the decision box
    ann = next(a for a in _annotations(tmp_path) if a["paper_id"] == pid)
    assert ann["linked_files"] == ["science_direct_4_included"]
