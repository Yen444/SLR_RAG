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


def _header_title(at, slot=1) -> str:
    return next(m.value for m in at.markdown if m.value.startswith("**Paper number"))


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
    next(t for t in at.text_input if t.label == "Reason (optional)").set_value("same data as another paper").run()
    next(t for t in at.text_input if t.label == "Reason note (optional)").set_value("see paper 2").run()
    assert not at.exception
    (ann,) = _annotations(tmp_path)
    assert ann["decision"] == "exclude" and ann["exclusion_reason"] == "other"
    assert ann["reason_text"] == "same data as another paper" and ann["reason_note"] == "see paper 2"


def test_extraction_form_saves(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    labels = [e.label for e in at.expander]
    assert any(l.startswith("**A. General information** · 0/5") for l in labels)
    assert any(l.startswith("**B. Input**") for l in labels) and any(l.startswith("**H. Additional notes**") for l in labels)
    next(s for s in at.selectbox if s.label == "A1. Application domain").set_value("Domain-specific")
    next(t for t in at.text_input if t.label == "Domain (if domain-specific)").set_value("healthcare")
    next(m for m in at.multiselect if m.label == "A3. Target application").set_value(["Process discovery"])
    next(t for t in at.text_area if t.label == "C1. Core mechanism").set_value("Discrete-event simulation.")
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


def _show_all(at):
    at.segmented_control(key="status_view").set_value("__all__").run()
    return at


def test_list_hidden_until_a_filter_is_picked(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch, files=("a.pdf", "b.pdf"))
    assert not _list_rows(at) and any("to list the papers" in c.value for c in at.caption)
    assert _header_title(at) == "**Paper number 0 about things**"  # a paper is still open
    _show_all(at)
    assert len(_list_rows(at)) == 2
    at.segmented_control(key="status_view").set_value(None).run()  # click the button again: hidden
    assert not _list_rows(at)
    at.multiselect(key="decision_filter").set_value(["include"]).run()
    assert len(_list_rows(at)) == 2


def test_paper_list_and_prev_next(tmp_path, monkeypatch):
    at = _show_all(_app(tmp_path, monkeypatch, files=("a.pdf", "b.pdf", "c.pdf")))
    assert _list_titles(at) == [f"Paper number {i} about things" for i in range(3)]
    assert _header_title(at) == "**Paper number 0 about things**"
    _list_rows(at)[2].click().run()
    assert _header_title(at) == "**Paper number 2 about things**"
    assert [b.proto.type for b in _list_rows(at)] == ["tertiary", "tertiary", "primary"]  # open paper highlighted
    at.button(key="next_1").click().run()  # wraps around
    assert _header_title(at) == "**Paper number 0 about things**"
    at.button(key="prev_1").click().run()
    assert _header_title(at) == "**Paper number 2 about things**"


def test_paper_list_folders_and_search(tmp_path, monkeypatch):
    at = _show_all(_app(tmp_path, monkeypatch, files=("top.pdf", "included/x.pdf", "excluded/y.pdf")))
    assert [c.value for c in at.caption if c.value.startswith("📁")] == ["📁 (top level)", "📁 excluded", "📁 included"]
    at.text_input(key="list_query").set_value("included").run()
    assert len(_list_rows(at)) == 1
    at.text_input(key="list_query").set_value("").run()
    at.multiselect(key="folder_filter").set_value(["excluded", "included"]).run()  # database filter
    assert len(_list_rows(at)) == 2
    assert at.segmented_control(key="status_view").options[0] == "All · 2"  # counts follow the filter
    assert not at.exception


def test_status_counts_filter_list(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch, files=("a.pdf", "b.pdf", "c.pdf"))
    view = at.segmented_control(key="status_view")
    assert view.options == ["All · 3", "In progress · 0", "Done · 0", "Excluded · 0"]
    next(s for s in at.selectbox if s.label == "Status").set_value("done").run()
    view = at.segmented_control(key="status_view")
    assert view.options == ["All · 3", "In progress · 0", "Done · 1", "Excluded · 0"]
    view.set_value("done").run()
    assert _list_titles(at) == ["Paper number 0 about things"]
    at.segmented_control(key="status_view").set_value("in_progress").run()
    assert not _list_rows(at) and any("No papers match" in c.value for c in at.caption)


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
    at = _show_all(_app(tmp_path, monkeypatch, csvs={"sd/results.csv": sd, "gs/results.csv": gs},
                        files=("sd/sd_1_included.pdf", "sd/sd_2.pdf", "sd/unrelated.pdf", "gs/gs_1_included.pdf")))
    # titles from the CSVs, grouped by database folder
    assert _list_titles(at) == ["Scholar paper", "Screened paper one", "Screened paper two"]
    assert [c.value for c in at.caption if c.value.startswith("📁")][:2] == ["📁 gs", "📁 sd"]
    _list_rows(at)[1].click().run()
    assert any("sd_1 · Nai 2025 · [DOI](https://doi.org/10.1/x)" in c.value for c in at.caption)
    assert any("Screening note: check method" in c.value for c in at.caption)
    assert any("1 CSV row(s) without a PDF" in e.label for e in at.expander)

    next(s for s in at.selectbox if s.label == "Decision").set_value("exclude").run()
    assert at.segmented_control(key="status_view").options[0] == "All · 2"
    assert at.segmented_control(key="status_view").options[-1] == "Excluded · 1"
    at.segmented_control(key="status_view").set_value("__excluded__").run()
    assert _list_titles(at) == ["Screened paper one · exclude"]
    assert not at.exception
