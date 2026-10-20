"""Smoke test of the Streamlit UI (no OpenAI/Gemini calls, no PDF.js download)."""

import json

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


def _app(tmp_path, monkeypatch, files=("p0.pdf", "p1.pdf")):
    papers = tmp_path / "papers"
    for i, name in enumerate(files):
        (papers / name).parent.mkdir(parents=True, exist_ok=True)
        make_pdf(papers / name, [f"content of paper {i}", "second page"], title=f"Paper number {i} about things")
    papers.mkdir(exist_ok=True)
    settings_file = tmp_path / "settings.yaml"
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
    next(s for s in at.selectbox if s.label == "Exclusion reason").set_value("duplicate").run()
    next(t for t in at.text_input if t.label == "Reason (optional)").set_value("same data as another paper").run()
    next(t for t in at.text_area if t.label == "Reason note (optional)").set_value("see paper 2").run()
    assert not at.exception
    (ann,) = _annotations(tmp_path)
    assert ann["decision"] == "exclude" and ann["exclusion_reason"] == "duplicate"
    assert ann["reason_text"] == "same data as another paper" and ann["reason_note"] == "see paper 2"


def test_extraction_form_saves(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    next(s for s in at.selectbox if s.label == "Study design").set_value("RCT")
    next(t for t in at.text_area if t.label == "Key findings").set_value("It works.")
    next(b for b in at.button if b.label == "💾 Save").click().run()
    assert not at.exception
    fields = _annotations(tmp_path)[0]["fields"]
    assert fields["study_design"] == "RCT" and fields["key_findings"] == "It works."


def test_paper_list_and_prev_next(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch, files=("a.pdf", "b.pdf", "c.pdf"))
    list_buttons = [b for b in at.button if (b.key or "").startswith("list_")]
    assert len(list_buttons) == 3
    assert _header_title(at) == "**Paper number 0 about things**"
    list_buttons[2].click().run()
    assert _header_title(at) == "**Paper number 2 about things**"
    at.button(key="next_1").click().run()  # wraps around
    assert _header_title(at) == "**Paper number 0 about things**"
    at.button(key="prev_1").click().run()
    assert _header_title(at) == "**Paper number 2 about things**"


def test_paper_list_grouped_by_folder(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch, files=("top.pdf", "included/x.pdf", "excluded/y.pdf"))
    labels = [e.label for e in at.expander]
    assert any(l.startswith("📁 (top level)") for l in labels)
    assert any(l.startswith("📁 excluded") for l in labels) and any(l.startswith("📁 included") for l in labels)
    assert not at.exception


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
    assert url == f"/app/static/pdfjs/web/viewer.html?file=%2Fapp%2Fstatic%2Fpdf%2F{paper.paper_id}.pdf"
    assert (viewer.PDF_DIR / f"{paper.paper_id}.pdf").exists()
    jump = viewer.viewer_url(paper, page=3, search="randomised trial", nonce=7)
    assert jump.endswith("#page=3&search=randomised%20trial&phrase=true&n=7")
