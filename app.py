"""Streamlit UI. Run with: streamlit run app.py"""

import html
import mimetypes

import streamlit as st

from slr_rag import chat, extraction, index, llm, screening, store, viewer
from slr_rag.catalog import scan_papers
from slr_rag.config import FieldDef, load_review, load_settings

mimetypes.add_type("text/javascript", ".mjs")  # PDF.js modules; the Windows registry may not know .mjs

st.set_page_config(page_title="SLR RAG", page_icon="📚", layout="wide")
ss = st.session_state

# Panels are keyed containers (class "st-key-<key>"), styled here so PDF and review areas are clearly apart.
st.html("""
<style>
/* Streamlit's top toolbar is fixed over the page; keep it slim and start content below it */
header[data-testid="stHeader"].stAppHeader, header[data-testid="stHeader"] { height: 2.5rem; min-height: 2.5rem; }
.block-container, [data-testid="stMainBlockContainer"] { padding-top: 3rem; padding-bottom: 1rem; }
[class*="st-key-header_"] {
  background: #fff; border: 1px solid #d5dbe3; border-radius: 10px; padding: 2px 8px; }
[class*="st-key-header_"] button { min-height: 1.9rem; padding: 0 .6rem; }
/* one line: title (cut with … if long, full title on hover), then the file name */
.paper-head { display: flex; align-items: baseline; gap: .6rem; min-width: 0; }
.paper-head b { font-size: .95rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.paper-head span { font-size: .78rem; color: #64748b; white-space: nowrap; flex-shrink: 0; }
[class*="st-key-pdfpanel_"] {
  background: #3c4043; border-radius: 12px; padding: 6px; }
[class*="st-key-reviewpanel_"] {
  background: #fff; border: 1px solid #d5dbe3; border-radius: 12px; padding: 8px 12px 12px;
  box-shadow: 0 1px 4px rgba(15, 23, 42, .06); }
[class*="st-key-decisionbox_"] {
  background: #f6f8fb; border: 1px solid #e2e8f0; border-radius: 10px; padding: 6px 10px 8px; }
/* compact review panel: tighter gaps, smaller labels and inputs */
[class*="st-key-reviewpanel_"] [data-testid="stVerticalBlock"] { gap: .45rem; }
[class*="st-key-reviewpanel_"] [data-testid="stWidgetLabel"] { min-height: 0; margin-bottom: 0; }
[class*="st-key-reviewpanel_"] [data-testid="stWidgetLabel"] p { font-size: .78rem; color: #475569; }
[class*="st-key-reviewpanel_"] input, [class*="st-key-reviewpanel_"] textarea,
[class*="st-key-reviewpanel_"] [data-baseweb="select"] { font-size: .85rem; }
[class*="st-key-reviewpanel_"] [data-baseweb="select"] > div { min-height: 2.1rem; }
[class*="st-key-reviewpanel_"] [data-baseweb="tab"] { height: 2.2rem; }
/* compact sidebar */
[data-testid="stSidebarUserContent"] { padding-top: 0; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .5rem; }
/* paper list: one full-width clickable row per paper */
/* similar-paper results: compact rows */
[class*="st-key-results_"] [data-testid="stVerticalBlock"],
[class*="st-key-links_"] [data-testid="stVerticalBlock"],
[class*="st-key-relnote_"] [data-testid="stVerticalBlock"] { gap: .1rem; }
[class*="st-key-results_"] [data-testid="stMarkdownContainer"] p,
[class*="st-key-links_"] [data-testid="stMarkdownContainer"] p { font-size: .85rem; line-height: 1.3; margin: 0; }
/* database rows: click to open / close that database's paper list */
[class*="st-key-db_"] button, [class*="st-key-db_"] button > div { justify-content: flex-start; min-height: 2.1rem; }
[class*="st-key-db_"] button p { font-size: .85rem; font-weight: 600; }
.st-key-paperlist [data-testid="stVerticalBlock"] { gap: 0; }
.st-key-paperlist button { width: 100%; justify-content: flex-start; text-align: left;
  padding: 3px 8px; min-height: 0; border-radius: 6px; }
.st-key-paperlist button:hover { background: #eef2f7; }
.st-key-paperlist button > div { justify-content: flex-start; min-width: 0; }
.st-key-paperlist button p { font-size: .8rem; line-height: 1.3; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; }
.st-key-paperlist [data-testid="stCaptionContainer"] { padding: 6px 8px 2px; }
</style>
""")

# Config is re-read on every rerun, so edits to config/*.yaml apply without restarting.
try:
    settings = load_settings()
    review = load_review()
except Exception as e:
    st.error(f"Config error: {e}")
    st.stop()


@st.cache_data(show_spinner="Scanning papers…")
def _scan(papers_dir: str, token: int, csv_stamp: tuple):
    """PDFs to review, and the screening CSVs with the PDF <-> row links (None without CSVs)."""
    papers = scan_papers(settings)
    records = screening.load_records(settings)
    if records.empty:
        return papers, None, records
    sc = screening.scope(papers, records)
    return sc.papers, sc, records


@st.cache_resource(show_spinner="Downloading the PDF viewer (one time)…")
def _pdfjs_ready() -> str | None:
    try:
        viewer.ensure_pdfjs()
        return None
    except Exception as e:
        return str(e)


ss.setdefault("scan_token", 0)
ss.setdefault("compare", False)
ss.setdefault("goto_counter", 0)
csv_stamp = tuple((f.name, f.stat().st_mtime_ns) for f in screening.csv_files(settings))
papers, scope, records = _scan(str(settings.papers_dir), ss.scan_token, csv_stamp)
by_id = {p.paper_id: p for p in papers}
saved = store.load_all(settings)


def _sync_reviewed_csvs():
    """Keep data/reviewed/ (an updated copy of each screening CSV) current: rewrite it after any change."""
    files = settings.annotations_dir.glob("*.json") if settings.annotations_dir.exists() else []
    stamp = hash((tuple(sorted((f.name, f.stat().st_mtime_ns) for f in files)), csv_stamp,
                  tuple(f.name for f in review.fields)))
    if ss.get("reviewed_stamp") != stamp:
        ss.reviewed_errors = screening.write_reviewed(records, scope, papers, settings, review)
        ss.reviewed_stamp = None if ss.reviewed_errors else stamp  # retry on the next run if a file was locked


if scope:
    _sync_reviewed_csvs()


def ann_of(pid: str) -> dict:
    """Always read fresh from disk, so both panels see each other's changes."""
    return store.get(by_id[pid], settings, review)


ALL = "__all__"  # "All" button of the status filter: every paper that isn't excluded
EXCLUDED = "__excluded__"  # "Excluded" button: papers excluded in this app


def status_of(pid: str) -> str:
    return saved.get(pid, {}).get("status", review.statuses[0])


def database_of(pid: str) -> str:
    """Top-level folder of papers/ (one per database); '' for PDFs directly in papers/."""
    return by_id[pid].folder.split("/")[0]


def decision_of(pid: str) -> str:
    return saved.get(pid, {}).get("decision", review.decisions[0])


def nice(name: str) -> str:
    return name.replace("_", " ").capitalize()


def label(pid: str, with_folder: bool = False) -> str:
    tag = "" if decision_of(pid) == review.decisions[0] else f" · {decision_of(pid)}"
    folder = f"📁 {by_id[pid].folder} · " if with_folder and by_id[pid].folder else ""
    return f"{folder}{by_id[pid].title}{tag}"


def with_current(options: list, current) -> list:
    """Keep a saved value selectable even if it was removed from the config."""
    return options if current is None or current in options else [*options, current]


# ---------- callbacks (run before the next rerun) ----------

def _rescan():
    ss.scan_token += 1


def _save_attr(pid: str, attr: str, widget_key: str):
    a = ann_of(pid)
    a[attr] = ss[widget_key]
    if attr == "decision" and a["decision"] != review.exclude_decision:
        a["exclusion_reason"] = None
    store.save(a, settings)
    st.toast("Saved ✓")


def _open(slot: int, pid: str):
    ss[f"slot{slot}"] = pid
    if slot == 2:
        ss.compare = True


def _step(slot: int, ordered: list, delta: int):
    current = ss.get(f"slot{slot}")
    i = ordered.index(current) + delta if current in ordered else 0
    ss[f"slot{slot}"] = ordered[i % len(ordered)]


def _link(a_id: str, b_id: str):
    store.link(ann_of(a_id), ann_of(b_id), settings)
    st.toast("Linked ✓")


def _unlink(a_id: str, b_id: str):
    store.unlink(ann_of(a_id), ann_of(b_id), settings)
    st.toast("Unlinked ✓")


def _link_selected(pid: str, widget_key: str, adding_key: str):
    if ss.get(widget_key):
        store.link(ann_of(pid), ann_of(ss[widget_key]), settings)
        ss[widget_key] = None  # ready for the next one
        ss[adding_key] = False  # close the pickers
        st.toast("Linked ✓")


def _toggle(key: str):
    ss[key] = not ss.get(key, False)


def _save_and_close(pid: str, attr: str, widget_key: str, open_key: str):
    _save_attr(pid, attr, widget_key)
    ss[open_key] = False


def _toggle_db(db: str):
    ss.open_db = None if ss.get("open_db") == db else db


def _goto(pid: str, page: int | None, quote: str | None = None):
    """Jump the PDF viewer to a page and highlight the start of a quote."""
    ss.goto_counter += 1
    phrase = " ".join((quote or "").split()[:8]) or None
    ss[f"goto_{pid}"] = {"page": page, "search": phrase, "nonce": ss.goto_counter}


def render_paper_list(pids: list[str], slot: int):
    """One full-width row per paper in a scroll box; clicking anywhere on a row opens the paper."""
    if not pids:
        st.caption("No papers match.")
        return
    open_ids = {ss.get("slot1"), ss.get("slot2") if ss.compare else None}
    show_folders = len({by_id[p].folder for p in pids}) > 1
    folder = None
    with st.container(key="paperlist", height=520 if len(pids) > 14 else "content", border=True):
        for n, pid in enumerate(pids, 1):
            if show_folders and by_id[pid].folder != folder:
                folder = by_id[pid].folder
                st.caption(f"📁 {folder or '(top level)'}")
            st.button(f"{n}. {label(pid)}", key=f"list_{pid}", type="primary" if pid in open_ids else "tertiary",
                      width="stretch", on_click=_open, args=(slot, pid))


# ---------- sidebar ----------

with st.sidebar:
    st.markdown("#### 📚 SLR RAG")

    with st.expander("⚙️ Models & tools"):
        model_names = list(settings.chat_models)
        if ss.get("chat_model_name") not in settings.chat_models:
            ss.pop("chat_model_name", None)  # removed from config
        st.selectbox("Chat model", model_names, index=model_names.index(settings.chat_model),
                     key="chat_model_name", format_func=lambda n: f"{n} ({settings.chat_models[n].model})",
                     help="Used for chat and Fill with AI. Add models under chat_models in config/settings.yaml.")
        emb_spec = settings.embedding_models[settings.embedding_model]
        st.caption(f"Embeddings: {settings.embedding_model} ({emb_spec.model})")
        st.slider("PDF viewer height", 500, 1600, 900, step=50, key="pdf_height")
        st.button("🔄 Rescan papers folder", on_click=_rescan, width="stretch")
        if st.button("🧠 Build / update index", width="stretch",
                     help="Embed new papers for related-paper search. Already indexed papers are skipped."):
            bar = st.progress(0.0, text="Indexing…")
            try:
                n, no_text = index.index_papers(
                    papers, settings, progress=lambda i, t, p: bar.progress(i / t, text=f"{i}/{t}: {p.title[:40]}")
                )
                bar.empty()
                st.success(f"Indexed {n} new paper(s).")
                if no_text:
                    st.warning("No extractable text (scanned PDF?): " + ", ".join(p.filename for p in no_text))
            except Exception as e:
                bar.empty()
                st.error(f"Indexing failed: {e}")
        if scope:
            table = screening.export(records, scope, papers, settings, review)
            st.download_button("⬇️ Export CSV", table.to_csv(index=False).encode("utf-8-sig"),
                               "slr_extraction_export.csv", "text/csv", width="stretch",
                               help="All rows of your screening CSVs plus the review and extraction columns")
        elif papers:
            csv = store.export_table(papers, settings, review).to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇️ Export CSV", csv, "slr_export.csv", "text/csv", width="stretch")
        if scope:
            st.caption(f"Updated copies of your CSVs: `{(settings.data_dir / 'reviewed').relative_to(settings.data_dir.parent)}`")
        if scope and scope.hidden:
            st.caption(f"{len(scope.hidden)} PDF(s) in `{settings.papers_dir.name}/` are not in a CSV and are hidden.")

    # warnings at the top, so they are seen
    for error in ss.get("reviewed_errors", []) if scope else []:
        st.warning(f"Could not update {error}. Close it (e.g. in Excel); it is retried after your next change.")
    if scope and scope.missing:
        with st.expander(f"⚠️ {len(scope.missing)} CSV row(s) without a PDF"):
            st.caption("Put these PDFs next to their CSV and click **Rescan papers folder**.")
            for row in scope.missing:
                where = f"📁 {row[screening.FOLDER]} · " if row.get(screening.FOLDER) else ""
                st.caption(f"{where}{row[screening.ID] or '?'} · {row[screening.TITLE] or row[screening.PDF]}")
    missing = {llm.missing_key(s.provider) for s in (settings.chat_models[ss.chat_model_name], emb_spec)} - {None}
    for env in sorted(missing):
        st.warning(f"`{env}` is not set. Add it to `.env` to use AI features.")

    # Databases (top-level folders of papers/) are listed; clicking one opens its papers.
    # Status buttons, decision and search filter the papers. Status only changes when set by hand.
    databases = sorted({database_of(p.paper_id) for p in papers}, key=str.lower)
    if ss.get("open_db") not in databases:
        ss.open_db = None
    in_db = [p for p in papers if ss.open_db is None or database_of(p.paper_id) == ss.open_db]
    excluded = {p.paper_id for p in papers if decision_of(p.paper_id) == review.exclude_decision}
    active = [p.paper_id for p in in_db if p.paper_id not in excluded]
    counts = {s: sum(status_of(pid) == s for pid in active) for s in review.statuses[1:]}
    n_excluded = sum(p.paper_id in excluded for p in in_db)
    names = ({ALL: f"All · {len(active)}"} | {s: f"{nice(s)} · {n}" for s, n in counts.items()}
             | {EXCLUDED: f"Excluded · {n_excluded}"})
    view = st.segmented_control("Show", list(names), format_func=names.get, key="status_view",
                                label_visibility="collapsed")
    c1, c2 = st.columns([1, 1], gap="small")
    decision_filter = c1.multiselect("Decision", review.decisions, key="decision_filter", placeholder="Decision",
                                     label_visibility="collapsed")
    query = c2.text_input("Search", key="list_query", placeholder="🔎 Search",
                          label_visibility="collapsed").strip().lower()

    def shown(pid: str) -> bool:
        if view == EXCLUDED:
            return pid in excluded
        if view not in (None, ALL) and status_of(pid) != view:
            return False
        # excluded papers only show under "Excluded" or when that decision is filtered for
        return decision_of(pid) in decision_filter if decision_filter else pid not in excluded

    matching = [
        p.paper_id
        for p in sorted(papers, key=lambda p: (p.folder != "", p.folder.lower(), p.filename.lower()))
        if shown(p.paper_id) and (not query or query in p.title.lower() or query in p.filename.lower())
    ]
    # `ordered` (what the list shows, folder then file name) is also what ◀ ▶ step through.
    ordered = [pid for pid in matching if ss.open_db is None or database_of(pid) == ss.open_db]
    # The compare controls are drawn below the list, but the list needs their value now.
    ss.setdefault("open_side", "left")
    target = 2 if ss.compare and ss.open_side == "right" else 1
    if query and ss.open_db is None:  # search across all databases
        render_paper_list(matching, target)
    else:
        for db in databases:
            pids = [pid for pid in matching if database_of(pid) == db]
            is_open = ss.open_db == db
            st.button(f"{db or '(top level)'} · {len(pids)}", key=f"db_{db or '_top'}", width="stretch",
                      type="primary" if is_open else "secondary", on_click=_toggle_db, args=(db,),
                      icon=":material/folder_open:" if is_open else ":material/folder:")
            if is_open:
                render_paper_list(pids, target)

    st.toggle("Compare two papers", key="compare")
    if ss.compare:
        st.radio("Clicking a paper opens it", ["left", "right"], horizontal=True, key="open_side")


if not papers:
    if scope:
        st.info(f"None of the PDFs in `{settings.papers_dir}` is listed in its folder's CSV file. "
                "PDFs are matched by the `pdf_filename` column or by `item_id`.")
    else:
        st.info(f"No PDFs found. Put your papers in `{settings.papers_dir}` and click **Rescan papers folder**.")
    st.stop()


# ---------- panel parts ----------

def field_widget(f: FieldDef, current, key: str):
    kw = {"key": key, "help": f.description or None}
    if f.type in ("text", "longtext") and isinstance(current, list):  # field was a multi-select before
        current = "; ".join(map(str, current))
    if f.type == "text":
        return st.text_input(f.label, value=current or "", placeholder=f.placeholder or None, **kw).strip() or None
    if f.type == "longtext":
        return st.text_area(f.label, value=current or "", height=f.height or 80, placeholder=f.placeholder or None,
                            **kw).strip() or None
    if f.type == "number":
        value = float(current) if isinstance(current, (int, float)) and not isinstance(current, bool) else None
        return st.number_input(f.label, value=value, **kw)
    if f.type == "bool":
        names = {None: "—", True: "Yes", False: "No"}
        opts = [None, True, False]
        return st.selectbox(f.label, opts, index=opts.index(current) if current in opts else 0,
                            format_func=names.get, **kw)
    if f.type == "select":
        opts = [None, *with_current(f.options, current)]
        return st.selectbox(f.label, opts, index=opts.index(current),
                            format_func=lambda x: "—" if x is None else x, **kw)
    if f.type == "multiselect":
        current = current or []
        opts = f.options + [x for x in current if x not in f.options]
        return st.multiselect(f.label, opts, default=current, **kw)
    raise ValueError(f.type)


def render_header(slot: int, pid: str):
    paper = by_id[pid]
    title, file = html.escape(paper.title), html.escape(paper.filename.split("/")[-1])
    with st.container(key=f"header_{slot}"):
        c1, c2, c3 = st.columns([1, 14, 1], vertical_alignment="center")
        c1.button("◀", key=f"prev_{slot}", on_click=_step, args=(slot, ordered, -1), help="Previous paper",
                  disabled=not ordered)
        c2.html(f'<div class="paper-head"><b title="{title}">{title}</b><span>{file}</span></div>')
        c3.button("▶", key=f"next_{slot}", on_click=_step, args=(slot, ordered, 1), help="Next paper",
                  disabled=not ordered)


def render_pdf(slot: int, pid: str, height: int):
    with st.container(key=f"pdfpanel_{slot}"):
        error = _pdfjs_ready()
        if error:
            st.warning(f"Could not download the PDF.js viewer ({error}). Showing the browser's viewer instead.")
            st.iframe(by_id[pid].path, height=height)
            return
        goto = ss.get(f"goto_{pid}") or {}
        st.iframe(viewer.viewer_url(by_id[pid], goto.get("page"), goto.get("search"), goto.get("nonce", 0)),
                  height=height)


def render_decision(slot: int, pid: str, ann: dict):
    with st.container(key=f"decisionbox_{slot}"):
        c1, c2, c3 = st.columns(3)
        statuses = with_current(review.statuses, ann["status"])
        c1.selectbox("Status", statuses, index=statuses.index(ann["status"]), key=f"status_{slot}_{pid}",
                     on_change=_save_attr, args=(pid, "status", f"status_{slot}_{pid}"))
        decisions = with_current(review.decisions, ann["decision"])
        c2.selectbox("Decision", decisions, index=decisions.index(ann["decision"]), key=f"decision_{slot}_{pid}",
                     on_change=_save_attr, args=(pid, "decision", f"decision_{slot}_{pid}"))
        excluding = ann["decision"] == review.exclude_decision
        reasons = [None, *with_current(review.exclusion_reasons, ann["exclusion_reason"])]
        c3.selectbox("Exclusion reason", reasons, index=reasons.index(ann["exclusion_reason"]),
                     format_func=lambda x: "— choose —" if x is None else x, key=f"reason_{slot}_{pid}",
                     disabled=not excluding, on_change=_save_attr, args=(pid, "exclusion_reason", f"reason_{slot}_{pid}"),
                     help=None if excluding else f"Set the decision to '{review.exclude_decision}' to choose a reason")
        st.text_input("Exclusion note", value=ann.get("exclusion_note", ""), key=f"xnote_{slot}_{pid}",
                      placeholder="Details about the decision",
                      on_change=_save_attr, args=(pid, "exclusion_note", f"xnote_{slot}_{pid}"))
        render_links(slot, pid, ann)
        render_relation_note(slot, pid, ann)


def render_relation_note(slot: int, pid: str, ann: dict):
    """The note is shown as text; the box to write it only opens with "Add / Edit relation note"."""
    open_key, widget_key = f"editnote_{slot}_{pid}", f"rel_note_{slot}_{pid}"
    note = ann.get("relation_note", "")
    editing = ss.get(open_key, False)
    with st.container(key=f"relnote_{slot}"):
        c1, c2 = st.columns([1.2, 1], vertical_alignment="center")
        c1.caption("**Relation note**")
        c2.button("Cancel" if editing else ("Edit relation note" if note else "Add relation note"),
                  key=f"notebtn_{slot}_{pid}", type="tertiary", width="stretch",
                  icon=":material/close:" if editing else (":material/edit:" if note else ":material/add:"),
                  on_click=_toggle, args=(open_key,))
        if editing:
            st.text_input("Relation note", value=note, key=widget_key, label_visibility="collapsed",
                          placeholder="How this paper relates to the linked papers (Enter to save)",
                          on_change=_save_and_close, args=(pid, "relation_note", widget_key, open_key))
        else:
            st.caption(note or "No note yet.")


def render_links(slot: int, pid: str, ann: dict):
    """The related papers (always listed). "Add related paper" shows a database and a paper picker;
    the chosen paper is linked (both ways) right away and appears in the list."""
    other = 2 if slot == 1 else 1
    adding_key = f"adding_{slot}_{pid}"
    with st.container(key=f"links_{slot}"):  # title and file name side by side
        c1, c2 = st.columns([1.2, 1], vertical_alignment="center")
        c1.caption(f"**Related papers ({len(ann['linked_papers'])})**")
        adding = ss.get(adding_key, False)
        c2.button("Cancel" if adding else "Add related paper", key=f"add_{slot}_{pid}", type="tertiary",
                  icon=":material/close:" if adding else ":material/add:", width="stretch",
                  on_click=_toggle, args=(adding_key,))
        if adding:
            db_key, paper_key = f"reldb_{slot}_{pid}", f"relpaper_{slot}_{pid}"
            databases = sorted({database_of(p.paper_id) for p in papers}, key=str.lower)
            c1, c2 = st.columns([1, 2])
            db = c1.selectbox("Database", databases, index=None, key=db_key, placeholder="Database",
                              format_func=lambda d: d or "(top level)")
            candidates = [p.paper_id for p in papers if db is not None and database_of(p.paper_id) == db
                          and p.paper_id != pid and p.paper_id not in ann["linked_papers"]]
            c2.selectbox("Paper", candidates, index=None, key=paper_key, format_func=lambda x: by_id[x].title,
                         placeholder="Choose a database first" if db is None else "Choose a paper (type to search)",
                         disabled=db is None, on_change=_link_selected, args=(pid, paper_key, adding_key))
        if not ann["linked_papers"]:
            st.caption("None yet.")
        saved_names = dict(zip(ann["linked_papers"], ann.get("linked_files", [])))
        for lid in ann["linked_papers"]:
            c = st.columns([6.5, 3.5, 0.7, 0.7], vertical_alignment="center")
            if lid not in by_id:
                c[0].caption("(PDF no longer in the folder)")
                c[1].caption(saved_names.get(lid, lid))
            else:
                c[0].markdown(f"🔗 {by_id[lid].title}")
                c[1].caption(store.file_stem(by_id[lid].filename))
                c[2].button("", icon=":material/open_in_new:", key=f"openl_{slot}_{pid}_{lid}", on_click=_open,
                            args=(other, lid), help="Open side by side", type="tertiary")
            c[3].button("", icon=":material/link_off:", key=f"unlink_{slot}_{pid}_{lid}", on_click=_unlink,
                        args=(pid, lid), help="Unlink", type="tertiary")


def render_extraction(slot: int, pid: str, ann: dict):
    if not review.fields:
        st.info("No fields defined. Add them under `fields:` in `config/review.yaml`.")
        return
    paper = by_id[pid]
    # Fill with AI results stay in memory (draft = field values, pending = values + quotes + pages)
    # until Save stores them with the fields; Discard drops them.
    draft_key, ver_key, pending_key = f"draft_{pid}", f"ver_{pid}", f"ai_pending_{pid}"

    draft = ss.get(draft_key, {})
    c1, c2 = st.columns([1, 2], vertical_alignment="center")
    if draft:
        c2.caption("AI values are filled in below. Check them, then click **Save**.")
    if c1.button("✨ Fill with AI", key=f"ai_{slot}_{pid}", type="primary", width="stretch",
                 help="Read the paper and pre-fill every field. Nothing is saved until you click Save."):
        with st.spinner("Reading the paper and extracting fields…"):
            try:
                result = extraction.suggest(paper, settings, review, model=ss.chat_model_name)
            except Exception as e:
                st.error(f"AI extraction failed: {e}")
                result = None
        if result is not None:
            ss[pending_key] = {"suggestions": result, "model": ss.chat_model_name}
            draft = {name: r["value"] for name, r in result.items() if r and r.get("value") not in (None, [])}
            draft |= {f"{name}_specify": r["specify"] for name, r in result.items() if r and r.get("specify")}
            ss[draft_key] = draft
            ss[ver_key] = ss.get(ver_key, 0) + 1  # new widget keys, so widgets pick up the AI values
            st.rerun()

    pending = ss.get(pending_key)
    suggestions = pending["suggestions"] if pending else ann.get("ai_suggestions") or {}

    ver = ss.get(ver_key, 0)
    def current(name: str):
        return draft.get(name, ann["fields"].get(name))

    sections: dict[str, list[FieldDef]] = {}
    for f in review.fields:
        sections.setdefault(f.section, []).append(f)

    with st.form(f"form_{slot}_{pid}_{ver}", border=False):
        values = {}
        for title, fields in sections.items():
            filled = sum(current(f.name) not in (None, "", []) for f in fields)
            box = st.expander(f"**{title}** · {filled}/{len(fields)}", expanded=False) if title else st.container()
            with box:
                row = []  # short fields go two per row; long text, multi-select and "specify" get a full row
                for f in fields:
                    key = f"fld_{slot}_{pid}_{ver}_{f.name}"
                    if f.specify:
                        c_main, c_spec = st.columns([3, 2], vertical_alignment="bottom")
                        with c_main:
                            values[f.name] = field_widget(f, current(f.name), key=key)
                        values[f.specify_name] = c_spec.text_input(
                            f.specify, value=current(f.specify_name) or "", key=f"{key}_specify").strip() or None
                        continue
                    if f.type in ("longtext", "multiselect"):
                        box = st.container()
                    else:
                        row = row or list(st.columns(2))
                        box = row.pop(0)
                    with box:
                        values[f.name] = field_widget(f, current(f.name), key=key)
        c1, c2 = st.columns(2)
        save_clicked = c1.form_submit_button("💾 Save", type="primary", width="stretch")
        discard_clicked = c2.form_submit_button("Discard unsaved", width="stretch")

    if save_clicked:
        a = ann_of(pid)
        a["fields"].update(values)
        if pending:
            a["ai_suggestions"], a["ai_model"] = pending["suggestions"], pending["model"]
        store.save(a, settings)
        ss.pop(draft_key, None)
        ss.pop(pending_key, None)
        st.toast("Saved ✓")
        st.rerun()
    if discard_clicked:
        ss.pop(draft_key, None)
        ss.pop(pending_key, None)
        ss[ver_key] = ver + 1
        st.rerun()

    evidence = [(f, suggestions[f.name]) for f in review.fields
                if (suggestions.get(f.name) or {}).get("page")]
    if evidence:
        with st.expander("🔍 Show evidence in the PDF"):
            for f, s in evidence:
                st.button(f"p. {s['page']} · {f.label}", key=f"goto_{slot}_{pid}_{f.name}",
                          on_click=_goto, args=(pid, s["page"], s.get("evidence")),
                          help=f"“{s.get('evidence') or ''}” (click to jump there and highlight it)")


def render_chat(slot: int, pid: str, ann: dict, height: int):
    """The conversation lives in this browser session only (not saved; lost on restart)."""
    chat_key = f"chat_{pid}"
    history = ss.get(chat_key, [])
    box = st.container(height=height)
    with box:
        if not history:
            st.caption("Ask anything about this paper, e.g. *What is the main limitation?* "
                       "or *Which papers in my collection are related to this one?* "
                       "The chat is not saved: it is gone after a restart.")
        for m in history:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

    prompt = st.chat_input("Ask about this paper…", key=f"chatin_{slot}_{pid}")
    if prompt:
        history = [*history, {"role": "user", "content": prompt}]
        with box:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                try:
                    answer = st.write_stream(chat.stream_answer(by_id[pid], history, settings, ss.chat_model_name))
                except Exception as e:
                    st.error(f"Chat failed: {e}")
                    return
        ss[chat_key] = [*history, {"role": "assistant", "content": answer}]
        st.rerun()

    if history and st.button("🗑️ Clear chat", key=f"clear_{slot}_{pid}"):
        ss.pop(chat_key, None)
        st.rerun()


def render_results(slot: int, pid: str, ann: dict, results: list[dict], kind: str):
    """One row per paper: title and file name side by side; hover the title for the matching text."""
    other = 2 if slot == 1 else 1
    results = [r for r in results if r["paper_id"] in by_id]
    if not results:
        return
    widths = [0.5, 6, 3.5, 1.2, 0.7, 0.7]
    with st.container(key=f"results_{kind}_{slot}"):
        for col, text in zip(st.columns(widths), ["#", "Title", "File", "Similarity"]):
            col.caption(f"**{text}**")
        for rank, r in enumerate(results, 1):
            rid = r["paper_id"]
            where = f"p. {r['page']}" if r.get("page") else "abstract"
            c = st.columns(widths, vertical_alignment="center")
            c[0].caption(f"{rank}.")
            c[1].markdown(by_id[rid].title, help=f"{where}: {r['snippet'][:400]}…")
            c[2].caption(store.file_stem(by_id[rid].filename))
            c[3].caption(f"{r['score']:.2f}")
            if rid in ann["linked_papers"]:
                c[4].caption("🔗", help="Linked")
            else:
                c[4].button("", icon=":material/add_link:", key=f"link_{kind}_{slot}_{pid}_{rid}", type="tertiary",
                            on_click=_link, args=(pid, rid), help="Link to this paper")
            c[5].button("", icon=":material/open_in_new:", key=f"open_{kind}_{slot}_{pid}_{rid}", type="tertiary",
                        on_click=_open, args=(other, rid), help="Open side by side")


def render_related(slot: int, pid: str, ann: dict):
    """Two tasks: papers similar to this one, and papers matching a topic. Results stay until the next search."""
    similar_key, topic_key = f"similar_{pid}", f"topic_{pid}"

    st.markdown("**1. Similar to this paper**")
    if st.button("Find similar papers", key=f"sim_{slot}_{pid}", icon=":material/compare_arrows:",
                 help=f"Top {settings.similar_top_k} papers by similarity of title + abstract (all databases)"):
        try:
            ss[similar_key] = index.similar_to_paper(by_id[pid], settings)
        except Exception as e:
            st.error(str(e))
    render_results(slot, pid, ann, ss.get(similar_key, []), "sim")

    st.markdown("**2. Search papers by topic**")
    c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
    q = c1.text_input("Topic", key=f"q_{slot}_{pid}", placeholder="e.g. resource calendars in simulation",
                      label_visibility="collapsed")
    if c2.button("Search", key=f"search_{slot}_{pid}", disabled=not q, width="stretch",
                 help="Papers with a passage matching the topic (all databases)"):
        try:
            ss[topic_key] = index.search(q, settings, exclude=pid)
        except Exception as e:
            st.error(str(e))
    render_results(slot, pid, ann, ss.get(topic_key, []), "topic")


def render_review(slot: int, pid: str, chat_height: int):
    ann = ann_of(pid)
    with st.container(key=f"reviewpanel_{slot}"):
        render_decision(slot, pid, ann)
        t1, t2, t3 = st.tabs(["📝 Extraction", "💬 Chat", "🔗 Related papers"])
        with t1:
            render_extraction(slot, pid, ann)
        with t2:
            render_chat(slot, pid, ann, chat_height)
        with t3:
            render_related(slot, pid, ann)


def current_paper(slot: int) -> str:
    key = f"slot{slot}"
    if ss.get(key) not in by_id:
        pool = ordered or [p.paper_id for p in papers]
        ss[key] = pool[min(slot - 1, len(pool) - 1)]
    return ss[key]


# ---------- main ----------

height = ss.get("pdf_height", 900)
if ss.compare:
    for slot, col in enumerate(st.columns(2), 1):
        with col:
            pid = current_paper(slot)
            render_header(slot, pid)
            render_pdf(slot, pid, height)
            render_review(slot, pid, chat_height=420)
else:
    pid = current_paper(1)
    render_header(1, pid)
    left, right = st.columns([6, 5])
    with left:
        render_pdf(1, pid, height)
    with right:
        render_review(1, pid, chat_height=height - 330)
