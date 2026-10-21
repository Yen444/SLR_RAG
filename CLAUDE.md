# CLAUDE.md

Local Streamlit tool for reviewing a paper collection: PDF viewer, config-driven AI-assisted extraction, per-paper chat, similar-paper search. See README.md for usage.

## Commands
- `uv sync`: install
- `uv run pytest`: tests (no OpenAI calls; LLM/embeddings are faked)
- `uv run streamlit run app.py`: UI
- `uv run python scripts/ingest.py`: index new papers

## Architecture
- `slr_rag/config.py`: pydantic models for `config/settings.yaml` + `config/review.yaml`. `SLR_SETTINGS` / `SLR_REVIEW` env vars override paths (used by tests).
- `catalog.py`: `Paper` objects; `paper_id` = sha1 of file content (12 chars).
- `pdf_text.py`: PyMuPDF page text, cached; `full_text()` adds `[Page N]` markers.
- `screening.py`: optional screening CSVs inside `papers_dir` (one sub-folder per database with its CSV + PDFs, included items only). Column names differ per database: `load_records` copies them into internal `_title`, `_item_id`, … columns via `settings.csv_columns` (not exported). Links rows to PDFs in the CSV's folder or below (`_pdf_filename`, else `_item_id`), so `app.py` shows only linked papers with CSV title/abstract/metadata, plus `export` (original columns + review/extraction columns). No CSVs = all PDFs.
- `store.py`: per-paper JSON in `data/annotations/`, atomic writes, bidirectional links.
- `llm.py`: the only place models are created. Named `chat_models` / `embedding_models` in settings.yaml → LangChain `init_chat_model` / `init_embeddings` (any provider). Never import a provider class (ChatOpenAI etc.) elsewhere.
- `index.py`: Chroma (cosine), two collections per embedding-model key: `paperlevel-*` (title + abstract, one vector per paper, id = paper_id) for "similar to this paper", and `papers-*` (chunks) for topic search. `index_papers` fills whichever is missing. **RAG is only for cross-paper similarity**, not for chatting about one paper.
- `viewer.py`: PDF.js (pinned version) downloaded to `static/pdfjs`, PDFs copied to `static/pdf/<id>.pdf`, shown via `st.iframe("/app/static/pdfjs/web/viewer.html?file=…#page=N&search=…")`. Needs `enableStaticServing` in `.streamlit/config.toml` (read only at startup).
- `chat.py`: full paper text in the system prompt + tools `search_collection` / `similar_papers`; manual streaming tool loop.
- `extraction.py`: dynamic pydantic model from config, `{value, evidence, page}` per field, via `with_structured_output` (fields required-but-nullable so OpenAI strict mode accepts it; Gemini uses `response_json_schema`). The schema has no descriptions or enums; field descriptions and options go into the prompt (`field_guide`), and `_normalize` maps answers back to the exact option names. `review.yaml` `sections:` are flattened into `fields` (each `FieldDef.section` is set); `specify` adds a `<name>_specify` value.
- `app.py`: UI. Annotations are always re-read from disk (`ann_of`) and saved immediately in widget callbacks. Panels are `st.container(key=...)` styled via `.st-key-*` CSS at the top of the file; theme in `.streamlit/config.toml`. Tests stub `viewer.ensure_pdfjs` and `viewer.PDF_DIR`.

## Gotchas
- `Chroma.similarity_search_by_vector_with_relevance_scores` returns cosine **distances**, not relevance; `_group_by_paper` converts with `1 - distance`.
- ChatOpenAI silently drops `temperature` for gpt-5 (reasoning) models.
- Gemini rejects big structured-output schemas with `400 ... too many states for serving` (long enums or descriptions). Don't put options back into the schema as `Literal`s.
- Streamlit: use `width="stretch"`, not the deprecated `use_container_width`.
- Widget keys include slot + paper_id (+ a version counter for the extraction form so AI values can replace widget state).

## Running the app (dev workflow)
- Keep **one** Streamlit server. A second `streamlit run` silently takes the next port (8502, …), and the user may end up on a stale one. Check with `netstat -ano | grep -E ":850[0-9].*LISTEN"` before starting another.
- Streamlit reruns `app.py` on change but keeps already-imported `slr_rag/*` modules in memory. After editing anything in `slr_rag/` (or `.streamlit/config.toml`), **restart the server**; a browser refresh is enough only for `app.py`-only changes. Tell the user which one is needed. A stale server shows errors like `AttributeError: 'Paper' object has no attribute 'folder'`.
- The user often runs the server in their own PowerShell terminal (venv activated, `streamlit run app.py`). Don't stop a process you didn't start; ask the user to Ctrl+C it.
- Commands the user types with `!` in Claude Code run in Git Bash: use forward slashes (`uv run scripts/ingest.py`); `scripts\ingest.py` becomes `scriptsingest.py`.
- Windows console is cp1252: set `PYTHONIOENCODING=utf-8` when printing chat output (emoji) from ad-hoc scripts.
- Real API keys live in `.env` (Gemini: `GOOGLE_API_KEY`). Never print them. Tests must not read it (`test_app` stubs `slr_rag.config.load_dotenv`).
