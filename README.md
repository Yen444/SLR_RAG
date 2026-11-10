# SLR_RAG

A local tool for working through a collection of research papers (e.g. for a systematic literature review).

- **Browse** papers in the sidebar. Each database (top-level folder of `papers/`) is listed with its number of papers; click one to show its papers, click again to close it. Two rows of buttons filter the lists and combine: **Status** (All / Todo / In progress / Done / Pending) and **Decision** (Include / Exclude / Maybe; click the selected one again to show all decisions). Counts follow the other row. Decisions never hide papers from a status; only Todo leaves out pending papers. The app always starts on Todo, also after a browser refresh. The search narrows the lists too; a search with no database open searches all of them. Click a row to open a paper, or use ◀ ▶.
- **View** one paper, or two side by side (compare mode), in the PDF.js viewer: scroll, page box, zoom, thumbnails and Ctrl+F search. It's downloaded once into `static/` on first start.
- **Extract** information into fields you define in `config/review.yaml`. **Fill with AI** pre-fills every field with a supporting quote and page number; you review, edit and save.
- **Chat** about the open paper. The full paper text is sent to the model, so answers cite pages. You can also ask *"which papers in my collection are related?"* and the chat searches your collection.
- **Track** status (todo / in progress / done, set by hand), the decision with exclusion reason (dropdown) and an exclusion note, related papers (pick a database, then a paper; links work in both directions and across databases) with a relation note. Click 🕓 in the paper header to mark a paper as **Pending** (to deal with later, whatever its status or decision) and optionally write a pending note; the **Pending** button lists these papers (🕓 and their status before the title, the note as tooltip). Pending papers are left out of Todo / In progress / Done and their counts until you unmark them; **All** still shows them. These are saved as soon as you change them (a "Saved ✓" message appears); the extraction form is saved with its **Save** button.
- **Find related papers** in your collection (RAG: embeddings + Chroma):
  - *Similar to this paper* compares one vector per paper (the title + the 300 words after "Abstract" in the PDF, `paper_level_words`) and returns the top-k by similarity score.
  - *Search by topic* finds papers with a matching passage (chunk-level vectors).
- **Find papers by author** in the 👤 Authors tab, across all databases (from the CSV author columns; no indexing or API key needed). It finds the exact text you type in the author column, ignoring case. Each database writes names differently (*WMP van der Aalst*, *Wil M. P. van der Aalst*), so a surname finds the most. Google Scholar cuts long author lists short (…), so authors beyond the cut can't be found.
- **Any model provider:** Gemini, OpenAI, Anthropic, local Ollama, … Pick the chat model in the sidebar.
- Everything is saved as JSON, one file per paper in `data/annotations/`, and can be exported to CSV.

Stack: Python, LangChain, Chroma, Streamlit. Default models: Gemini.

## Setup

```bash
uv sync
cp .env.example .env          # then put your GOOGLE_API_KEY (Gemini) and/or OPENAI_API_KEY in .env
```

Put your PDFs in `papers/` (subfolders are fine).

## Run

```bash
uv run python scripts/ingest.py     # index papers for similar-paper search (only new papers are embedded)
uv run streamlit run app.py         # open the UI
```

You can also index from the UI (sidebar → **Build / update index**). After adding PDFs, click **Rescan papers folder**.

## Configuration

| File | What it controls |
| --- | --- |
| `config/review.yaml` | Extraction fields, statuses, decisions, exclusion reasons |
| `config/settings.yaml` | Folders, the list of models, chunking, number of similar papers |

### Models

`config/settings.yaml` has a named list of chat models and embedding models:

```yaml
chat_model: gemini-flash          # default (switchable in the sidebar)
embedding_model: gemini
chat_models:
  gemini-flash: {provider: google_genai, model: gemini-2.5-flash}
  gpt-5-mini:   {provider: openai, model: gpt-5-mini}
  my-local:     {provider: ollama, model: llama3.1, params: {temperature: 0}}
```

`provider` is any [LangChain provider](https://docs.langchain.com/oss/python/langchain/models) name. Gemini and OpenAI are installed. For others, install the integration first (e.g. `uv add langchain-anthropic` or `uv add langchain-ollama`) and set its API key in `.env`. `params` is passed to the model as-is (e.g. `base_url` for an OpenAI-compatible server).

Each embedding model has its own index, so after changing `embedding_model`, run the indexer again.

Field types: `text`, `longtext`, `number`, `bool`, `select`, `multiselect` (the last two need `options`). The `description` of a field is shown as help text and is also the instruction the AI uses when filling it.

Fields can be grouped under `sections:` (each with a `title` and `fields`). Each section is shown as a collapsible part of the form with a "filled / total" count. Optional field settings:
- `placeholder`: example text in an empty box.
- `height`: box height in pixels, for `longtext`.
- `specify`: label of an extra free-text box next to the field, saved as `<name>_specify` and exported as its own column. Use it for "Other (specify)" or "domain, if domain-specific".

`ai_instructions` (top level) adds review-specific rules for **Fill with AI**. Config changes apply on the next interaction in the app, with no restart needed. Removing a field from the config hides it but keeps its saved values in the JSON.

## Screening CSVs (work on included papers only)

If you screened your papers in an earlier step, give each database its own folder in `papers/`. Put that database's screening CSV (**only the included items**) and its PDFs in the folder:

```
papers/
  science_direct/
    science_direct_results.csv
    science_direct_4_included.pdf
    ...
  google_scholar/
    google_scholar_results.csv
    google_scholar_1_included.pdf
    ...
```

Then:

- Only PDFs that belong to a CSV row are shown, grouped by database folder. A PDF is linked to a row of the CSV in its own folder (or a folder above it) by the `pdf_filename` column, or else by `item_id` (`<item_id>.pdf`, `<item_id>_included.pdf`). Other PDFs are hidden.
- Linking related papers and *Similar to this paper* / *Search* work across all databases. Linked papers and search results show their database folder.
- Titles come from the CSV (the CSV abstract is not used; *Similar to this paper* reads the abstract from the PDF). Each database names these columns differently (`title`, `Title`, `Document Title`, `Item Title`, …); `csv_columns` in `config/settings.yaml` lists the names that are recognised and can be extended.
- CSV rows without a PDF are listed in the sidebar (⚠️), so nothing is forgotten.
- Every paper starts as **include**. Excluding one in the app moves it under the **Excluded** button, where you can undo it.
- **Updated copies of your CSVs** are kept current automatically in `data/reviewed/<database>/<name>_reviewed.csv`, rewritten after every change. Each has the CSV's own columns, with `final_decision` set to the decision after this review (e.g. `excluded`) and the screening decision kept in `screening_final_decision`, plus the review and extraction columns. If one is open in Excel it can't be written; the app warns and retries after your next change.
- **Export CSV** writes one new file for all databases: all your original rows and columns, plus `source_csv` / `source_folder`, `final_decision_updated`, the review columns (`review_status`, `review_pending`, `review_pending_note`, `review_decision`, `review_decision_reasons` (in your order, `;`-separated), `review_decision_note`, `linked_papers` (file names without .pdf, e.g. `google_scholar_62_included`), `review_relation_note`, …) and one column per extraction field. The original CSVs and the PDFs are never changed.

Without CSV files the app shows every PDF in `papers/`.

## Tags and approaches (open coding)

Use these when the categories aren't fixed in advance. They grow while you read.

- **On each paper** (decision box): **Tags** and **Approaches**. Pick existing ones or type a new name and press Enter. A paper can have any number of both. A name that differs from an existing one only by case or spaces is saved as the existing one, so `petri net` becomes `Petri net`.
- **Approaches are numbered** (`3. Petri-net-based simulation`) and can have a short description. Numbers never change and are never reused, even after a merge or delete.
- **Overview** (sidebar → **🏷️ Tags & approaches**):
  - counts, plus how many papers have no tag or approach yet;
  - the papers with a tag or in an approach (click a title to open the paper);
  - approach names and descriptions, edited right in the table;
  - rename or merge tags, merge or delete approaches;
  - which tags occur together, and which tags each approach's papers have. Use these to find approaches in your tags.
- **Export:** the Export CSV and `data/reviewed/` copies get `review_tags` and `review_approaches` columns. **Export approaches** downloads the codebook (number, name, description, papers).

## Data

```
data/
  annotations/<paper_id>.json   # your work: status, decision, fields, AI suggestions, links, notes, tags, approaches
  approaches.json               # numbered approaches with descriptions
  text/<paper_id>.json          # cached page text
  chroma/                       # vector index
  catalog.json                  # file → paper_id cache
```

`paper_id` is a hash of the PDF content, so renaming or moving a PDF keeps its annotations. `papers/` and `data/` are git-ignored. **Back up `data/annotations/` and `data/approaches.json`**, since that's your review work.

## Notes

- Scanned PDFs without a text layer can't be read (the indexer warns about them). Run OCR on them first.
- Costs: indexing embeds each paper once. Chat and **Fill with AI** send the full paper text on each call. Gemini and OpenAI both cache repeated prompt prefixes automatically, which makes follow-up chat questions cheaper.

## Tests

```bash
uv run pytest
```
