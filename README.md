# SLR_RAG

A local tool for working through a collection of research papers (e.g. for a systematic literature review).

- **Browse** papers in the sidebar. The buttons show the counts (All / In progress / Done / Excluded). Click one, or pick a decision or database folder, or type a search, to list the matching papers (click the button again to hide the list). Click a row to open a paper, or use ◀ ▶.
- **View** one paper, or two side by side (compare mode), in the PDF.js viewer: scroll, page box, zoom, thumbnails and Ctrl+F search. It's downloaded once into `static/` on first start.
- **Extract** information into fields you define in `config/review.yaml`. **Fill with AI** pre-fills every field with a supporting quote and page number; you review, edit and save.
- **Chat** about the open paper. The full paper text is sent to the model, so answers cite pages. You can also ask *"which papers in my collection are related?"* and the chat searches your collection.
- **Track** status (todo / in progress / done, set by hand), the decision with exclusion reason (dropdown), your own reason and a reason note, plus notes and links between papers.
- **Find related papers** in your collection (RAG: embeddings + Chroma):
  - *Similar to this paper* compares one vector per paper (title + abstract) and returns the top-k by similarity score.
  - *Search by topic* finds papers with a matching passage (chunk-level vectors).
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
- Titles, abstracts (for *Similar to this paper*), authors, year, journal and DOI come from the CSV. Each database names these columns differently (`title`, `Title`, `Document Title`, `Item Title`, …); `csv_columns` in `config/settings.yaml` lists the names that are recognised and can be extended. Your screening `note` is shown under the title.
- CSV rows without a PDF are listed in the sidebar (⚠️), so nothing is forgotten.
- Every paper starts as **include**. Excluding one in the app moves it under the **Excluded** button, where you can undo it.
- **Export CSV** writes one new file for all databases: all your original rows and columns, plus `source_csv` / `source_folder`, `final_decision_updated`, the review columns (`review_status`, `review_decision`, `review_exclusion_reason`, …) and one column per extraction field. The original CSVs and the PDFs are never changed.

Without CSV files the app shows every PDF in `papers/`.

## Data

```
data/
  annotations/<paper_id>.json   # your work: status, decision, fields, AI suggestions, links, notes, chat
  text/<paper_id>.json          # cached page text
  chroma/                       # vector index
  catalog.json                  # file → paper_id cache
```

`paper_id` is a hash of the PDF content, so renaming or moving a PDF keeps its annotations. `papers/` and `data/` are git-ignored. **Back up `data/annotations/`**, since that's your review work.

## Notes

- Scanned PDFs without a text layer can't be read (the indexer warns about them). Run OCR on them first.
- Costs: indexing embeds each paper once. Chat and **Fill with AI** send the full paper text on each call. Gemini and OpenAI both cache repeated prompt prefixes automatically, which makes follow-up chat questions cheaper.

## Tests

```bash
uv run pytest
```
