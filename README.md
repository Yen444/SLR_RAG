# SLR_RAG

A local tool for working through a collection of research papers (e.g. for a systematic literature review).

- **Browse** all papers in the sidebar, grouped by sub-folder of `papers/` and marked with their status. Click one to open it, or use ◀ ▶.
- **View** one paper, or two side by side (compare mode), in the PDF.js viewer: scroll, page box, zoom, thumbnails and Ctrl+F search. It's downloaded once into `static/` on first start.
- **Extract** information into fields you define in `config/review.yaml`. **Fill with AI** pre-fills every field with a supporting quote and page number; you review, edit and save.
- **Chat** about the open paper. The full paper text is sent to the model, so answers cite pages. You can also ask *"which papers in my collection are related?"* and the chat searches your collection.
- **Track** status (todo / in progress / done), the decision with exclusion reason (dropdown), your own reason and a reason note, plus notes and links between papers.
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

Field types: `text`, `longtext`, `number`, `bool`, `select`, `multiselect` (the last two need `options`). The `description` of a field is shown as help text and is also the instruction the AI uses when filling it. Config changes apply on the next interaction in the app, with no restart needed. Removing a field from the config hides it but keeps its saved values in the JSON.

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
