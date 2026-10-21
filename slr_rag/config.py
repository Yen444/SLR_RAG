"""Load and validate config/settings.yaml and config/review.yaml."""

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"

FieldType = Literal["text", "longtext", "number", "bool", "select", "multiselect"]


class FieldDef(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    label: str = ""
    type: FieldType = "text"
    options: list[str] = []
    description: str = ""
    placeholder: str = ""  # example text shown in an empty text box
    height: int | None = None  # longtext only: box height in px
    specify: str = ""  # label of an extra free-text box, saved as <name>_specify (e.g. "Other (specify)")
    section: str = ""  # set from the `sections:` list

    @model_validator(mode="after")
    def _check(self):
        if self.type in ("select", "multiselect") and not self.options:
            raise ValueError(f"field '{self.name}' is {self.type} but has no options")
        if not self.label:
            self.label = self.name.replace("_", " ").capitalize()
        return self

    @property
    def specify_name(self) -> str:
        return f"{self.name}_specify"


class Section(BaseModel):
    title: str
    fields: list[FieldDef] = []


class ReviewConfig(BaseModel):
    statuses: list[str] = Field(min_length=1)
    done_status: str = ""
    decisions: list[str] = Field(min_length=1)
    exclude_decision: str = "exclude"
    exclusion_reasons: list[str] = []
    ai_instructions: str = ""  # context for "Fill with AI", e.g. what the review is about
    sections: list[Section] = []  # fields grouped under a heading; added to `fields` in order
    fields: list[FieldDef] = []

    @model_validator(mode="after")
    def _check(self):
        self.fields = [*self.fields, *(f.model_copy(update={"section": s.title}) for s in self.sections for f in s.fields)]
        self.sections = []
        if not self.done_status:
            self.done_status = self.statuses[-1]
        if self.done_status not in self.statuses:
            raise ValueError(f"done_status '{self.done_status}' is not in statuses")
        names = [f.name for f in self.fields] + [f.specify_name for f in self.fields if f.specify]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"duplicate field names: {sorted(duplicates)}")
        return self


class ModelSpec(BaseModel):
    """One entry of chat_models / embedding_models: a LangChain provider + model name + extra kwargs."""

    provider: str  # e.g. google_genai, openai, anthropic, ollama, mistralai (see LangChain init_chat_model)
    model: str
    params: dict = {}  # passed to the model class, e.g. temperature, base_url


# Screening-CSV columns: for each piece of information, the column names that may hold it.
# The first name that exists in a CSV is used (ignoring case, spaces and underscores).
DEFAULT_CSV_COLUMNS = {
    "item_id": ["item_id"],
    "pdf_filename": ["pdf_filename"],
    "title": ["title", "Document Title", "Item Title"],
    "abstract": ["abstract"],
    "authors": ["author", "authors"],
    "year": ["year", "Publication Year"],
    "venue": ["journal", "booktitle", "Publication Title", "Source"],
    "doi": ["doi", "Item DOI"],
    "note": ["note"],
    "final_decision": ["final_decision"],
}


class Settings(BaseModel):
    papers_dir: Path = Path("papers")
    data_dir: Path = Path("data")
    chat_model: str  # key in chat_models (default, can be switched in the UI)
    embedding_model: str  # key in embedding_models
    chat_models: dict[str, ModelSpec] = Field(min_length=1)
    embedding_models: dict[str, ModelSpec] = Field(min_length=1)
    chunk_size: int = 1500
    chunk_overlap: int = 200
    similar_top_k: int = 10
    paper_level_words: int = 300  # "similar to this paper" compares the title + this many words after "Abstract"
    max_context_chars: int = 400_000
    csv_columns: dict[str, list[str]] = {}  # merged over DEFAULT_CSV_COLUMNS

    @model_validator(mode="after")
    def _check(self):
        if self.chat_model not in self.chat_models:
            raise ValueError(f"chat_model '{self.chat_model}' is not in chat_models: {sorted(self.chat_models)}")
        if self.embedding_model not in self.embedding_models:
            raise ValueError(
                f"embedding_model '{self.embedding_model}' is not in embedding_models: {sorted(self.embedding_models)}"
            )
        self.csv_columns = DEFAULT_CSV_COLUMNS | self.csv_columns
        if not self.papers_dir.is_absolute():
            self.papers_dir = (ROOT / self.papers_dir).resolve()
        if not self.data_dir.is_absolute():
            self.data_dir = (ROOT / self.data_dir).resolve()
        return self

    @property
    def text_dir(self) -> Path:
        return self.data_dir / "text"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def annotations_dir(self) -> Path:
        return self.data_dir / "annotations"


def _read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_settings(path: Path | None = None) -> Settings:
    """Load settings; SLR_SETTINGS env var can point to another file (used by tests)."""
    load_dotenv(ROOT / ".env")
    path = path or Path(os.environ.get("SLR_SETTINGS", CONFIG_DIR / "settings.yaml"))
    return Settings(**_read_yaml(path))


def load_review(path: Path | None = None) -> ReviewConfig:
    path = path or Path(os.environ.get("SLR_REVIEW", CONFIG_DIR / "review.yaml"))
    return ReviewConfig(**_read_yaml(path))
