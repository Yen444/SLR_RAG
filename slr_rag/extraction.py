"""AI-assisted extraction: fill the configured fields from the paper, with a supporting quote per field."""

from typing import Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, create_model

from . import llm
from .catalog import Paper
from .config import FieldDef, ReviewConfig, Settings
from .pdf_text import full_text, get_pages

SYSTEM_PROMPT = """You extract structured information from a scientific paper for a systematic literature review.
For every field: give the value (null if the paper does not report it), a short verbatim quote
from the paper supporting it, and the page number from the [Page N] markers.
Do not guess: prefer null over an unsupported value."""


def _value_type(f: FieldDef):
    return {
        "text": str,
        "longtext": str,
        "number": float,
        "bool": bool,
        "select": Literal[tuple(f.options)] if f.options else str,
        "multiselect": list[Literal[tuple(f.options)]] if f.options else list[str],
    }[f.type]


def build_model(review: ReviewConfig) -> type[BaseModel]:
    """Pydantic model with one {value, evidence, page} entry per configured field.

    All keys are required-but-nullable, which OpenAI strict structured output needs."""
    defs = {}
    for f in review.fields:
        answer = create_model(
            f"Answer_{f.name}",
            value=(Optional[_value_type(f)], Field(description=f.description or f.label)),
            evidence=(Optional[str], Field(description="Short verbatim quote from the paper supporting the value")),
            page=(Optional[int], Field(description="Page number of the quote")),
        )
        defs[f.name] = (answer, Field(description=f"{f.label}: {f.description}".strip(": ")))
    return create_model("PaperExtraction", **defs)


def suggest(paper: Paper, settings: Settings, review: ReviewConfig, model: str | None = None) -> dict[str, dict]:
    """Returns {field_name: {"value", "evidence", "page"}}. `model` is a key of settings.chat_models."""
    structured = llm.chat_model(settings, model).with_structured_output(build_model(review))
    text = full_text(get_pages(paper, settings), settings.max_context_chars)
    result = structured.invoke([SystemMessage(SYSTEM_PROMPT), HumanMessage(f"Title: {paper.title}\n\n{text}")])
    return result.model_dump()
