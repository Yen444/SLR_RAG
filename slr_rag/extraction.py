"""AI-assisted extraction: fill the configured fields from the paper, with a supporting quote per field."""

from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, create_model

from . import llm
from .catalog import Paper
from .config import FieldDef, ReviewConfig, Settings
from .pdf_text import full_text, get_pages

SYSTEM_PROMPT = """You extract structured information from a scientific paper for a systematic literature review.
For every field: give the value (null if the paper does not report it), a short verbatim quote
from the paper supporting it, and the page number from the [Page N] markers.
Do not guess: prefer null over an unsupported value.
Some fields have a `specify` text: fill it only when the field asks for it (e.g. an "Other" option was chosen).
The fields to fill are described below."""


def _value_type(f: FieldDef):
    # Options are plain strings, not enums: long option lists make Gemini reject the schema
    # ("too many states"). The prompt lists the options and _normalize maps answers back to them.
    return {"text": str, "longtext": str, "number": float, "bool": bool, "select": str, "multiselect": list[str]}[f.type]


def _normalize(f: FieldDef, value):
    """Map answers to the exact option spelling (case/space-insensitive); unknown answers are kept."""
    if f.type not in ("select", "multiselect") or value is None:
        return value
    canon = {" ".join(o.split()).casefold(): o for o in f.options}

    def one(v: str) -> str:
        return canon.get(" ".join(str(v).split()).casefold(), v)

    if f.type == "select":
        return one(value)
    return list(dict.fromkeys(one(v) for v in value))  # dedupe, keep order


def build_model(review: ReviewConfig) -> type[BaseModel]:
    """Pydantic model with one {value, evidence, page} entry per configured field.

    All keys are required-but-nullable, which OpenAI strict structured output needs. The schema has no
    descriptions (they are in the prompt, see field_guide): Gemini rejects schemas with a lot of text."""
    defs = {}
    for f in review.fields:
        extra = {"specify": (Optional[str], ...)} if f.specify else {}
        answer = create_model(f"Answer_{f.name}", value=(Optional[_value_type(f)], ...), **extra,
                              evidence=(Optional[str], ...), page=(Optional[int], ...))
        defs[f.name] = (answer, ...)
    return create_model("PaperExtraction", **defs)


def field_guide(review: ReviewConfig) -> str:
    """What each field means, for the prompt."""
    lines = []
    for f in review.fields:
        line = f"- {f.name} ({f.label}, {f.type})" + (f": {f.description}" if f.description else "")
        if f.options:
            line += f" Options ({'one' if f.type == 'select' else 'any number'}): " + " | ".join(f.options) + "."
        if f.specify:
            line += f" `specify`: {f.specify}."
        lines.append(line)
    return "Fields:\n" + "\n".join(lines)


def suggest(paper: Paper, settings: Settings, review: ReviewConfig, model: str | None = None) -> dict[str, dict]:
    """Returns {field_name: {"value", "evidence", "page"[, "specify"]}}. `model` is a key of settings.chat_models."""
    structured = llm.chat_model(settings, model).with_structured_output(build_model(review))
    text = full_text(get_pages(paper, settings), settings.max_context_chars)
    system = "\n\n".join(x for x in (SYSTEM_PROMPT, review.ai_instructions.strip(), field_guide(review)) if x)
    result = structured.invoke([SystemMessage(system), HumanMessage(f"Title: {paper.title}\n\n{text}")])
    answers = result.model_dump()
    for f in review.fields:
        if answers.get(f.name):
            answers[f.name]["value"] = _normalize(f, answers[f.name]["value"])
    return answers
