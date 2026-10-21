"""Chat about one paper. The full text goes in the prompt; tools search the rest of the collection."""

from collections.abc import Iterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from . import index, llm
from .catalog import Paper
from .config import Settings
from .pdf_text import full_text, get_pages

SYSTEM_PROMPT = """You are a research assistant helping with a systematic literature review.
You are discussing ONE paper. Its full text is below, with [Page N] markers.

- Answer from the paper and cite pages like (p. 4). If the paper does not say, say so.
- To find OTHER papers in the user's collection (related work, same topic, similar methods),
  call the tools. Only mention collection papers the tools returned, with their paper_id.

Title: {title}
<paper>
{text}
</paper>"""

MAX_TOOL_ROUNDS = 4


def _format_results(results: list[dict]) -> str:
    if not results:
        return "No matching papers found in the collection."
    lines = []
    for r in results:
        where = f"(p. {r['page']})" if r["page"] else "Abstract:"
        lines.append(f"- paper_id={r['paper_id']} | score={r['score']} | {r['title']}\n  {where} {r['snippet']}")
    return "\n\n".join(lines)


def make_tools(paper: Paper, settings: Settings):
    @tool
    def search_collection(query: str) -> str:
        """Search the user's paper collection for papers matching a topic, method or question."""
        return _format_results(index.search(query, settings, exclude=paper.paper_id))

    @tool
    def similar_papers() -> str:
        """Find the papers in the user's collection most similar to the current paper overall."""
        try:
            return _format_results(index.similar_to_paper(paper, settings))
        except LookupError as e:
            return str(e)

    return [search_collection, similar_papers]


def stream_answer(paper: Paper, history: list[dict], settings: Settings, model: str | None = None) -> Iterator[str]:
    """Yield the answer text as it streams. `history` ends with the new user message.

    `model` is a key of settings.chat_models (default: settings.chat_model)."""
    tools = make_tools(paper, settings)
    by_name = {t.name: t for t in tools}
    model_with_tools = llm.chat_model(settings, model).bind_tools(tools)

    text = full_text(get_pages(paper, settings), settings.max_context_chars)
    messages = [SystemMessage(SYSTEM_PROMPT.format(title=paper.title, text=text))]
    for m in history:
        messages.append(HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]))

    for _ in range(MAX_TOOL_ROUNDS + 1):
        response = None
        for chunk in model_with_tools.stream(messages):
            if chunk.text:
                yield chunk.text
            response = chunk if response is None else response + chunk
        if response is None or not response.tool_calls:
            return
        messages.append(response)
        for call in response.tool_calls:
            query = call["args"].get("query")
            yield f"\n\n*🔎 Searching collection{f': {query}' if query else ' for similar papers'}…*\n\n"
            result = by_name[call["name"]].invoke(call["args"])
            messages.append(ToolMessage(result, tool_call_id=call["id"]))
