"""Chat tool loop with a fake model (no OpenAI calls)."""

from langchain_core.messages import AIMessageChunk, ToolMessage

from slr_rag import chat
from slr_rag.catalog import scan_papers

from .conftest import make_pdf


class FakeChat:
    """First turn: calls search_collection. Second turn: answers using the tool result."""

    seen: list = []

    def __init__(self, **_):
        pass

    def bind_tools(self, tools):
        return self

    def stream(self, messages):
        FakeChat.seen.append(messages)
        if not any(isinstance(m, ToolMessage) for m in messages):
            yield AIMessageChunk(content="", tool_call_chunks=[
                {"name": "search_collection", "args": '{"query": "pets"}', "id": "call_1", "index": 0}
            ])
        else:
            yield AIMessageChunk(content="Related: ")
            yield AIMessageChunk(content="paper B.")


def test_stream_answer_runs_tools(settings, monkeypatch):
    make_pdf(settings.papers_dir / "a.pdf", ["about cats"])
    (paper,) = scan_papers(settings)
    picked = []
    monkeypatch.setattr(chat.llm, "chat_model", lambda s, name=None: picked.append(name) or FakeChat())
    monkeypatch.setattr(chat.index, "search", lambda q, s, exclude=None: [
        {"paper_id": "b1", "title": "Paper B", "score": 0.9, "page": 2, "snippet": "dogs"}
    ])

    out = "".join(chat.stream_answer(paper, [{"role": "user", "content": "related papers?"}], settings, "gpt"))

    assert picked == ["gpt"]
    assert "Searching collection: pets" in out and out.endswith("Related: paper B.")
    system = FakeChat.seen[0][0].content
    assert "[Page 1]" in system and "about cats" in system
    tool_msg = next(m for m in FakeChat.seen[1] if isinstance(m, ToolMessage))
    assert "paper_id=b1" in tool_msg.content
