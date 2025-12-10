from typing import Optional

from langchain_core.tools import Tool

from .rag_pipeline import rag_answer


def _rag_tool_fn(question: str) -> str:
    """
    Simple wrapper so the Tool returns just a string answer.
    """
    result = rag_answer(question)
    return result["answer"]


# LangChain Tool object we can plug into agents later
rag_qa_tool: Tool = Tool(
    name="cricket_rag_qa",
    description=(
        "Answers questions about the cricket training PDF using "
        "retrieval-augmented generation (FAISS + local LLM). "
        "Use this whenever the user asks about cricket coaching, skills, "
        "batting, bowling, or anything from the PDF."
    ),
    func=_rag_tool_fn,
)
