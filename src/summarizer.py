import os

from .rag_pipeline import load_pdf, build_local_llm


def summarize_pdf(max_chars: int = 3000, summary_words: int = 200) -> str:
    """
    Load the cricket PDF and generate a concise summary using the local LLM.
    This does NOT use RAG; it just summarizes the raw document content.
    """
    pdf_path = os.path.join("data", "sample.pdf")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError("data/sample.pdf not found. Put your PDF there.")

    # Load all pages
    docs = load_pdf(pdf_path)
    full_text = "\n\n".join(d.page_content for d in docs)
    full_text = full_text[:max_chars]  # keep it small enough for the model

    generator = build_local_llm()

    prompt = (
        f"You are a helpful assistant. Summarize the following cricket training document "
        f"in about {summary_words} words. Focus on the main skills and structure of the content.\n\n"
        f"Document:\n{full_text}\n\n"
        "Summary:"
    )

    result = generator(prompt)[0]["generated_text"]
    return result
