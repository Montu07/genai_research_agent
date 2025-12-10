import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from transformers import pipeline

def load_pdf(path: str):
    loader = PyPDFLoader(path)
    docs = loader.load()
    return docs

def split_docs(docs, chunk_size=800, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    return chunks

def build_faiss_index(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectordb = FAISS.from_documents(chunks, embeddings)
    return vectordb

def build_retriever(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    return retriever

def build_local_llm():
    """
    Use a small local model for text-to-text generation.
    This does NOT need any API key.
    """
    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",   # smaller than -large
        max_length=256
    )
    return generator

def answer_question(question: str, retriever, llm_pipe):
    # 1. Retrieve relevant chunks
    docs = retriever.invoke(question)

    # join text, but avoid sending a HUGE context to the model
    context = "\n\n".join(d.page_content for d in docs)
    # limit to first ~3000 characters so it fits in model input
    context = context[:3000]

    prompt = (
        "You are a helpful assistant answering questions about a cricket training PDF.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Give a concise and clear answer based only on the context."
    )

    result = llm_pipe(prompt)[0]["generated_text"]
    return result, docs

def main():
    pdf_path = os.path.join("data", "sample.pdf")
    print(f"Loading PDF from: {pdf_path}")

    if not os.path.exists(pdf_path):
        print("❌ PDF not found! Make sure data/sample.pdf exists.")
        return

    docs = load_pdf(pdf_path)
    print(f"✅ Loaded {len(docs)} pages from PDF")

    chunks = split_docs(docs)
    print(f"✅ Split into {len(chunks)} chunks")

    vectordb = build_faiss_index(chunks)
    print("✅ FAISS index built")

    retriever = build_retriever(vectordb)
    llm_pipe = build_local_llm()
    print("✅ Local LLM pipeline loaded (google/flan-t5-base)\n")

    while True:
        question = input("Ask a question about the PDF (or 'q' to quit): ")
        if question.lower() in ("q", "quit", "exit"):
            break

        answer, docs = answer_question(question, retriever, llm_pipe)

        print("\n=== ANSWER ===")
        print(answer)
        print("\n=== TOP CONTEXT CHUNKS USED ===")
        for i, d in enumerate(docs, start=1):
            print(f"\n--- Chunk {i} ---")
            print(d.page_content[:300], "...")
        print("============================\n")

if __name__ == "__main__":
    main()
