import os
from functools import lru_cache

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from transformers import pipeline
from langchain_core.documents import Document



def load_all_documents(folder="data"):
    """Load ALL PDFs & text files in the data folder."""
    docs = []
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)

        if fname.endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())

        elif fname.endswith(".txt") or fname.endswith(".md"):
            text = open(path, "r", encoding="utf-8").read()
            docs.append(Document(page_content=text))

    return docs


def split_docs(docs, chunk_size=600, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


def build_faiss_index(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectordb = FAISS.from_documents(chunks, embeddings)
    return vectordb


def build_retriever(vectordb):
    return vectordb.as_retriever(search_kwargs={"k": 5})


def build_local_llm():
    """
    Use Mistral-7B-Instruct for high-quality RAG answers.
    """
    return pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.2",
        device_map="auto",
        torch_dtype="auto",
        max_new_tokens=300,
        temperature=0.2,
        do_sample=False
    )


@lru_cache(maxsize=1)
def init_rag():
    """
    Load PDFs, build FAISS index & LLM only once.
    """
    print("Loading all documents from /data ...")
    docs = load_all_documents("data")
    print(f"Loaded {len(docs)} documents.")

    chunks = split_docs(docs)
    print(f"Split into {len(chunks)} chunks.")

    vectordb = build_faiss_index(chunks)
    retriever = build_retriever(vectordb)
    llm_pipe = build_local_llm()

    return retriever, llm_pipe


def rag_answer(question: str, max_context_chars: int = 5000) -> dict:
    retriever, llm = init_rag()

    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    context = context[:max_context_chars]

    prompt = f"""
You are an expert assistant for PROMPT ENGINEERING.

Use ONLY the information in the context below to answer the question.
If the context strongly does NOT contain enough information, then reply:
"Not found in the documents."

Otherwise, based on the context, answer as completely and clearly as you can.

Context:
{context}

Question: {question}

Answer clearly and concisely using ONLY the context. If possible, list key points as bullet points.
"""

    response = llm(prompt)[0]["generated_text"]

    return {
        "answer": response,
        "context_docs": docs,
    }
