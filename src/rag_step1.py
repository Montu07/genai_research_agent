import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

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

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    while True:
        question = input("\nAsk something about the PDF (or 'q' to quit): ")
        if question.lower() in ("q", "quit", "exit"):
            break

        results = retriever.invoke(question)
        print("\n=== TOP MATCHING CHUNKS ===")
        for i, doc in enumerate(results, start=1):
            print(f"\n--- Chunk {i} ---")
            print(doc.page_content[:300], "...")
        print("============================")

if __name__ == "__main__":
    main()
