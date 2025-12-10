def print_sources(docs, max_docs: int = 3, snippet_len: int = 200):
    print("\n--- SOURCES (Top Chunks) ---")
    for i, d in enumerate(docs[:max_docs]):
        page = d.metadata.get("page", "N/A") if hasattr(d, "metadata") else "N/A"
        snippet = d.page_content.replace("\n", " ")[:snippet_len]
        print(f"[{i+1}] page={page}: {snippet}...")
    print("----------------------------\n")
