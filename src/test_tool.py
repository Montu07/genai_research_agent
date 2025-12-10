from .tools import rag_qa_tool

def main():
    while True:
        q = input("Ask a question about the PDF (or 'q' to quit): ")
        if q.lower() in ("q", "quit", "exit"):
            break

        answer = rag_qa_tool.run(q)
        print("\n=== ANSWER FROM TOOL ===")
        print(answer)
        print("========================\n")

if __name__ == "__main__":
    main()
