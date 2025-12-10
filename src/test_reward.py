from .rag_pipeline import rag_answer
from .eval_answer import compute_reward


def main():
    while True:
        q = input("Ask a question about the PDF (or 'q' to quit): ")
        if q.lower() in ("q", "quit", "exit"):
            break

        result = rag_answer(q)
        answer = result["answer"]
        docs = result["context_docs"]

        reward = compute_reward(q, answer, docs)

        print("\n=== ANSWER ===")
        print(answer)
        print("\nReward score (0–1):", round(reward, 3))
        print("============================\n")

if __name__ == "__main__":
    main()
