from .rag_pipeline import build_local_llm, rag_answer
from .controller_agent import HybridControllerAgent
from .q_learning import TRAIN_QUESTIONS
from .eval_answer import compute_reward
from .utils_print import print_sources


def answer_naive(llm_pipe, question: str) -> str:
    """
    Baseline: call the LLM directly with no document context.
    """
    prompt = (
        "You are a general assistant. Answer this question based only on your "
        "own knowledge, without access to any specific document.\n\n"
        f"Question: {question}\n\n"
        "Answer clearly:\n"
    )
    return llm_pipe(prompt)[0]["generated_text"]


def main():
    print("=== Naive vs RAG vs RL-RAG Comparison ===\n")
    print("Select a question index:\n")

    for i, q in enumerate(TRAIN_QUESTIONS):
        print(f"{i}: {q}")

    max_idx = len(TRAIN_QUESTIONS) - 1
    choice = input(f"\nEnter index (0-{max_idx}): ")

    try:
        q_idx = int(choice)
        if q_idx < 0 or q_idx > max_idx:
            raise ValueError()
        question = TRAIN_QUESTIONS[q_idx]
    except Exception:
        print("Invalid index.")
        return

    print(f"\n[QUESTION] {question}\n")

    # --------------------------------------------------
    # 1) NAIVE LLM (no PDF / no RAG)
    # --------------------------------------------------
    print(">>> 1) Naive LLM (no RAG)")
    llm_pipe = build_local_llm()
    naive_answer = answer_naive(llm_pipe, question)

    # Use RAG docs as a common reference context for scoring
    rag_fixed = rag_answer(question, max_context_chars=3000)
    ref_docs = rag_fixed["context_docs"]

    naive_reward = compute_reward(question, naive_answer, ref_docs)

    print("\n[NAIVE ANSWER]")
    print(naive_answer)
    print(f"\n[NAIVE SCORE] reward = {naive_reward:.3f}")
    print_sources(ref_docs)

    # --------------------------------------------------
    # 2) PLAIN RAG (fixed context size)
    # --------------------------------------------------
    print(">>> 2) Plain RAG (fixed context = 3000 chars)")
    rag_answer_fixed = rag_fixed["answer"]
    rag_reward = compute_reward(question, rag_answer_fixed, ref_docs)

    print("\n[RAG ANSWER]")
    print(rag_answer_fixed)
    print(f"\n[RAG SCORE] reward = {rag_reward:.3f}")
    print_sources(ref_docs)

    # --------------------------------------------------
    # 3) RL-RAG (Hybrid Controller: Q-learning + UCB)
    # --------------------------------------------------
    print(">>> 3) RL-RAG (Hybrid controller)")
    agent = HybridControllerAgent()
    rl_result = agent.answer_with_rl(question)

    rl_answer = rl_result["answer"]
    rl_docs = rl_result["context_docs"]
    rl_reward = compute_reward(question, rl_answer, rl_docs)

    print("\n[RL-RAG ANSWER]")
    print(rl_answer)
    print(f"\n[RL-RAG SCORE] reward = {rl_reward:.3f}")
    print_sources(rl_docs)

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------
    print("=== COMPARISON SUMMARY ===")
    print(f"Naive LLM reward   : {naive_reward:.3f}")
    print(f"Plain RAG reward   : {rag_reward:.3f}")
    print(f"RL-RAG reward      : {rl_reward:.3f}")
    print("===========================")


if __name__ == "__main__":
    main()
