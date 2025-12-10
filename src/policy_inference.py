import numpy as np

from .q_learning import TRAIN_QUESTIONS, NUM_ACTIONS
from .rl_env import QAContextSizeEnv
from .rag_pipeline import rag_answer


def load_q_table(path: str = "q_table.npy") -> np.ndarray:
    Q = np.load(path)
    if Q.shape != (len(TRAIN_QUESTIONS), NUM_ACTIONS):
        raise ValueError(f"Unexpected Q shape: {Q.shape}")
    return Q


def main():
    # Load learned Q-table
    Q = load_q_table()
    env = QAContextSizeEnv()

    print("=== RL Policy Inference ===")
    print("Using learned Q-table from q_table.npy\n")

    while True:
        print("Select a question (or 'q' to quit):\n")
        for i, q in enumerate(TRAIN_QUESTIONS):
            print(f"{i}: {q}")
        choice = input("\nEnter question index (0-3) or 'q': ")

        if choice.lower() in ("q", "quit", "exit"):
            break

        try:
            idx = int(choice)
        except ValueError:
            print("Please enter a valid number.\n")
            continue

        if idx < 0 or idx >= len(TRAIN_QUESTIONS):
            print("Index out of range.\n")
            continue

        question = TRAIN_QUESTIONS[idx]

        # Get best action from Q-table
        best_action = int(np.argmax(Q[idx]))
        ctx_map = env.context_size_map
        max_ctx = ctx_map[best_action]

        print(f"\n[POLICY] For this question, best action = {best_action} "
              f"-> max_context_chars = {max_ctx}")

        # Use the chosen context size to answer
        result = rag_answer(question, max_context_chars=max_ctx)
        answer = result["answer"]

        print("\n=== QUESTION ===")
        print(question)
        print("\n=== ANSWER (policy-optimized) ===")
        print(answer)
        print("============================\n")


if __name__ == "__main__":
    main()
