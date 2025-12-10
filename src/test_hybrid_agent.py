from .controller_agent import HybridControllerAgent
from .q_learning import TRAIN_QUESTIONS


def main():
    agent = HybridControllerAgent()

    print("\n=== Hybrid RL Agent Demo ===")
    for idx, q in enumerate(TRAIN_QUESTIONS):
        print(f"\n[Question {idx}] {q}")
        result = agent.answer_with_rl(idx, q)

        print("\n=== PIPELINE RESULT ===")
        print(f"Question: {result['question']}")
        print(f"Selected Action: {result['selected_action']}")
        print(f"Context Used: {result['context_used']}")
        print(f"Reward: {result['reward']:.3f}")
        print("\nAnswer:\n", result["answer"])
        print("="*40)


if __name__ == "__main__":
    main()
