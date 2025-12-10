import random

from .rl_env import QAContextSizeEnv


def main():
    env = QAContextSizeEnv()

    while True:
        q = input("Enter a question about the PDF for a new episode (or 'q' to quit): ")
        if q.lower() in ("q", "quit", "exit"):
            break

        state = env.reset(q)
        print(f"\n[ENV] New episode started. State: {state}")

        # Random agent: picks an action randomly from {0,1,2}
        action = random.randint(0, env.action_space_n - 1)
        print(f"[AGENT] Chose action: {action} (0=short, 1=medium, 2=long)")

        next_state, reward, done, info = env.step(action)

        print("\n=== ANSWER ===")
        print(info["answer"])
        print("========================")

        print(f"[ENV] max_context_chars used: {info['max_context_chars']}")
        print(f"[ENV] Reward: {reward:.3f}")
        print(f"[ENV] Next state: {next_state}")
        print(f"[ENV] Done: {done}")
        print("\n---------------------------------\n")


if __name__ == "__main__":
    main()
