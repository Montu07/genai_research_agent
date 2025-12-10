import random
import numpy as np

from .rl_env import QAContextSizeEnv

# Fixed set of training questions (states)
TRAIN_QUESTIONS = [
    "What are prompt patterns and why are they useful?",
    "What is chain-of-thought prompting?",
    "How can prompting help reduce hallucinations in LLMs?",
    "What is the difference between zero-shot and few-shot prompting?",
    "What is the role of role-based or persona prompts?",
    "What are common causes of prompt failures?",
]

NUM_STATES = len(TRAIN_QUESTIONS)
NUM_ACTIONS = 3  # 0=short, 1=medium, 2=long


def train_q_learning(
    num_episodes: int = 100,
    alpha: float = 0.3,    # learning rate
    gamma: float = 0.9,    # discount factor
    epsilon_start: float = 0.9,
    epsilon_end: float = 0.1,
):
    env = QAContextSizeEnv()

    # Q-table: [state_index, action]
    Q = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=float)

    # linearly decay epsilon
    def epsilon_for_episode(ep):
        if num_episodes <= 1:
            return epsilon_end
        frac = ep / (num_episodes - 1)
        return epsilon_start + frac * (epsilon_end - epsilon_start)

    for episode in range(num_episodes):
        # choose which question (state) to use this episode
        state_idx = random.randint(0, NUM_STATES - 1)
        question = TRAIN_QUESTIONS[state_idx]

        epsilon = epsilon_for_episode(episode)

        # reset env with this question
        state = env.reset(question)

        # epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, NUM_ACTIONS - 1)
        else:
            action = int(np.argmax(Q[state_idx]))

        next_state, reward, done, info = env.step(action)

        # Since the episode ends after one step, the Q-update is simple:
        best_next = np.max(Q[state_idx])  # same state_idx used for value bootstrap
        old_value = Q[state_idx, action]
        Q[state_idx, action] = old_value + alpha * (reward + gamma * best_next - old_value)

        print(
            f"Episode {episode+1}/{num_episodes} | "
            f"state={state_idx} | action={action} | reward={reward:.3f} | eps={epsilon:.3f}"
        )

    return Q


def evaluate_policy(Q):
    """Print the best action per question according to the learned Q-table."""
    print("\n=== Learned Policy (best action per question) ===")
    for i, q in enumerate(TRAIN_QUESTIONS):
        best_action = int(np.argmax(Q[i]))
        if best_action == 0:
            ctx = "short (1000 chars)"
        elif best_action == 1:
            ctx = "medium (3000 chars)"
        else:
            ctx = "long (5000 chars)"

        print(f"Question {i}: '{q}'")
        print(f"  -> Best action: {best_action}  => context = {ctx}")
        print(f"  -> Q-values: {Q[i]}")
        print()


def main():
    Q = train_q_learning(num_episodes=60)
    # save Q-table for later inference
    import numpy as np
    np.save("q_table.npy", Q)
    print("\nSaved Q-table to q_table.npy")
    evaluate_policy(Q)


if __name__ == "__main__":
    main()
