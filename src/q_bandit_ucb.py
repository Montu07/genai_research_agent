import math
import random
import numpy as np

from .rl_env import QAContextSizeEnv
from .q_learning import TRAIN_QUESTIONS, NUM_ACTIONS


def ucb_select_action(avg_rewards, counts, total_count, c: float = 2.0) -> int:
    """
    avg_rewards: [num_actions]
    counts: [num_actions] (# times each action taken)
    total_count: total number of action selections so far
    c: exploration strength (bigger -> more exploration)
    """
    ucb_values = []

    for a in range(NUM_ACTIONS):
        if counts[a] == 0:
            # force try each action at least once
            return a

        bonus = c * math.sqrt(math.log(total_count) / counts[a])
        ucb = avg_rewards[a] + bonus
        ucb_values.append(ucb)

    return int(np.argmax(ucb_values))


def train_ucb_bandit(num_episodes: int = 60):
    """
    UCB-based bandit training:
    - State = question index
    - Action = context size (0,1,2)
    - Reward = compute_reward from environment
    We maintain separate bandit stats per question.
    """

    env = QAContextSizeEnv()

    # For each question: track counts & average rewards per action
    action_counts = np.zeros((len(TRAIN_QUESTIONS), NUM_ACTIONS), dtype=int)
    action_avg_rewards = np.zeros((len(TRAIN_QUESTIONS), NUM_ACTIONS), dtype=float)
    total_action_counts = np.zeros(len(TRAIN_QUESTIONS), dtype=int)

    for episode in range(num_episodes):
        # randomly pick which question to train on this episode
        q_idx = random.randint(0, len(TRAIN_QUESTIONS) - 1)
        question = TRAIN_QUESTIONS[q_idx]

        env.reset(question)

        total_action_counts[q_idx] += 1
        total_for_q = total_action_counts[q_idx]

        # choose action with UCB
        avg_rewards_for_q = action_avg_rewards[q_idx]
        counts_for_q = action_counts[q_idx]

        action = ucb_select_action(
            avg_rewards=avg_rewards_for_q,
            counts=counts_for_q,
            total_count=total_for_q,
            c=2.0,  # exploration strength
        )

        next_state, reward, done, info = env.step(action)

        # update bandit stats for this question & action
        action_counts[q_idx, action] += 1
        n = action_counts[q_idx, action]

        # incremental average update
        old_avg = action_avg_rewards[q_idx, action]
        new_avg = old_avg + (reward - old_avg) / n
        action_avg_rewards[q_idx, action] = new_avg

        print(
            f"Episode {episode+1}/{num_episodes} | "
            f"q_idx={q_idx} | action={action} | reward={reward:.3f} | "
            f"count={n} | avg_reward={new_avg:.3f}"
        )

    return action_counts, action_avg_rewards


def summarize_policy(action_counts, action_avg_rewards):
    print("\n=== UCB Bandit Policy Summary ===")
    for i, q in enumerate(TRAIN_QUESTIONS):
        best_action = int(np.argmax(action_avg_rewards[i]))
        if best_action == 0:
            ctx = "short (1000 chars)"
        elif best_action == 1:
            ctx = "medium (3000 chars)"
        else:
            ctx = "long (5000 chars)"

        print(f"\nQuestion {i}: '{q}'")
        print(f"  Action counts: {action_counts[i]}")
        print(f"  Avg rewards : {np.round(action_avg_rewards[i], 3)}")
        print(f"  => Best action by UCB bandit: {best_action} -> {ctx}")


def main():
    action_counts, action_avg_rewards = train_ucb_bandit(num_episodes=60)
    summarize_policy(action_counts, action_avg_rewards)


if __name__ == "__main__":
    main()
