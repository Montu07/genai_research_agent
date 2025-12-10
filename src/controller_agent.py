import numpy as np

from .semantic_router import SemanticQuestionRouter
from .rag_pipeline import rag_answer
from .q_learning import TRAIN_QUESTIONS, NUM_ACTIONS
from .q_bandit_ucb import ucb_select_action
from .eval_answer import compute_reward

THRESHOLD = 0.10   # If Q-values are close → use UCB exploration


class HybridControllerAgent:
    """
    Hybrid controller that combines:
    - Q-learning policy (exploitation)
    - UCB bandit (exploration)
    and uses a semantic router to map ANY user question
    to the closest training question.
    """

    def __init__(self, q_table_path: str = "q_table.npy"):
        # Load Q-learning table
        self.Q = np.load(q_table_path)

        # Semantic router to generalize beyond the 6 training questions
        self.router = SemanticQuestionRouter()

        # Initialize bandit stats (counts & avg rewards)
        self.action_counts = np.zeros((len(TRAIN_QUESTIONS), NUM_ACTIONS), dtype=int)
        self.action_avg_rewards = np.zeros((len(TRAIN_QUESTIONS), NUM_ACTIONS), dtype=float)
        self.total_counts = np.zeros(len(TRAIN_QUESTIONS), dtype=int)

    def choose_action(self, question_idx: int) -> int:
        """
        Hybrid decision rule:
        1. Check Q-values for this training question:
            - If best & second-best differ < THRESHOLD → exploration via UCB
            - Else → greedy Q-learning (best action)
        """
        q_values = self.Q[question_idx]

        # Sort actions by Q-value (descending)
        sorted_actions = q_values.argsort()[::-1]
        best_action = sorted_actions[0]
        second_best = sorted_actions[1]

        # Check if Q-values are too close → uncertainty
        if abs(q_values[best_action] - q_values[second_best]) < THRESHOLD:
            # Use UCB exploration
            a_counts = self.action_counts[question_idx]
            a_avg = self.action_avg_rewards[question_idx]
            total = self.total_counts[question_idx] + 1

            print("[HYBRID] Q-values too close → Using UCB exploration!")
            action = ucb_select_action(
                avg_rewards=a_avg,
                counts=a_counts,
                total_count=total,
                c=2.0,
            )
            return action

        # Otherwise → greedy Q-learning
        print("[HYBRID] Q-values clear → Using Q-learning best action")
        return best_action

    def update_bandit_stats(self, question_idx: int, action: int, reward: float) -> None:
        """
        Update UCB bandit stats for learning.
        """
        self.total_counts[question_idx] += 1
        self.action_counts[question_idx][action] += 1

        n = self.action_counts[question_idx][action]
        old_avg = self.action_avg_rewards[question_idx][action]
        new_avg = old_avg + (reward - old_avg) / n
        self.action_avg_rewards[question_idx][action] = new_avg

    def answer_with_rl(self, question: str) -> dict:
        """
        Full agent pipeline for ANY user question:
        - route question to closest training question (semantic router)
        - choose action with hybrid RL policy (Q-learning + UCB)
        - run RAG with context size determined by the action
        - compute reward using eval_answer.compute_reward
        - update bandit stats
        """
        # 1) Semantic routing: map user question → training question index
        question_idx, sim = self.router.map_to_state(question)
        print(f"[ROUTER] Mapped to training question {question_idx} (similarity={sim:.3f})")

        # 2) Choose action (0=short, 1=medium, 2=long)
        action = self.choose_action(question_idx)

        # 3) Map action → context size
        ctx_sizes = {
            0: 1000,
            1: 3000,
            2: 5000,
        }
        max_ctx = ctx_sizes[action]

        print(f"[CONTROLLER] Selected action {action} -> context {max_ctx}")

        # 4) Run RAG
        result = rag_answer(question, max_context_chars=max_ctx)
        answer_text = result["answer"]
        context_docs = result["context_docs"]

        # 5) Compute reward (semantic overlap with gold answers if available)
        reward = compute_reward(question, answer_text, context_docs)
        print(f"[CONTROLLER] Reward = {reward:.3f}")

        # 6) Update bandit stats
        self.update_bandit_stats(question_idx, action, reward)

        # 7) Return full trace for analysis / logging
        return {
            "question_idx": question_idx,
            "router_similarity": sim,
            "question": question,
            "selected_action": action,
            "context_used": max_ctx,
            "answer": answer_text,
            "reward": reward,
            "context_docs": context_docs,  # for sources & evaluation
        }
