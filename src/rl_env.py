from typing import Dict, Any

from .rag_pipeline import rag_answer
from .eval_answer import compute_reward


class QAContextSizeEnv:
    """
    Simple RL environment:
    - Episode = one question about the PDF.
    - Action = choose context size (short / medium / long).
    - Reward = quality of answer (overlap + length) computed by compute_reward.
    """

    def __init__(self):
        # Define discrete actions
        # 0 -> short, 1 -> medium, 2 -> long
        self.action_space_n = 3

        # Map actions to max_context_chars
        self.context_size_map = {
            0: 1000,   # short
            1: 3000,   # medium
            2: 5000,   # long
        }

        self.current_question: str | None = None
        self.last_answer: str | None = None
        self.last_reward: float = 0.0
        self.done: bool = False

    def reset(self, question: str) -> Dict[str, float]:
        """
        Start a new episode with a given question.
        Returns the initial state.
        """
        self.current_question = question
        self.last_answer = None
        self.last_reward = 0.0
        self.done = False

        # Initial state can be very simple: just question length and last reward = 0
        state = {
            "question_len": len(question),
            "last_reward": 0.0,
        }
        return state

    def step(self, action: int) -> tuple[Dict[str, float], float, bool, Dict[str, Any]]:
        """
        Take one action:
        - Choose a context size based on action.
        - Run RAG answer.
        - Compute reward.
        - Mark episode as done (we keep it one-step for now).
        Returns: (next_state, reward, done, info)
        """
        if self.current_question is None:
            raise ValueError("Call reset(question) before step().")

        if action not in self.context_size_map:
            raise ValueError(f"Invalid action {action}. Must be 0, 1, or 2.")

        max_ctx = self.context_size_map[action]

        # 1. Call RAG with chosen context size
        result = rag_answer(self.current_question, max_context_chars=max_ctx)
        answer = result["answer"]
        docs = result["context_docs"]

        # 2. Compute reward
        reward = compute_reward(self.current_question, answer, docs)
        self.last_answer = answer
        self.last_reward = reward

        # 3. In this simple version, episode ends after one step
        self.done = True

        # 4. Build next state (could include more features later)
        next_state = {
            "question_len": len(self.current_question),
            "last_reward": reward,
        }

        info = {
            "answer": answer,
            "max_context_chars": max_ctx,
        }

        return next_state, reward, self.done, info
