import numpy as np
from sentence_transformers import SentenceTransformer
from .q_learning import TRAIN_QUESTIONS

class SemanticQuestionRouter:
    """
    Maps ANY user question to the closest training question
    using sentence embeddings (cosine similarity).
    """

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.train_questions = TRAIN_QUESTIONS
        self.train_embeddings = self.model.encode(self.train_questions, convert_to_numpy=True)

    def map_to_state(self, user_question: str):
        """
        Returns (state_idx, similarity score)
        """
        user_emb = self.model.encode([user_question], convert_to_numpy=True)[0]

        # Compute cosine similarity with all 6 training questions
        sims = np.dot(self.train_embeddings, user_emb) / (
            np.linalg.norm(self.train_embeddings, axis=1) * np.linalg.norm(user_emb)
        )

        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        return best_idx, best_score
