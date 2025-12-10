import re
from typing import List
from langchain_core.documents import Document

# Map from question text (lowercased) to an index
QUESTION_TO_INDEX = {
    "what are prompt patterns and why are they useful?": 0,
    "what is chain-of-thought prompting?": 1,
    "how can prompting help reduce hallucinations in llms?": 2,
    "what is the difference between zero-shot and few-shot prompting?": 3,
    "what is the role of role-based or persona prompts?": 4,
    "what are common causes of prompt failures?": 5,
}

# Short gold answers (you can tweak / refine these)
GOLD_ANSWERS = {
    0: "Prompt patterns are reusable templates that capture successful ways of interacting with LLMs. They help structure prompts and reliably achieve specific behaviors.",
    1: "Chain-of-thought prompting asks the model to write out intermediate reasoning steps before giving the final answer, which often improves reasoning quality.",
    2: "Prompting can reduce hallucinations by grounding the model in provided context, asking it to say I don't know, adding verification steps and constraining it to cite sources.",
    3: "Zero-shot prompting gives only the task description, while few-shot prompting also provides labeled examples in the prompt to steer the model.",
    4: "Role-based or persona prompts put the model in a specific role, such as teacher or security analyst, to influence tone, style and what knowledge it uses.",
    5: "Prompt failures come from vague instructions, missing context, overloaded prompts, conflicting constraints and asking for information beyond the model's training data.",
}


def _normalize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t]
    return tokens


def compute_reward(question: str, answer: str, context_docs: List[Document]) -> float:
    """
    Reward in [0,1] based on word overlap between generated answer
    and a short 'gold' reference answer for that question.
    Fallback: length-based reward if question not in map.
    """
    q_key = question.strip().lower()
    idx = QUESTION_TO_INDEX.get(q_key)

    # Fallback if this question isn't one of the 6 training questions
    if idx is None or idx not in GOLD_ANSWERS:
        return min(len(answer) / 300.0, 1.0)

    gold = GOLD_ANSWERS[idx]

    gold_tokens = set(_normalize(gold))
    ans_tokens = set(_normalize(answer))

    if not gold_tokens:
        return min(len(answer) / 300.0, 1.0)

    overlap = gold_tokens.intersection(ans_tokens)
    score = len(overlap) / len(gold_tokens)

    # Clip to [0,1]
    return max(0.0, min(score, 1.0))
