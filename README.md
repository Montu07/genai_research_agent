# Reinforcement Learning–Enhanced Agentic AI System for Retrieval-Augmented Generation

This project integrates **Reinforcement Learning (RL)** with **Retrieval-Augmented Generation (RAG)** to dynamically optimize context selection for an LLM.  
The system enhances answer quality by selecting the best context window (1000 / 3000 / 5000 chars) using **Q-learning**.

The project includes:
- A full RAG pipeline
- A Q-learning training environment
- A hybrid controller with semantic routing + RL policy
- Evaluation pipeline comparing Naive LLM vs RAG vs RL-RAG
- A real-user CLI assistant demonstrating improved performance

---

## 🚀 Project Features

### ✅ Reinforcement Learning (Q-Learning)
- Learns best context size for each question
- State space = question categories
- Actions = {short, medium, long context}
- Reward = similarity score measuring answer quality
- Stores Q-table for inference

### ✅ Retrieval-Augmented Generation (RAG)
Implemented using `langchain` + `HuggingFace`.

Pipeline:
- Loads PDFs from `/data/`
- Splits documents into text chunks
- Embeds chunks via HuggingFace embeddings
- Builds FAISS / Chroma vector index
- Retrieves top-k similar chunks for a user question
- Truncates context to RL-selected length
- Builds structured prompt:
  
  *“You are an expert assistant for PROMPT ENGINEERING. Use ONLY the context below…”*

- Sends final prompt to LLM and returns the answer

---

## 🧠 System Architecture

### Components:
1. **Semantic Router**
   - Maps user question → nearest trained question category

2. **RL Controller**
   - Loads Q-table  
   - Chooses best context size (`1000/3000/5000`)

3. **RAG Pipeline**
   - PDF loading  
   - Chunking  
   - Embedding  
   - Vector search  
   - Prompt construction  

4. **LLM Inference**
   - Uses HuggingFace model for grounded answers  

5. **Evaluation Pipeline**
   - Runs:
     - Naive (no RAG)
     - Plain RAG
     - RL-RAG
   - Compares rewards and outputs

---

## 📊 RL Training Results

Q-learning outputs:

Saved Q-table to q_table.npy
Saved training rewards to training_rewards.npy

=== Learned Policy (best action per question) ===
Question 0 → Best action: long (5000 chars)
Question 1 → Best action: long (5000 chars)
Question 2 → Best action: long (5000 chars)
Question 3 → Best action: short (1000 chars)
Question 4 → Best action: short (1000 chars)
Question 5 → Best action: medium (3000 chars)


Interpretation:
- Concept-heavy questions need **more context**
- Definition-style questions work better with **short context**
- Mixed-depth questions prefer **medium**

---

## 📈 Learning Curve

The moving-average reward increases across episodes, showing the policy improves over time.

(Include your PNG here after pushing to GitHub)

---

## 🧪 Evaluation (Naive vs RAG vs RL-RAG)

The evaluation script compares:

- **Naive LLM (no retrieval)**
- **Fixed RAG (3000-char context)**
- **RL-RAG (adaptive context from Q-table)**

Example summary:

Naive LLM reward : 0.455
Plain RAG reward : 0.682
RL-RAG reward : 0.682 (optimal action selected)


RL-RAG consistently chooses the best context window and improves reliability.

---

## 🎤 Real User CLI Demonstration

User can type **any natural question** about prompt engineering.

Pipeline:
1. Semantic router maps question category  
2. RL chooses best context length  
3. RAG retrieves supporting text  
4. LLM produces grounded answer  

Example interaction:

Your question: what is role-based prompting?
[ROUTER] mapped → Q4
[CONTROLLER] Selected action = long (5000)
[ANSWER] Role-based prompting is..


---

## 📁 Project Structure

src/
│── rag_pipeline.py
│── rl_env.py
│── train_q_learning.py
│── eval_trained_policy_cli.py
│── real_user_cli_runner.py
│── semantic_router.py
data/
│── (PDF files used to build the knowledge base)
models/
│── q_table.npy
│── training_rewards.npy
README.md


---

## 🔧 Installation

```bash
git clone https://github.com/<your-repo>/genai_research_agent.git
cd genai_research_agent
pip install -r requirements.txt

▶️ How to Run
1️⃣ Train Q-learning
python -m src.train_q_learning
2️⃣ Evaluate Naive vs RAG vs RL-RAG
python -m src.eval_trained_policy_cli
3️⃣ Run Real-User CLI
python -m src.real_user_cli_runner

🧩 Future Work

Multi-agent orchestration (CrewAI / LangChain agents)

Add LORA fine-tuning for improved factual grounding

Add online learning (RL that updates with user feedback)

Add knowledge graphs for deeper reasoning

Add Streamlit web UI

🛡 Ethical Considerations

No personal data is collected

RL does not store user queries or identities

System avoids hallucinations via RAG grounding

Safety prompts prevent harmful outputs

👨‍💻 Author

Abhiram Varanasi
Northeastern University
