# 💬 Multi-User AI Chat System with Long-Term Memory

A multi-user AI chat system built with LangGraph, Large Language Models, and vector databases, designed to provide personalized, context-aware conversations with persistent memory across sessions.

The system supports multiple users and multiple chat threads per user, automatically extracting and storing relevant information to improve responses over time.

---

## 🚀 Features

- Multi-user support
    - Isolated chatbot instances per user
    - Secure user-specific memory and conversation state
- Multi-chat threads
    - Independent chat sessions per user
    - Persistent conversation history per chat
- Long-term memory with vector storage
    - Automatic extraction of relevant user information
    - Semantic search over stored memories
    - Memory categorization (personal, professional, preferences, important facts)
- Intelligent memory extraction
    - LLM-based smart memory extraction
    - Rule-based fallback extraction system
    - Importance-based memory filtering
- Context optimization
  - Dynamic context building using relevant memories
  - Smart message trimming to fit LLM context limits
- LangGraph-based conversational workflow
    - Modular graph nodes:
        - Memory retrieval
        - Context optimization
        - Response generation
        - Memory extraction
    - State persistence using SQLite checkpoints
- Session persistence
    - Conversation state stored across app restarts
    - Per-chat thread state management
- Streamlit web interface
    - Real-time chat interface
    - Chat history visualization
    - User and chat management
- Configurable LLM setup
    - Environment-based configuration
    - Model and temperature control

---

## 🧰 Tech Stack

- **Python**
- **Streamlit**
- **LangChain & LangGraph**
- **Vector Database**
- **SQLite**
- **OpenAI API** 
- **Pydantic** 
- **uv** 

---

## 🔐 Environment Variables

The app requires an OpenAI-compatible API key.

Create a `.env` file in the project root with:

```bash
OPENAI_API_KEY=your_api_key_here
```
---

## 🧑‍💻 Run Locally (without Docker, using `uv`)

This option is recommended for local development.

### 1️⃣ Install `uv` (if you don’t have it)

Using **Git Bash**, Linux, or macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Restart your terminal after installation.

### 2️⃣ Install dependencies

```bash
uv sync
```

### 3️⃣ Set environment variables

Ensure your `.env` file exists and contains your API key.

### 4️⃣ Run the app

```bash
uv run streamlit run app.py
```

Then open:

👉 http://localhost:8501

---

