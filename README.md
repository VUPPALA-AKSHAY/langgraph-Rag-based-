# Agentic AI eBook — RAG Chatbot

> **Live demo:** [https://m4tur8pq8krtgt7it5tbbt.streamlit.app](https://m4tur8pq8krtgt7it5tbbt.streamlit.app)

This is a fully working Retrieval-Augmented Generation (RAG) chatbot that answers questions **strictly from the real Agentic AI eBook PDF**. It pulls relevant passages from a vector database, passes them to a large language model, and streams the answer back to you — along with which chunks it used and a confidence score.

## What it does

- **Ask a question** in the chat UI and get an answer sourced only from the eBook
- **Streaming responses** — the answer appears word by word in real time
- **Confidence scoring** — each answer includes a score based on how well the retrieved chunks match the question
- **Retrieved context view** — click "Retrieved context chunks" to see exactly which parts of the eBook the bot referenced
- **Conversation memory** — the chat remembers the last few messages so follow-up questions make sense
- **Off-topic guardrail** — if the question is outside the eBook's coverage, it explicitly says so instead of making things up

## Tech Stack

| Component | Technology |
|---|---|
| **Vector Database** | Pinecone serverless — index `agentic-ai-ebook`, 768 dimensions, cosine similarity |
| **Embeddings** | Google Gemini `gemini-embedding-001` — converts text into 768-dim vectors for search |
| **Orchestration** | LangGraph — a `retrieve → generate → score` pipeline with memory persistence |
| **LLM** | Groq `openai/gpt-oss-120b` — ultra-fast inference with streaming at temperature 1 |
| **Frontend UI** | Streamlit — interactive chat interface in the browser |
| **Backend API** | FastAPI — REST endpoints `POST /chat` and `GET /health` for programmatic access |
| **PDF Processing** | PyPDF — loads and extracts text from the eBook PDF |
| **Text Splitting** | LangChain Text Splitters — chunks the eBook into 1000-character segments with 200-char overlap |
| **Environment** | python-dotenv — manages API keys from a `.env` file |

## How the architecture works

When you type a question, here is what happens behind the scenes:

1. **Retrieve** — The question gets expanded into related sub-queries, each one is embedded by Gemini, and Pinecone searches for the top-k most similar text chunks from the eBook
2. **Generate** — A system prompt (with strict rules about only using the eBook), your conversation history, and the retrieved chunks are assembled into a prompt and sent to Groq's GPT-oss-120b model. The answer streams back token by token
3. **Score** — The system calculates a confidence score based on the best chunk's similarity score and how many chunks were relevant to the answer
4. **Memory** — The last few messages are persisted so the bot can handle follow-up questions like "its second part" or "in the previous section"
5. **Persistence** — Messages and thread IDs are saved to `chat_history.json` so conversations survive page reloads

The whole pipeline is built as a LangGraph state machine with three nodes — retrieve, generate, and score — connected in sequence, with a MemorySaver checkpoint in between.

## Setup instructions

```powershell
# Clone the repo
cd Task_r

# Install all dependencies
pip install -r requirements.txt

# Copy the example env file and add your API keys
Copy-Item .env.example .env
# Edit .env with: GEMINI_API_KEY, GROQ_API_KEY, PINECONE_API_KEY
# Get keys from: https://aistudio.google.com | https://console.groq.com | https://app.pinecone.io
```

## Ingest the eBook into Pinecone

Before you can chat, the eBook needs to be chunked, embedded, and uploaded to Pinecone:

```powershell
python ingest.py
# Creates the serverless index if it doesn't exist, then upserts all chunks with metadata
```

You can also click **"Re-ingest PDF to Pinecone"** in the app sidebar after deploying.

## Run the app

```powershell
# Chat UI
streamlit run app.py
# Or double-click run_app.bat

# REST API (separate terminal)
uvicorn api:app --reload --port 8000
```

The chat UI is at `http://localhost:8501` and the API is at `http://localhost:8000`.

## Sample queries to try

Here are six questions you can ask the chatbot to see how it works:

1. **"What is Agentic AI and how does it differ from traditional AI?"** — Tests whether the bot can explain the core distinction between regular chatbots and agentic systems that can perceive, reason, plan, and act

2. **"What are the practical applications of Agentic AI in different industries?"** — Checks if the bot can pull industry-specific use cases and examples from the eBook

3. **"What are the key components of an Agentic AI system?"** — Verifies the bot can list and describe the five core components: model, tools, memory, planner, and guardrails

4. **"How does the ReAct pattern work and when should it be used?"** — Tests the bot's ability to explain the Reason + Act cycle and contrast it with plan-and-execute approaches

5. **"What ethical considerations and compliance frameworks apply to Agentic AI?"** — Checks if the bot addresses the governance and safety aspects discussed in the eBook

6. **"What is the difference between the core pillars and the structural layers in Agentic AI?"** — Tests the bot's ability to handle nuanced comparison questions and avoid swapping columns in tables

## API response format

The REST API returns structured JSON:

```json
{
  "answer": "…",
  "contexts": [
    {
      "chunk_id": 1,
      "text": "…",
      "score": 0.83,
      "page": 12
    }
  ],
  "confidence": 0.81
}
```

## Push to GitHub

```powershell
git add .
git commit -m "RAG chatbot: LangGraph + Pinecone + Gemini + Groq"
git push
# The .env file is gitignored so no keys are exposed
# Submit the repo link in your deployment form
```

## Project structure

```
Task_r/
├── app.py                  # Streamlit chat UI
├── api.py                  # FastAPI REST server
├── rag_graph.py            # LangGraph pipeline (retrieve → generate → score)
├── config.py               # Environment variable and secret configuration
├── embeddings.py           # Gemini embedding functions
├── ingest.py               # PDF chunking, embedding, and Pinecone upsertion
├── requirements.txt        # All Python dependencies
├── .env.example            # Template for API keys (no real keys included)
├── .gitignore              # Keeps .env and .streamlit/ out of the repo
├── data/
│   ├── Ebook-Agentic-AI-real.pdf    # The full eBook
│   └── Agentic_AI_eBook.pdf         # Demo version created by create_sample_pdf.py
├── create_sample_pdf.py    # Generates a demo PDF if you need a placeholder
└── run_app.bat             # Quick launch script for Streamlit
```
