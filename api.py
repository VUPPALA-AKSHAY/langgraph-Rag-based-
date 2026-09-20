from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import config
from rag_graph import query

app = FastAPI(title="Agentic AI eBook RAG API")

class ChatRequest(BaseModel):
    question: str
    top_k: int = config.TOP_K
    session_id: str = "default"

@app.get("/health")
def health():
    try:
        from ingest import get_index
        stats = get_index().describe_index_stats()
        indexed, vectors = True, stats.get("total_vector_count", 0)
    except Exception as e:
        indexed, vectors, e = False, 0, str(e)[:200]
    return {"status": "ok", "indexed": indexed, "vectors": vectors,
            "index": config.PINECONE_INDEX, "llm": config.GROQ_MODEL,
            "embeddings": config.EMBEDDING_MODEL, "memory_window": config.MEMORY_WINDOW,
            "error": None if indexed else e}

@app.post("/chat")
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(400, "question must not be empty")
    try:
        return query(req.question, top_k=req.top_k, thread_id=req.session_id)
    except Exception as e:
        raise HTTPException(500, f"RAG query failed: {e}")
