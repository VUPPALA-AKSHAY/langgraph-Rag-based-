import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

try:
    import streamlit as st
    _st_secrets = dict(st.secrets)
except Exception:
    _st_secrets = {}

def _s(key, default=""):
    return os.getenv(key) or _st_secrets.get(key, default)

BASE_DIR = Path(__file__).parent
PDF_PATH = Path(_s("PDF_PATH", "data/Ebook-Agentic-AI-real.pdf"))

GEMINI_API_KEY = _s("GEMINI_API_KEY", "")
EMBEDDING_MODEL = _s("EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIM = int(_s("EMBEDDING_DIM", "768"))

PINECONE_API_KEY = _s("PINECONE_API_KEY", "")
PINECONE_INDEX = _s("PINECONE_INDEX", "agentic-ai-ebook")
PINECONE_CLOUD = _s("PINECONE_CLOUD", "aws")
PINECONE_REGION = _s("PINECONE_REGION", "us-east-1")
PINECONE_METRIC = "cosine"

GROQ_API_KEY = _s("GROQ_API_KEY", "")
GROQ_MODEL = _s("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_TEMPERATURE = float(_s("GROQ_TEMPERATURE", "1"))
GROQ_MAX_TOKENS = int(_s("GROQ_MAX_TOKENS", "2048"))
GROQ_TOP_P = float(_s("GROQ_TOP_P", "1"))
GROQ_REASONING_EFFORT = _s("GROQ_REASONING_EFFORT", "medium")

CHUNK_SIZE = int(_s("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(_s("CHUNK_OVERLAP", "200"))
TOP_K = int(_s("TOP_K", "4"))
MEMORY_WINDOW = int(_s("MEMORY_WINDOW", "10"))

STRICT_SYSTEM_PROMPT = """You are a RAG assistant that answers STRICTLY from the provided context chunks taken from the book "Agentic AI eBook".

Rules:
1. Use ONLY the context below. Do not use outside knowledge.
2. If NONE of the chunks contain information related to the question, reply exactly: "I don't know based on the eBook — the question is outside the knowledge base." Then suggest which eBook chapter might cover it. If at least one chunk is relevant, you MUST answer — do not refuse.
3. Quote or cite chunk numbers like [Chunk 1] when you use them.
4. Be concise, factual, and helpful. No hallucination.
5. If the user greets you (hi/hello), greet back briefly and say what the eBook covers.
6. Use the conversation history only to resolve follow-ups (e.g. "its second part"); the FACTS must still come from the context chunks.
7. READ TABLES CAREFULLY: pay attention to column headers (e.g. LLMs vs Agents, Traditional AI vs Agentic AI). Never swap columns. If a property says "Needs prompts to function" under the LLMs column, it applies to LLMs, NOT to Agents.
8. For yes/no questions, start with Yes/No based solely on the chunks, then explain with citations.
9. For comparison or counting questions, keep lists separate and use only what appears in the chunks.
10. Think step-by-step: first identify which chunks are relevant, then synthesize the answer from them.
"""
