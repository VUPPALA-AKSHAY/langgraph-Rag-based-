import time
from typing import List
from google import genai
from google.genai import types

import config

def _client() -> genai.Client:
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing - set it in .env.")
    return genai.Client(api_key=config.GEMINI_API_KEY)

def _embed_batch(texts: List[str], task_type: str) -> List[List[float]]:
    client = _client()
    out: List[List[float]] = []
    for i in range(0, len(texts), 10):
        batch = [t if t.strip() else " " for t in texts[i:i + 10]]
        for attempt in range(6):
            try:
                res = client.models.embed_content(
                    model=config.EMBEDDING_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=config.EMBEDDING_DIM,
                    ),
                )
                out.extend([list(e.values) for e in res.embeddings])
                break
            except Exception as e:
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    wait = 50
                    print(f"Quota hit — sleeping {wait}s (attempt {attempt + 1}/6)...")
                    time.sleep(wait)
                elif attempt == 5:
                    raise
                else:
                    time.sleep(2 ** attempt)
        time.sleep(1)
    return out

def embed_documents(texts: List[str]) -> List[List[float]]:
    vecs = _embed_batch(texts, "RETRIEVAL_DOCUMENT")
    assert all(len(v) == config.EMBEDDING_DIM for v in vecs), "Embedding dim mismatch vs Pinecone index!"
    return vecs

def embed_queries(texts: List[str]) -> List[List[float]]:
    return _embed_batch(texts, "RETRIEVAL_QUERY")

def embed_query(text: str) -> List[float]:
    return _embed_batch([text], "RETRIEVAL_QUERY")[0]
