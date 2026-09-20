import os
from typing import TypedDict, List, Dict, Any, Generator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

import config
from embeddings import embed_query

_INDEX = None

def get_pinecone_index():
    global _INDEX
    if _INDEX is None:
        from pinecone import Pinecone
        if not config.PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY is missing - set it in .env.")
        _INDEX = Pinecone(api_key=config.PINECONE_API_KEY).Index(config.PINECONE_INDEX)
    return _INDEX

def get_llm(streaming=False):
    from langchain_groq import ChatGroq
    if not config.GROQ_API_KEY and not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is missing - set it in .env.")
    return ChatGroq(model=config.GROQ_MODEL, temperature=config.GROQ_TEMPERATURE,
                    max_tokens=config.GROQ_MAX_TOKENS,
                    reasoning_effort=config.GROQ_REASONING_EFFORT,
                    model_kwargs={"top_p": config.GROQ_TOP_P},
                    streaming=streaming)

class RAGState(TypedDict):
    question: str
    top_k: int
    messages: List[Dict[str, str]]
    retrieved: List[Dict[str, Any]]
    answer: str
    confidence: float

def _human_page(p) -> str:
    try:
        return str(int(float(p)) + 1)
    except (TypeError, ValueError):
        return str(p)

def _expand_queries(question: str) -> List[str]:
    import re
    q = question.strip()
    queries = [q]
    low = q.lower()
    if "pillar" in low and "layer" in low:
        queries.extend([
            "core pillars Perception Reasoning Planning Learning Execution production line AI example",
            "structural layers MAS Perception Representation Decision-Making Planning Action Interaction Learning",
            "2.1 The Core Pillars From Perception to Execution Agentic AI systems function like a well",
            "3.1 Structural Layers of a Multi-Agent System Perception Representation Decision-Making Planning Action Interaction Learning",
            "core pillars vs structural layers MAS difference manufacturing example",
            "Learning Continuous Improvement production line AI learns to recognize emerging defect patterns continuous improvement",
            "Learning pillar Perception Reasoning Planning Execution definition example",
        ])
    if "chef" in low:
        queries.append("A Chef Expertly manages every task Agentic AI analogy Chef Coach Coordinator Project Manager")
    if "sarah" in low:
        queries.append("Sarah entrepreneur proactive AI assistant")
    if "supply chain" in low or "rerout" in low:
        queries.append("supply chain MAS inventory reroute demand surge")
    if "agentic ai stand apart" in low or "stand apart" in low:
        queries.append("How Agentic AI Stands Apart Learns Continuously Focuses on Goals Acts Independently")
    is_complex = len(q) > 75 or any(x in low for x in
        ["compare", " and ", " vs ", " vs.", " with ", " about ", " difference", " versus", " how many", " count"])
    if is_complex:
        parts = [p.strip(" .?\"'") for p in re.split(r'\b and \b|\b with \b|\b compare\b|\b vs\.?\b|,\s*', q, flags=re.I)]
        for p in parts:
            if 15 < len(p) < len(q) and p not in queries:
                queries.append(p)
    seen, uniq = set(), []
    for x in queries:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq[:10]

def pinecone_retrieve(question: str, k: int) -> List[Dict[str, Any]]:
    queries = _expand_queries(question)
    from embeddings import embed_queries
    qvecs = embed_queries(queries)
    seen = {}
    for qvec in qvecs:
        res = get_pinecone_index().query(vector=qvec, top_k=k, include_metadata=True)
        for m in (res.matches or []):
            mid = getattr(m, "id", None) or m.metadata.get("chunk_id", "")
            key = str(mid) if mid != "" else m.metadata.get("text", "")[:80]
            score = float(m.score or 0.0)
            if key not in seen or score > seen[key]["score"]:
                seen[key] = {"text": m.metadata.get("text", ""),
                             "score": round(score, 4),
                             "page": _human_page(m.metadata.get("page", "0")),
                             "source": m.metadata.get("source", "")}
    ranked = sorted(seen.items(), key=lambda kv: (kv[1]["score"], float(kv[1]["page"]), kv[0]), reverse=True)[:k]
    ranked = [v for _, v in ranked]
    for i, c in enumerate(ranked):
        c["chunk_id"] = i + 1
    return ranked

def build_prompt(question: str, retrieved: List[Dict], messages: List[Dict]) -> str:
    context_block = "\n\n".join(
        f"[Chunk {c['chunk_id']} | p.{c['page']} | score={c['score']}]\n{c['text']}"
        for c in retrieved
    ) or "(no context retrieved)"
    history = "\n".join(f"{m['role']}: {m['content']}" for m in messages[-config.MEMORY_WINDOW:]) \
        or "(no prior conversation)"
    return (f"{config.STRICT_SYSTEM_PROMPT}\n\n"
            f"CONVERSATION HISTORY (for follow-up context only):\n{history}\n\n"
            f"CONTEXT:\n{context_block}\n\nQUESTION: {question}\nANSWER:")

def trim_messages(messages: List[Dict], question: str, answer: str) -> List[Dict]:
    return ([*messages, {"role": "user", "content": question},
             {"role": "assistant", "content": answer}])[-config.MEMORY_WINDOW:]

def score_contexts(retrieved: List[Dict], answer: str) -> float:
    scores = [c["score"] for c in retrieved]
    if not scores:
        return 0.0
    best = max(scores)
    above = sum(1 for s in scores if s > 0.5)
    coverage = above / len(scores)
    confidence = round(min(0.98, best * 0.7 + coverage * 0.3), 4)
    if "know based on the ebook" in (answer or "").lower():
        confidence = round(min(confidence, 0.25), 4)
    return confidence

def retrieve_node(state: RAGState) -> RAGState:
    state["retrieved"] = pinecone_retrieve(state["question"], state.get("top_k") or config.TOP_K)
    return state

def generate_node(state: RAGState) -> RAGState:
    prompt = build_prompt(state["question"], state["retrieved"], state.get("messages", []))
    answer = get_llm().invoke(prompt).content
    state["answer"] = answer
    state["messages"] = trim_messages(state.get("messages", []), state["question"], answer)
    return state

def score_node(state: RAGState) -> RAGState:
    state["confidence"] = score_contexts(state.get("retrieved", []), state.get("answer", ""))
    return state

def build_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_node("score", score_node)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "score")
    g.add_edge("score", END)
    return g.compile(checkpointer=MemorySaver())

_GRAPH = None
_LAST: Dict[str, Dict[str, Any]] = {}

def _graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH

def _cfg(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id or "default"}}

def query(question: str, top_k=None, thread_id="default") -> Dict[str, Any]:
    out: RAGState = _graph().invoke(
        {"question": question, "top_k": top_k or config.TOP_K,
         "retrieved": [], "answer": "", "confidence": 0.0},
        config=_cfg(thread_id))
    res = {"answer": out["answer"], "contexts": out["retrieved"], "confidence": out["confidence"]}
    _LAST[thread_id or "default"] = res
    return res

def query_stream(question: str, top_k=None, thread_id="default") -> Generator[str, None, None]:
    tid = thread_id or "default"
    k = top_k or config.TOP_K
    retrieved = pinecone_retrieve(question, k)
    snap = _graph().get_state(_cfg(tid))
    history = (snap.values or {}).get("messages", [])
    prompt = build_prompt(question, retrieved, history)

    parts: List[str] = []
    for chunk in get_llm(streaming=True).stream(prompt):
        tok = chunk.content or ""
        if tok:
            parts.append(tok)
            yield tok

    answer = "".join(parts)
    _graph().update_state(_cfg(tid), {
        "question": question, "top_k": k,
        "messages": trim_messages(history, question, answer),
        "retrieved": retrieved, "answer": answer,
        "confidence": score_contexts(retrieved, answer)})
    _LAST[tid] = {"answer": answer, "contexts": retrieved,
                  "confidence": score_contexts(retrieved, answer)}

def get_last_result(thread_id="default") -> Dict[str, Any]:
    return _LAST.get(thread_id or "default", {"answer": "", "contexts": [], "confidence": 0.0})
