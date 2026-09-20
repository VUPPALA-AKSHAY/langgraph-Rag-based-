import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import config
from rag_graph import query_stream, get_last_result

MESSAGES_FILE = Path("chat_history.json")

def _save():
    with open(MESSAGES_FILE, "w", encoding="utf-8") as f:
        json.dump({"messages": st.session_state.messages,
                   "thread_id": st.session_state.thread_id}, f, ensure_ascii=False)

st.set_page_config(page_title="Agentic AI eBook RAG Chatbot", layout="wide")
st.title("Agentic AI eBook RAG Chatbot")
st.caption("LangGraph + Pinecone + Gemini embeddings + Groq gpt-oss-120b (streaming)")

if MESSAGES_FILE.exists():
    try:
        with open(MESSAGES_FILE, encoding="utf-8") as f:
            saved = json.load(f)
        st.session_state.messages = saved.get("messages", [])
        st.session_state.thread_id = saved.get("thread_id", "default")
    except Exception:
        st.session_state.messages = []
        st.session_state.thread_id = "default"
else:
    st.session_state.messages = []
    st.session_state.thread_id = "default"

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "default"

with st.sidebar:
    st.header("Knowledge base")
    try:
        from ingest import get_index
        n = get_index().describe_index_stats().get("total_vector_count", "?")
        st.success(f"Pinecone {config.PINECONE_INDEX} ready ({n} vectors)")
    except Exception as e:
        st.warning("Pinecone not reachable or not ingested yet.")
        st.code("python ingest.py", language="bash")
        st.caption(str(e)[:200])
    if st.button("Re-ingest PDF to Pinecone"):
        with st.spinner("Chunking + embedding + upserting..."):
            try:
                from ingest import ingest
                info = ingest()
                st.success(f"Indexed {info['chunks']} chunks from {info['pages']} pages")
            except Exception as e:
                st.error(f"Ingest failed: {e}")
    st.divider()
    if st.button("Clear Chat + Memory"):
        st.session_state.messages = []
        st.session_state.thread_id = "default"
        MESSAGES_FILE.unlink(missing_ok=True)
        st.rerun()
    if st.button("Clear Chat"):
        st.session_state.messages = []
        _save()
        st.rerun()

with st.expander("Try a sample query", expanded=False):
    st.markdown(
        "1. What is Agentic AI and how does it differ from traditional AI?\n"
        "2. What are the practical applications of Agentic AI in different industries?\n"
        "3. How is organizational readiness assessed for implementing Agentic AI?\n"
        "4. What are the key components and architecture of an Agentic AI system?\n"
        "5. What ethical considerations and compliance frameworks apply to Agentic AI?\n"
        "6. Which industries are adopting Agentic AI and what are the use cases?"
    )

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("confidence") is not None:
            st.caption(f"Confidence: {m['confidence']:.0%}")
            with st.expander("Retrieved context chunks"):
                for c in m.get("contexts", []):
                    st.markdown(f"Chunk {c['chunk_id']}")
                    st.caption(c["text"][:1200] + ("..." if len(c["text"]) > 1200 else ""))

q = st.chat_input("Ask anything from the Agentic AI eBook...")
if q:
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    missing = [k for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "PINECONE_API_KEY") if not (os.getenv(k) or config._st_secrets.get(k))]
    if missing:
        with st.chat_message("assistant"):
            st.warning(f"Missing keys in Secrets: {', '.join(missing)}")
    else:
        with st.chat_message("assistant"):
            try:
                answer = st.write_stream(
                    query_stream(q, top_k=config.TOP_K, thread_id=st.session_state.thread_id))
                res = get_last_result(st.session_state.thread_id)
                with st.expander("Retrieved context chunks"):
                    for c in res["contexts"]:
                        st.markdown(f"Chunk {c['chunk_id']}")
                        st.caption(c["text"][:1200] + ("..." if len(c["text"]) > 1200 else ""))
                st.session_state.messages.append({
                    "role": "assistant", "content": answer,
                    "confidence": res["confidence"], "contexts": res["contexts"]})
                _save()
            except Exception as e:
                st.error(f"Query failed: {e}")
