import json
import logging
import os
import uuid

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from endpoint_utils import get_inputs
from llama_index.llms.types import ChatMessage, MessageRole
from log_utils import init_pw_log_config
from rag import DEFAULT_PATHWAY_HOST, PATHWAY_HOST, chat_engine, vector_client
from streamlit.web.server.websocket_headers import _get_websocket_headers
from traceloop.sdk import Traceloop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

init_pw_log_config()

# ---- Where users should upload (public demo folder by default) ----
DRIVE_URL = os.environ.get(
    "GDRIVE_FOLDER_URL",
    "https://drive.google.com/drive/u/0/folders/1cULDv2OaViJBmOfG5WB0oWcgayNrGtVs",
)

st.set_page_config(
    page_title="Realtime Document AI pipelines", page_icon="./app/static/favicon.ico"
)

# ---- Sidebar ----
with st.sidebar:
    if PATHWAY_HOST == DEFAULT_PATHWAY_HOST:
        st.markdown("**Add Your Files to Google Drive**")
        st.write(f"➡️ [Open the Google Drive folder and upload files]({DRIVE_URL})")
        st.markdown(
            "*These go to the **public Pathway sandbox**. Do not upload confidential files.*"
        )
        st.write("---")
    else:
        st.markdown(f"**Connected to:** {PATHWAY_HOST}")

    st.markdown(
        """**Ready to build your own?**

Our [docs](https://pathway.com/developers/showcases/llamaindex-pathway/) walk through creating custom pipelines with LlamaIndex."""
    )

# ---- Load .env (for OPENAI_API_KEY, etc.) ----
load_dotenv()

# ---- Header ----
st.write("## Chat with your Google Drive documents in real time ⚡")

# ---- Per-session setup ----
if "messages" not in st.session_state:
    if "session_id" not in st.session_state:
        session_id = "uuid-" + str(uuid.uuid4())
        logging.info(json.dumps({"_type": "set_session_id", "session_id": session_id}))
        Traceloop.set_association_properties({"session_id": session_id})
        st.session_state["session_id"] = session_id

    headers = _get_websocket_headers()
    logging.info(
        json.dumps(
            {
                "_type": "set_headers",
                "headers": headers,
                "session_id": st.session_state.get("session_id", "NULL_SESS"),
            }
        )
    )

    # Starter message
    pathway_explaination = (
        "RAG combines search and AI generation: your documents are split into chunks, "
        "embedded into vectors, stored in a vector index (Pathway), retrieved for your question, "
        "and sent to the LLM via LlamaIndex to generate a grounded answer."
    )
    DEFAULT_MESSAGES = [
        ChatMessage(role=MessageRole.USER, content="What is RAG?"),
        ChatMessage(role=MessageRole.ASSISTANT, content=pathway_explaination),
    ]
    chat_engine.chat_history.clear()
    for msg in DEFAULT_MESSAGES:
        chat_engine.chat_history.append(msg)

    st.session_state.messages = [
        {"role": msg.role, "content": msg.content} for msg in chat_engine.chat_history
    ]
    st.session_state.chat_engine = chat_engine
    st.session_state.vector_client = vector_client

# ==============================
# TABS: Chat first, Explainer second
# ==============================
tab_chat, tab_explain = st.tabs(["💬 Chat (RAG)", "📘 RAG Explainer"])

# ------------------------------
# TAB 1: CHAT (RAG)
# ------------------------------
with tab_chat:
    # Latest indexed files
    last_modified_time, last_indexed_files = get_inputs()
    df = pd.DataFrame(last_indexed_files, columns=[last_modified_time, "status"])
    if "status" in df.columns and df.status.isna().any():
        del df["status"]
    df.set_index(df.columns[0])  # (left as-is, per your request)
    st.dataframe(df, hide_index=True, height=150, use_container_width=True)

    cs = st.columns([1, 1, 1, 1], gap="large")
    with cs[-1]:
        st.button("⟳ Refresh", use_container_width=True)

    # Chat input
    prompt = st.chat_input("Your question")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        logging.info(
            json.dumps(
                {
                    "_type": "user_prompt",
                    "prompt": prompt,
                    "session_id": st.session_state.get("session_id", "NULL_SESS"),
                }
            )
        )

    # Render history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Generate answer
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chat_engine.chat(prompt)
                sources = []
                try:
                    for source in getattr(response, "source_nodes", []) or []:
                        full_path = source.metadata.get("path", source.metadata.get("name"))
                        if full_path:
                            name = f"`{full_path.split('/')[-1]}`"
                            if name not in sources:
                                sources.append(name)
                except AttributeError:
                    logging.error(
                        json.dumps(
                            {
                                "_type": "error",
                                "error": f"No source (`source_nodes`) found in response: {str(response)}",
                                "session_id": st.session_state.get("session_id", "NULL_SESS"),
                            }
                        )
                    )

                sources_text = ", ".join(sources)
                response_text = (
                    response.response
                    + (f"\n\nDocuments looked up to obtain this answer: {sources_text}" if sources else "")
                )
                st.write(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

# ------------------------------
# TAB 2: EXPLAINER
# ------------------------------
with tab_explain:
    with st.expander("🔍 What is RAG (Retrieval‑Augmented Generation)?", expanded=True):
        st.markdown(
            """
**RAG** combines **search** and **AI generation** so answers are current and source‑grounded.

**TL;DR flow:**  
**Documents → Embeddings → Vector DB/Index → Retriever → LlamaIndex → LLM Answer**
"""
        )

    sub_tabs = st.tabs(["Core Concepts", "How LlamaIndex Fits", "REST & Architecture"])

    with sub_tabs[0]:
        st.markdown(
            """
### 1) Documents  
Your PDFs, Google Docs, Word files, web pages—this is the knowledge base the AI will reference.

### 2) Embeddings  
We split documents into small **chunks** (sentences/paragraphs) and turn each chunk into a **vector** (a list of numbers that captures meaning).  
Similar meanings → vectors are close in space (e.g., “car” ≈ “automobile”).

### 3) Vector Database / Index  
Stores those vectors and enables **similarity search** (find by meaning, not exact words).  
In this app, **Pathway’s DocumentStore** acts as the vector DB/index and keeps itself **up‑to‑date** as files change.

### 4) Retriever  
The “librarian.” Given your question, it searches the vector index and returns the **most relevant chunks** plus metadata (like file name, path).

### 5) Generator (LLM)  
A large language model (e.g., GPT) uses **your question + retrieved chunks** to generate a grounded answer.
"""
        )

    with sub_tabs[1]:
        st.markdown(
            """
### LlamaIndex’s Role (Orchestration)
- Calls the **retriever** to fetch top‑k relevant chunks.
- Packages those chunks into a prompt for the **LLM**.
- Manages **chat history** and follow‑up question rewriting so retrieval stays on topic.

In this app, we use LlamaIndex with a **Pathway retriever**:
1. Your question → LlamaIndex rewrites/condenses if needed.  
2. LlamaIndex queries **Pathway** for relevant chunks.  
3. LlamaIndex passes chunks + your question to the LLM to produce the final answer.
"""
        )

    with sub_tabs[2]:
        st.markdown(
            """
### REST Integration & Live Architecture
- **Pathway** exposes a **REST API** for its DocumentStore (the vector index).
- The retriever connects to that endpoint, so your Streamlit app stays lightweight.
- **Live updates:** when files are added/edited/deleted in the watched folder, Pathway **re‑parses, re‑embeds, and re‑indexes** automatically—no manual ETL.

**Why this matters:**  
You get real‑time RAG without running a separate vector DB or cron jobs. Your data updates → your answers update.
"""
        )
