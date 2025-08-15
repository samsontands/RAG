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
        st.write(
            f"➡️ [Open the Google Drive folder and upload files]({DRIVE_URL})"
        )
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

# ---- Header / badges ----
st.write("## Chat with your Google Drive documents in real time ⚡")

# ==============================
# RAG EXPLAINER (simple + complete)
# ==============================
with st.expander("🔍 What is RAG (Retrieval‑Augmented Generation)?", expanded=True):
    st.markdown(
        """
**RAG** combines **search** and **AI generation** so answers are current and source‑grounded.

**TL;DR flow:**  
**Documents → Embeddings → Vector DB/Index → Retriever → LlamaIndex → LLM Answer**
"""
    )

tabs = st.tabs(["Core Concepts", "How LlamaIndex Fits", "REST & Architecture"])

with tabs[0]:
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

with tabs[1]:
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

with tabs[2]:
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

st.write("---")

# ---- Per-session setup ----
if "messages" not in st.session_state:
    if "session_id" not in st.session_state:
        session_id
