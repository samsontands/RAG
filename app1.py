# app.py — minimal chat + Google Drive link (Streamlit Secrets)

import os
import uuid
import json
import streamlit as st

# If your rag module (or SDKs) read settings from env at import time,
# copy from st.secrets → os.environ BEFORE importing rag.
def _env_from_secrets(keys: list[str]):
    for k in keys:
        v = st.secrets.get(k)
        if v:
            os.environ[k] = str(v)

_env_from_secrets([
    "OPENAI_API_KEY",         # if your rag stack needs it
    "PATHWAY_HOST",           # if rag reads this at import-time
    "ANTHROPIC_API_KEY",      # optional
    "AZURE_OPENAI_API_KEY",   # optional
    "AZURE_OPENAI_ENDPOINT",  # optional
])

# Import your RAG pieces AFTER env is set
from rag import DEFAULT_PATHWAY_HOST, PATHWAY_HOST, chat_engine  # noqa: E402

st.set_page_config(page_title="Realtime Document Chat", page_icon="🗂️")

# -------- Sidebar: Google Drive link only --------
DRIVE_URL = st.secrets.get(
    "GDRIVE_FOLDER_URL",
    "https://drive.google.com/drive/u/0/folders/1cULDv2OaViJBmOfG5WB0oWcgayNrGtVs",
)

with st.sidebar:
    st.markdown("### Add Your Files to Google Drive")
    st.write(f"➡️ [Open the Google Drive folder and upload files]({DRIVE_URL})")
    if os.environ.get("PATHWAY_HOST", "") == DEFAULT_PATHWAY_HOST:
        st.caption("These go to the public Pathway sandbox. Don’t upload confidential files.")
    else:
        st.caption(f"Connected to: {os.environ.get('PATHWAY_HOST')}")

# -------- Simple header --------
st.write("## Chat with your documents (Google Drive & SharePoint)")

# -------- Session bootstrapping --------
if "session_id" not in st.session_state:
    st.session_state.session_id = "uuid-" + str(uuid.uuid4())

if "messages" not in st.session_state:
    # Optional: seed a short default exchange so the window isn't empty
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me anything about your uploaded documents."}
    ]

# -------- Chat input --------
user_text = st.chat_input("Your question")
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})

# -------- Render chat history --------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# -------- Generate answer (only if last message is from user) --------
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            prompt = st.session_state.messages[-1]["content"]
            response = chat_engine.chat(prompt)

            # Try to collect short source names (optional)
            sources = []
            try:
                for node in getattr(response, "source_nodes", []) or []:
                    full_path = node.metadata.get("path", node.metadata.get("name"))
                    if full_path:
                        name = f"`{full_path.split('/')[-1]}`"
                        if name not in sources:
                            sources.append(name)
            except Exception:
                pass

            txt = response.response
            if sources:
                txt += "\n\nDocs consulted: " + ", ".join(sources)

            st.write(txt)
            st.session_state.messages.append({"role": "assistant", "content": txt})
