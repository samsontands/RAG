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
# (Removed stack logos block)

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

    pathway_explaination = (
        "Pathway is a high-throughput, low-latency data processing framework "
        "that handles live data & streaming for you."
    )
    DEFAULT_MESSAGES = [
        ChatMessage(role=MessageRole.USER, content="What is Pathway?"),
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

# ---- Show latest indexed files from the vector store ----
last_modified_time, last_indexed_files = get_inputs()
df = pd.DataFrame(last_indexed_files, columns=[last_modified_time, "status"])
if "status" in df.columns and df.status.isna().any():
    del df["status"]
df.set_index(df.columns[0])  # (left as-is, per your request)
st.dataframe(df, hide_index=True, height=150, use_container_width=True)

cs = st.columns([1, 1, 1, 1], gap="large")
with cs[-1]:
    st.button("⟳ Refresh", use_container_width=True)

# ---- Chat input ----
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

# ---- Render history ----
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ---- Generate answer ----
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
