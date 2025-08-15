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

# ---- Sidebar (Google Drive only in UI) ----
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

    # Keep generic docs, no SharePoint mentions
    st.markdown(
        """**Ready to build your own?**

Our [docs](https://pathway.com/developers/showcases/llamaindex-pathway/) walk through creating custom pipelines with LlamaIndex.
"""
    )

# ---- Load .env (for OPENAI_API_KEY, etc.) ----
load_dotenv()

# ---- Header (no SharePoint; no stack logos) ----
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
