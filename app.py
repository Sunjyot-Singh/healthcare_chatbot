"""Healthcare Information Assistant - Streamlit frontend.

Run with: streamlit run app.py

Requires faiss_index.index and metadata.pkl (produced by the Colab data-pipeline notebook, Part A)
to be present alongside this file, or point FAISS_INDEX_PATH / METADATA_PATH env vars at them.
"""

import uuid

import streamlit as st
from groq import Groq

from backend import config
from backend.chat import chat_turn
from backend.retrieval import load_embed_model, load_index_and_metadata

st.set_page_config(page_title="Healthcare Information Assistant", page_icon="\U0001FA7A", layout="centered")

# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------
api_key = st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None
if not api_key:
    api_key = st.sidebar.text_input("Groq API key", type="password",
                                     help="Get a free key at console.groq.com/keys")
if not api_key:
    st.info("Enter your Groq API key in the sidebar to start chatting.")
    st.stop()

groq_client = Groq(api_key=api_key)

# ---------------------------------------------------------------------------
# Load knowledge base (cached - only loads once per server process)
# ---------------------------------------------------------------------------
try:
    embed_model = load_embed_model()
    index, metadata = load_index_and_metadata()
except FileNotFoundError:
    st.error(
        f"Couldn't find `{config.FAISS_INDEX_PATH}` / `{config.METADATA_PATH}`. "
        "Run Part A of the data-pipeline notebook and place both files next to app.py."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Multi-conversation session state
# ---------------------------------------------------------------------------
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "current_id" not in st.session_state:
    st.session_state.current_id = None


def new_chat():
    conv_id = str(uuid.uuid4())
    st.session_state.conversations[conv_id] = {
        "title": "New chat",
        "display_history": [],   # [{"role", "content", "debug"?}]
        "llm_plain_history": [],
    }
    st.session_state.current_id = conv_id


if not st.session_state.conversations:
    new_chat()

# ---------------------------------------------------------------------------
# Sidebar: conversation management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Conversations")
    if st.button("+ New Chat", use_container_width=True):
        new_chat()
        st.rerun()

    st.divider()
    for conv_id, conv in list(st.session_state.conversations.items()):
        col1, col2 = st.columns([5, 1])
        with col1:
            label = conv["title"]
            if st.button(label, key=f"select_{conv_id}", use_container_width=True,
                         type="primary" if conv_id == st.session_state.current_id else "secondary"):
                st.session_state.current_id = conv_id
                st.rerun()
        with col2:
            if st.button("\U0001F5D1", key=f"delete_{conv_id}", help="Delete this chat"):
                del st.session_state.conversations[conv_id]
                if st.session_state.current_id == conv_id:
                    st.session_state.current_id = None
                if not st.session_state.conversations:
                    new_chat()
                elif st.session_state.current_id is None:
                    st.session_state.current_id = next(iter(st.session_state.conversations))
                st.rerun()

current = st.session_state.conversations[st.session_state.current_id]

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("\U0001FA7A Healthcare Information Assistant")
st.warning(
    "**This is general health information, not a substitute for professional medical advice.** "
    "In a medical emergency, call your local emergency number immediately.",
    icon="\u26A0\uFE0F",
)

for msg in current["display_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("debug"):
            debug = msg["debug"]
            with st.expander("\U0001F50D View Retrieved Context / Sources"):
                st.markdown(f"**Path taken:** `{debug.get('path')}`")
                if debug.get("diagnosis_flag"):
                    st.markdown("_Diagnosis-seeking pattern detected in this question._")
                kept = debug.get("kept_chunks", [])
                if kept:
                    st.markdown("**Chunks used to answer:**")
                    for c in kept:
                        source_line = f" — [{c['source']}]({c['url']})" if c.get("url") else ""
                        st.markdown(f"- *{c['focus']}* (score {c['score']:.2f}){source_line}\n\n  {c['content']}")
                else:
                    st.markdown("_No chunks were used - either none were retrieved, or none passed "
                                "the relevance grader._")
                all_candidates = debug.get("all_candidates", [])
                if all_candidates:
                    st.markdown("**All FAISS candidates considered:**")
                    st.markdown(", ".join(f"{c['focus']} ({c['score']:.2f})" for c in all_candidates))

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
user_query = st.chat_input("Ask a general health question...")
if user_query:
    current["display_history"].append({"role": "user", "content": user_query})
    if current["title"] == "New chat":
        current["title"] = user_query[:40] + ("..." if len(user_query) > 40 else "")

    with st.spinner("Thinking..."):
        answer, debug, new_llm_history = chat_turn(
            user_query, current["llm_plain_history"], embed_model, index, metadata, groq_client
        )
    current["llm_plain_history"] = new_llm_history
    current["display_history"].append({"role": "assistant", "content": answer, "debug": debug})
    st.rerun()
