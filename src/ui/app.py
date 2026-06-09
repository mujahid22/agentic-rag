import re
import base64
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

import streamlit as st
from PIL import Image
from src.agent.tool_selector import select_tools
from src.agent.graph import stream_query
from src.utils.query_logger import load_queries

_AVATAR_PATH = Path(__file__).parent / "assets" / "mira_new.PNG"
_BG_PATH = Path(__file__).parent / "assets" / "mira_background.png"
MIRA_AVATAR = Image.open(_AVATAR_PATH) if _AVATAR_PATH.exists() else "assistant"
_BG_B64 = base64.b64encode(_BG_PATH.read_bytes()).decode() if _BG_PATH.exists() else ""

# Matches [Source: ... | Section: ... | Effective: ...]
_CITATION_RE = re.compile(r'\[Source:[^\]]+\]')

_CITATION_STYLE = (
    "font-family: 'Courier New', Courier, monospace; "
    "color: #999999; "
    "font-size: 0.82em;"
)

def _render(text: str) -> str:
    """Wrap citation blocks in a styled span for rendering."""
    return _CITATION_RE.sub(
        lambda m: f'<span style="{_CITATION_STYLE}">{m.group()}</span>',
        text,
    )

def _inject_background() -> None:
    if not _BG_B64:
        return
    st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image:
        linear-gradient(rgba(240, 240, 240, 0.94), rgba(240, 240, 240, 0.94)),
        url("data:image/png;base64,{_BG_B64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}
[data-testid="stMain"] {{
    background-color: transparent;
}}
[data-testid="stHeader"] {{
    background-color: transparent;
}}
</style>
""", unsafe_allow_html=True)

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Meridian Airlines — AI Assistant",
    page_icon="✈",
    layout="centered",
)
_inject_background()

# ── Session state ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "session_start" not in st.session_state:
    from datetime import datetime
    st.session_state.session_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

_SAMPLE_QUERIES = [
    "Can I cancel my Economy Saver ticket?",
    "How many kg of baggage in Business class?",
    "My flight MA-204 was delayed 4 hours — what am I owed?",
    "How do I earn Meridian Miles on partner flights?",
    "Can I bring my infant on board without a seat?",
    "What documents do I need to check in?",
    "What is the lounge access policy for Gold members?",
    "Do I need a visa to transit through a Meridian hub?",
]

# ── Admin dialog ──────────────────────────────────────────────
@st.dialog("About MIRA", width="large")
def _admin_dialog():
    tab_arch, tab_logs = st.tabs(["Architecture", "Query Logs"])

    with tab_arch:
        col_left, col_right = st.columns([3, 2])
        with col_left:
            st.markdown("### System Architecture")
            st.markdown(
                """
| Component | Technology |
|---|---|
| Agent framework | LangGraph (ReAct loop) |
| Generation model | Llama 3.1 8B — NVIDIA NIM |
| Embedding model | NV-Embed-V1 4096-dim — NVIDIA NIM |
| Vector database | Pinecone Serverless (aws / us-east-1, cosine) |
| Lexical search | BM25 — bm25s 0.3.9 |
| Re-ranker | Pinecone Rerank V0 |
| Retrieval strategy | Hybrid BM25 + Vector with confidence-based rerank skip |
| Topic tools | 16 section-scoped LangChain tools |
| UI | Streamlit 1.58.0 |
                """
            )
        with col_right:
            st.markdown("### Session Stats")
            st.metric("Queries this session", st.session_state.query_count)
            st.metric("Documents indexed", 55)
            st.metric("Vectors in Pinecone", 395)

    with tab_logs:
        st.markdown("### Current Session Query Logs")
        queries = load_queries(since=st.session_state.session_start)
        if not queries:
            st.info("No queries in this session yet. Ask MIRA something to see the trace here.")
        else:
            st.caption(f"{len(queries)} queries this session · newest first")
            for q in queries:
                tools_label = ", ".join(
                    t.replace("tool_", "").replace("_", " ").title()
                    for t in q["tools"]
                ) or "—"
                rt = f"{q['response_time_s']}s" if q["response_time_s"] else "—"
                header = f"🕐 {q['timestamp']}  |  ⏱ {rt}  —  {q['query'][:60]}{'…' if len(q['query']) > 60 else ''}"
                with st.expander(header):
                    st.markdown(f"**Query:** {q['query']}")
                    st.markdown(f"**Tools pre-selected:** {tools_label}")
                    if q["retrievals"]:
                        st.markdown("**Retrieval calls:**")
                        for r in q["retrievals"]:
                            route = r["route"]
                            score = f"{r['top_score']:.4f}" if r["top_score"] else "N/A (BM25)"
                            reranked = "✅ reranked" if r["reranked"] else "⚡ skipped (high confidence)"
                            st.markdown(
                                f"- Route: **{route}** · Top score: `{score}` · Rerank: {reranked} · Results: {r['results_returned']}"
                            )
                    st.markdown(f"**Response time:** {rt}")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    if _AVATAR_PATH.exists():
        _sb_av_b64 = base64.b64encode(_AVATAR_PATH.read_bytes()).decode()
        st.markdown(
            f'<div style="display:flex; justify-content:center; margin-bottom:8px;">'
            f'<img src="data:image/png;base64,{_sb_av_b64}" width="220" style="border-radius: 16px;">'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        """
<div style="text-align:center; line-height:1.6;">
  <p style="
      font-family: 'Georgia', 'Times New Roman', serif;
      font-size: 1.25em;
      font-weight: bold;
      color: #1a3a6c;
      letter-spacing: 0.06em;
      text-shadow: 1px 1px 3px rgba(0,0,0,0.12);
      margin: 6px 0 2px;
  ">✨ MIRA ✨</p>
  <p style="font-family:'Georgia',serif; font-size:0.9em; color:#4a6491; letter-spacing:0.12em; text-transform:uppercase; margin:0 0 10px;">Meridian Intelligent Route Assistant</p>
  <hr style="border:none; border-top:1px solid #ccc; margin:8px 0;">
  <p style="font-size:0.82em; color:#555; margin:4px 0; text-align:left;">⚡️ Powered by Agentic RAG</p>
  <p style="font-size:0.82em; color:#555; margin:4px 0; text-align:left;">📚 Grounded answers from official Meridian Airlines documentation</p>
  <hr style="border:none; border-top:1px solid #ccc; margin:8px 0;">
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<style>
section[data-testid="stSidebar"] {
    box-shadow: 4px 0 16px rgba(0, 0, 0, 0.18) !important;
}
/* Contact block: fixed to bottom; width:inherit takes the computed
   width from the DOM parent (sidebar content area) not the viewport,
   so it automatically matches the actual sidebar width on any screen. */
#sidebar-contact-block {
    position: fixed !important;
    bottom: 1rem !important;
    left: 1rem !important;
    width: inherit !important;
    z-index: 998 !important;
}
/* About MIRA button — visually relocated to top-right header */
section[data-testid="stSidebar"] .stButton > button {
    position: fixed !important;
    top: 14px !important;
    right: 1.2rem !important;
    left: auto !important;
    bottom: auto !important;
    width: auto !important;
    padding: 5px 16px !important;
    border-radius: 20px !important;
    background-color: rgba(255, 255, 255, 0.88) !important;
    border: 1.5px solid #1a3a6c !important;
    color: #1a3a6c !important;
    font-size: 0.82em !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.14) !important;
    z-index: 200 !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: rgba(26, 58, 108, 0.1) !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div id="sidebar-contact-block">
  <div style="border-top:1px solid #ccc; padding-top:10px; text-align:center;">
    <p style="font-family:'Georgia','Times New Roman',serif; font-size:1.1em; font-weight:bold; color:#1a3a6c; letter-spacing:0.06em; text-shadow:1px 1px 3px rgba(0,0,0,0.12); margin:0 0 6px; text-align:center;">✈️ Meridian Airlines</p>
    <div style="font-family:'Segoe UI',Arial,sans-serif; font-size:0.78em; color:#555; line-height:2;">
      <div style="text-align:center;">🌐 <a href="https://www.meridianair.com" style="color:#1a3a6c; text-decoration:none;">www.meridianair.com</a></div>
      <div style="text-align:center;">📧 <a href="mailto:support@meridianair.com" style="color:#1a3a6c; text-decoration:none;">support@meridianair.com</a></div>
      <div style="text-align:center;">📞 +1-800-637-4326</div>
      <div style="text-align:center;">📍 1 Meridian Plaza, Dallas, TX 75201</div>
    </div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("ℹ️  About MIRA"):
        _admin_dialog()


# ── Meridian Airlines header (always visible, fixed at top) ───
_bg_css = (
    f"background-image: linear-gradient(rgba(240,240,240,0.94),rgba(240,240,240,0.94)),"
    f"url('data:image/png;base64,{_BG_B64}');"
    "background-size: cover; background-position: center; background-attachment: fixed;"
) if _BG_B64 else "background: rgb(240,240,240);"

st.markdown(
    f"""
<style>
  /* Push content below the fixed header (~72px tall) */
  .block-container {{ padding-top: 78px !important; }}

  /* Chat bubble backgrounds — solid so the app background doesn't bleed through */
  [data-testid="stChatMessageContent"] {{
      background-color: rgba(255, 255, 255, 0.96) !important;
      border-radius: 12px !important;
      padding: 10px 14px !important;
      box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08) !important;
  }}
</style>
<div style="
    position: fixed;
    top: 0; left: 21rem; right: 0;
    z-index: 100;
    {_bg_css}
    text-align: center;
    padding: 6px 0 10px;
">
  <p style="
      font-family: 'Georgia', 'Times New Roman', serif;
      font-size: 2.2em;
      font-weight: bold;
      color: #1a3a6c;
      letter-spacing: 0.06em;
      margin: 0 0 2px;
      text-shadow: 1px 1px 3px rgba(0,0,0,0.12);
  ">✈️ Meridian Airlines</p>
  <p style="
      font-family: 'Georgia', serif;
      font-size: 0.95em;
      color: #4a6491;
      letter-spacing: 0.12em;
      margin: 0;
      text-transform: uppercase;
  ">Where Every Journey Matters</p>
</div>
    """,
    unsafe_allow_html=True,
)

# ── Sample questions (default screen only) ───────────────────
_samples_ph = st.empty()
if not st.session_state.messages:
    with _samples_ph.container():
        st.markdown("**Try asking:**")
        col_a, col_b = st.columns(2)
        for i, q in enumerate(_SAMPLE_QUERIES):
            with (col_a if i % 2 == 0 else col_b):
                if st.button(q, use_container_width=True, key=f"sq_{i}"):
                    st.session_state.pending_query = q
                    st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)

# ── Chat history ──────────────────────────────────────────────
for msg in st.session_state.messages:
    avatar = MIRA_AVATAR if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        if msg["role"] == "assistant" and msg.get("tools"):
            tool_labels = " · ".join(
                t.replace("tool_", "").replace("_", " ").title()
                for t in msg["tools"]
            )
            st.caption(f"Tools consulted: {tool_labels}")
        st.markdown(_render(msg["content"]), unsafe_allow_html=True)

# ── MIRA avatar injected inside chat input bar ───────────────
if _AVATAR_PATH.exists():
    _av_b64 = base64.b64encode(_AVATAR_PATH.read_bytes()).decode()
    st.markdown(f"""
<style>
[data-testid="stChatInput"] {{
    padding-left: 58px !important;
    position: relative !important;
}}
[data-testid="stChatInput"]::before {{
    content: "";
    position: absolute;
    left: 8px;
    top: 50%;
    transform: translateY(-50%);
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background-image: url("data:image/png;base64,{_av_b64}");
    background-size: cover;
    background-position: center;
    z-index: 10;
    box-shadow: 0 1px 4px rgba(0,0,0,0.2);
}}
[data-testid="stChatInput"] > div {{
    border: 1.5px solid #22c55e !important;
    border-radius: 0.75rem !important;
    transition: border-color 0.3s ease;
}}
[data-testid="stBottom"] {{
    background-color: rgb(240, 242, 246) !important;
}}
[data-testid="stBottom"] > * {{
    background-color: rgb(240, 242, 246) !important;
}}
</style>
""", unsafe_allow_html=True)

# ── Resolve pending query from sidebar buttons ────────────────
query = None
if st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = None
else:
    query = st.chat_input("Ask about your booking, baggage, miles, delays...")

# ── Handle new query ──────────────────────────────────────────
if query:
    _samples_ph.empty()  # Dismiss sample questions immediately on submit
    # Red border = MIRA is processing
    st.markdown("""
<style>
[data-testid="stChatInput"] > div {
    border: 1.5px solid #ef4444 !important;
    border-radius: 0.75rem !important;
}
</style>
""", unsafe_allow_html=True)
    # Display user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Pre-select tools
    tools_selected = select_tools(query, top_k=2)
    tool_labels = " · ".join(
        t.replace("tool_", "").replace("_", " ").title()
        for t in tools_selected
    )

    # Stream assistant response with styled citations
    with st.chat_message("assistant", avatar=MIRA_AVATAR):
        st.caption(f"Tools consulting: {tool_labels}")
        response_area = st.empty()
        full_response = ""
        with st.spinner("Retrieving from Meridian Airlines documentation..."):
            for token in stream_query(query):
                full_response += token
                response_area.markdown(_render(full_response) + "▌", unsafe_allow_html=True)
        response_area.markdown(_render(full_response), unsafe_allow_html=True)
        response = full_response

    # Persist to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "tools": tools_selected,
    })
    st.session_state.query_count += 1
    st.rerun()
