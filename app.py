"""
app.py — Conference Planner UI with streaming chatbot
Run: streamlit run app.py
"""

import time
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import warnings

# kill all HF + transformers logs
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

warnings.filterwarnings("ignore")

# disable tokenizer + HF noise
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import random 

import json
import pandas as pd
import streamlit as st
from openai import OpenAI

st.set_page_config(
    page_title="Conference Planner",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 2rem 2.5rem; max-width: 1400px; }

  .nav-bar {
    display: flex; align-items: center; justify-content: space-between;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid rgba(128,128,128,0.15);
    margin-bottom: 1.8rem;
  }
  .nav-logo { font-size: 17px; font-weight: 600; letter-spacing: -0.3px; color: var(--text-color); }
  .nav-sub  { font-size: 12px; color: rgba(128,128,128,0.7); margin-top: 2px; }
  .nav-badge {
    font-size: 11px; padding: 3px 11px; border-radius: 20px;
    border: 1px solid rgba(128,128,128,0.25);
    color: rgba(128,128,128,0.8); font-weight: 500; background: transparent;
  }
  .section-label {
    font-size: 10px; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: rgba(128,128,128,0.6); margin-bottom: 10px;
  }
  div[data-baseweb="select"] > div {
    border-radius: 10px !important;
    border: 1px solid rgba(128,128,128,0.25) !important;
  }
  div[data-baseweb="select"] > div:focus-within {
    border-color: rgba(128,128,128,0.5) !important; box-shadow: none !important;
  }
  [data-testid="stSlider"] [role="slider"] {
    background: var(--text-color) !important;
    border: 2px solid var(--background-color) !important;
    width: 18px !important; height: 18px !important;
  }
  [data-testid="stSlider"] p {
    color: var(--text-color) !important; font-weight: 600 !important; font-size: 14px !important;
  }
  [data-testid="stSlider"] label p {
    color: rgba(128,128,128,0.7) !important; font-size: 13px !important;
  }
  .stFormSubmitButton > button {
    border-radius: 10px !important; font-size: 14px !important;
    font-weight: 500 !important; height: 44px !important;
    width: 100% !important; margin-top: 4px !important;
  }
  .stTabs [data-baseweb="tab-list"] {
    gap: 0; border-bottom: 1px solid rgba(128,128,128,0.15); background: transparent;
  }
  .stTabs [data-baseweb="tab"] {
    font-size: 13px !important; font-weight: 500 !important; padding: 8px 16px !important;
    border-bottom: 2px solid transparent !important; background: transparent !important;
    color: rgba(128,128,128,0.6) !important;
  }
  .stTabs [aria-selected="true"] {
    color: var(--text-color) !important; border-bottom-color: var(--text-color) !important;
  }
  [data-testid="metric-container"] {
    border: 1px solid rgba(128,128,128,0.15); border-radius: 12px; padding: 14px 18px;
  }
  [data-testid="metric-container"] label {
    font-size: 10px !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 0.07em !important;
    color: rgba(128,128,128,0.6) !important;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 22px !important; font-weight: 600 !important;
  }
  .chat-placeholder {
    border: 1px solid rgba(128,128,128,0.15); border-radius: 14px;
    padding: 2.5rem 1.5rem; text-align: center;
    color: rgba(128,128,128,0.5); font-size: 13px; line-height: 1.7;
    min-height: 140px; display: flex; align-items: center; justify-content: center;
  }
  .agent-pill {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 13px; padding: 4px 12px; border-radius: 20px; margin: 3px 0;
    font-weight: 500; background: rgba(45,106,79,0.12); color: #2d6a4f;
  }
  .agent-dot { width: 6px; height: 6px; border-radius: 50%; background: #2d6a4f; }
  .chip-btn > button {
    font-size: 11px !important; border-radius: 20px !important;
    padding: 4px 10px !important; height: auto !important;
    border-color: rgba(128,128,128,0.25) !important;
    white-space: nowrap !important; overflow: hidden !important;
    text-overflow: ellipsis !important; width: 100% !important;
  }
  [data-testid="stChatMessage"] { background: transparent !important; }
  [data-testid="stChatInput"] textarea {
    border-radius: 12px !important; border-color: rgba(128,128,128,0.25) !important;
  }
  hr { border: none; border-top: 1px solid rgba(128,128,128,0.12); margin: 1.4rem 0; }
  .stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Groq client ────────────────────────────────────────────────────────
@st.cache_resource
def get_groq():
    return OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
    )


@st.cache_resource(show_spinner=False)
def load_pipeline():
    from graph import build_graph
    from tools.search import load_csv_to_qdrant
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "events.csv")
    if os.path.exists(csv_path):
        load_csv_to_qdrant(csv_path)
    return build_graph()


# ── Session state ──────────────────────────────────────────────────────
for key, default in [
    ("chat_history", []),
    ("result", None),
    ("suggestions", [
        "What sponsors should I prioritise?",
        "How do I increase ticket revenue?",
        "Which venue is best for networking?",
        "Draft a speaker outreach email",
    ]),
    ("pending_prompt", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helpers ────────────────────────────────────────────────────────────
def build_system_prompt(r: dict) -> str:
    summary = (
        f"Conference plan:\n"
        f"- Category: {r['event_category']}, Location: {r['geography']}, Audience: {r['audience_size']}\n"
        f"- Sponsors: {', '.join(s.get('name','') for s in r['sponsors'][:4])}\n"
        f"- Speakers: {', '.join(s.get('name','') for s in r['speakers'][:4])}\n"
        f"- Venues: {', '.join(v.get('name','') for v in r['venues'][:3])}\n"
        f"- Pricing: ₹{r['pricing'].get('early_bird_inr','?')} early bird, "
        f"₹{r['pricing'].get('standard_inr','?')} standard, "
        f"₹{r['pricing'].get('vip_inr','?')} VIP\n"
        f"- Estimated profit: ₹{r['pricing'].get('profit_inr','?')}\n"
        f"- GTM reach: {r['gtm_plan'].get('estimated_reach','?')} people\n"
        f"- GTM message: {r['gtm_plan'].get('messaging','')}"
    )
    return (
        "You are a sharp, practical conference planning expert. "
        "You have the following plan in context:\n\n" + summary + "\n\n"
        "Answer questions about this plan concisely. Give specific, actionable advice. "
        "Use ₹ for rupee amounts. No fluff."
    )

def slow_stream(stream, delay=0.15):  # adjust delay here
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            for char in content:
                yield char
                time.sleep(delay)

def refresh_suggestions(chat_history: list) -> list:
    """Ask Groq to generate 4 contextual follow-up suggestions based on conversation so far."""
    groq = get_groq()
    # build a short conversation summary for the suggestion prompt
    recent = [m for m in chat_history if m["role"] != "system"][-6:]
    convo_text = "\n".join(f"{m['role'].upper()}: {m['content'][:200]}" for m in recent)

    try:
        resp = groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": (
    "Given this conference planning conversation:\n\n"
    + convo_text +
    "\n\nGenerate exactly 4 short follow-up questions the user might ask next. "
    "Each must be under 8 words. Vary topics. Return ONLY JSON array.\n\n"
    """Example: ["How do we reach more developers?", "What's a realistic VIP price?",
    "Which speaker should headline?", "Timeline for ticket sales?"]"""
)
            }],
            max_tokens=150,
            temperature=0.8,
        )
        raw = resp.choices[0].message.content.strip()
        # parse JSON
        if raw.startswith("["):
            suggestions = json.loads(raw)
            if isinstance(suggestions, list) and len(suggestions) >= 4:
                return suggestions[:4]
    except Exception:
        pass
    # fallback: keep existing suggestions
    return st.session_state.suggestions


# ── Nav ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nav-bar">
  <div>
    <div class="nav-logo">Conference Planner</div>
    <div class="nav-sub">Multi-agent AI · Groq + Tavily + Qdrant RAG</div>
  </div>
  <div class="nav-badge">5 agents</div>
</div>
""", unsafe_allow_html=True)


# ── Layout ─────────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

# ══════════════════════════════════════════════
# LEFT — Planner
# ══════════════════════════════════════════════
with left:
    st.markdown('<div class="section-label">Configure your event</div>', unsafe_allow_html=True)

    with st.form("conf_form"):
        c1, c2 = st.columns(2)
        with c1:
            category = st.selectbox("Category", [
                "AI/ML", "Web3 / Blockchain", "ClimateTech", "Fintech",
                "DevOps / Cloud", "Cybersecurity", "Edtech", "Healthtech",
            ])
        with c2:
            geography = st.selectbox("Location", [
                "Bangalore, India", "Mumbai, India", "Delhi NCR, India",
                "Hyderabad, India", "Pune, India", "Chennai, India",
                "Singapore", "London, UK",
            ])
        audience  = st.slider("Expected attendance", 100, 5000, 500, 100, format="%d people")
        submitted = st.form_submit_button("Generate plan →", width='stretch')

    if submitted:
        from graph import initial_state
        LABELS = {
            "sponsor_agent": "Sponsors identified",
            "speaker_agent": "Speakers curated",
            "venue_agent":   "Venues scouted",
            "pricing_agent": "Revenue model built",
            "gtm_agent":     "GTM plan created",
        }
        accumulated = initial_state(category, geography, audience)
        with st.status("Running agents…", expanded=True) as status:
            app_graph = load_pipeline()
            for step in app_graph.stream(accumulated):
                for node, out in step.items():
                    st.markdown(
                        f'<div class="agent-pill"><div class="agent-dot"></div>'
                        f'{LABELS.get(node, node)}</div>',
                        unsafe_allow_html=True,
                    )
                    accumulated.update(out)
            status.update(label="Plan ready", state="complete", expanded=False)

        st.session_state.result = accumulated
        # seed chat
        sys_msg = build_system_prompt(accumulated)
        st.session_state.chat_history = [{"role": "system", "content": sys_msg}]
        st.session_state.suggestions = [
            "Which sponsor should I approach first?",
            "How do I price VIP tickets better?",
            "Best venue for a 500-person crowd?",
            "Draft a speaker outreach email",
        ]

    if st.session_state.result:
        result = st.session_state.result
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Plan details</div>', unsafe_allow_html=True)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Sponsors", "Speakers", "Venues", "Pricing", "GTM"]
        )

        with tab1:
            sponsors = result.get("sponsors", [])
            if sponsors:
                st.dataframe(pd.DataFrame(sponsors), column_config={
                    "relevance_score": st.column_config.ProgressColumn("Relevance", min_value=0, max_value=10),
                    "estimated_contribution_inr": st.column_config.NumberColumn("Contribution (₹)", format="₹%d"),
                }, width='stretch', hide_index=True)
                total = sum(s.get("estimated_contribution_inr", 0) for s in sponsors)
                st.caption(f"Total estimated sponsorship: ₹{total:,}")

        with tab2:
            speakers = result.get("speakers", [])
            if speakers:
                st.dataframe(pd.DataFrame(speakers), column_config={
                    "influence_score": st.column_config.ProgressColumn("Influence", min_value=0, max_value=10),
                }, width='stretch', hide_index=True)

        with tab3:
            venues = result.get("venues", [])
            if venues:
                st.dataframe(pd.DataFrame(venues), column_config={
                    "est_daily_cost_inr": st.column_config.NumberColumn("Daily Cost (₹)", format="₹%d"),
                    "fit_score": st.column_config.ProgressColumn("Fit", min_value=0, max_value=10),
                }, width='stretch', hide_index=True)

        with tab4:
            p = result.get("pricing", {})
            if p:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Early Bird", f"₹{p.get('early_bird_inr',0):,}")
                m2.metric("Standard",   f"₹{p.get('standard_inr',0):,}")
                m3.metric("VIP",        f"₹{p.get('vip_inr',0):,}")
                m4.metric("Virtual",    f"₹{p.get('virtual_inr',0):,}")
                st.markdown("<hr>", unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("Break-even", f"{p.get('break_even_attendees','?')} pax")
                m2.metric("Revenue",    f"₹{p.get('total_ticket_revenue_inr',0):,}")
                m3.metric("Profit",     f"₹{p.get('profit_inr',0):,}")
                st.markdown("<hr>", unsafe_allow_html=True)
                st.caption(p.get("pricing_rationale", ""))

        with tab5:
            g = result.get("gtm_plan", {})
            if g:
                m1, m2 = st.columns(2)
                m1.metric("Promo window", f"{g.get('timeline_weeks','?')} weeks")
                m2.metric("Reach",        f"{g.get('estimated_reach',0):,}")
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown(f"**Message** — {g.get('messaging','')}")
                st.markdown("<hr>", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("Communities")
                    for c in g.get("target_communities", []):
                        st.markdown(f"· {c}")
                with c2:
                    cal = g.get("content_calendar", [])
                    if cal:
                        st.caption("Content calendar")
                        st.dataframe(
                            pd.DataFrame(cal).sort_values("week"),
                            width='stretch', hide_index=True,
                        )


# ══════════════════════════════════════════════
# RIGHT — Chatbot
# ══════════════════════════════════════════════
with right:
    st.markdown('<div class="section-label">Ask the planner</div>', unsafe_allow_html=True)

    if not st.session_state.result:
        st.markdown(
            '<div class="chat-placeholder">Generate a plan first,<br>'
            'then ask me anything about it.</div>',
            unsafe_allow_html=True,
        )
    else:
        # ── Chat history display ──────────────────────────────────────
        chat_container = st.container(height=380)
        with chat_container:
            display_msgs = [m for m in st.session_state.chat_history if m["role"] != "system"]
            if not display_msgs:
                st.markdown(
                    '<div style="padding:2rem;text-align:center;'
                    'color:rgba(128,128,128,0.4);font-size:13px;margin-top:1rem;">'
                    'Plan ready. Ask anything below.</div>',
                    unsafe_allow_html=True,
                )
            for msg in display_msgs:
                with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🎯"):
                    st.markdown(msg["content"])

        # ── Context-aware suggestion chips ───────────────────────────
        suggestions = st.session_state.suggestions
        chip_cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            col = chip_cols[i % 2]
            with col:
                # truncate for display but store full text
                display = suggestion if len(suggestion) <= 32 else suggestion[:30] + "…"
                btn_key = f"chip_{i}_{suggestion[:20]}"
                with st.container():
                    st.markdown('<div class="chip-btn">', unsafe_allow_html=True)
                    if st.button(display, key=btn_key, width='stretch', help=suggestion):
                        st.session_state.pending_prompt = suggestion
                    st.markdown('</div>', unsafe_allow_html=True)

        # ── Chat input ────────────────────────────────────────────────
        pending   = st.session_state.pop("pending_prompt", None)
        user_input = st.chat_input("Ask about your conference plan…") or pending

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            # ── Streaming response ────────────────────────────────────
            groq = get_groq()

            # show user message immediately then stream assistant
            with chat_container:
                # show user message
                with st.chat_message("user", avatar="🧑"):
                    st.markdown(user_input)

                # stream assistant INSIDE bubble
                with st.chat_message("assistant", avatar="🎯"):
                    stream = groq.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=st.session_state.chat_history,
                        max_tokens=600,
                        temperature=0.7,
                        stream=True,
                    )

                    placeholder = st.empty()
                    full_reply = ""

                    for chunk in stream:
                        content = chunk.choices[0].delta.content
                        if content:
                            for char in content:
                                full_reply += char
                                placeholder.markdown(full_reply + "▌")
                                delay = random.uniform(0.005, 0.015)
                                time.sleep(delay)

                    placeholder.markdown(full_reply)

            st.session_state.chat_history.append({"role": "assistant", "content": full_reply})

            # ── Refresh suggestions based on new context ──────────────
            st.session_state.suggestions = refresh_suggestions(st.session_state.chat_history)

            st.rerun()