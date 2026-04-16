"""
agents/sponsor.py  — Person 3 owns the prompts inside this file.

Flow:
  1. Qdrant RAG  → who sponsored similar past events?
  2. Tavily      → live web search for current sponsors
  3. LLM         → synthesise into structured JSON
"""
# conference_agent/pathfix.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from state import ConferenceState
from tools.search import web_search, query_similar_events
from tools.llm import call_llm_json


def sponsor_agent(state: ConferenceState) -> dict:
    print("\n🔍 Sponsor Agent running...")

    # ── 1. RAG context from historical events ───────────────────────
    similar = query_similar_events(state["event_category"], state["geography"])
    rag_block = ""
    if similar:
        lines = [
            f"  • {e.get('event_name','?')} ({e.get('year','?')}, {e.get('city','?')})"
            f" — sponsors: {e.get('sponsors','?')}"
            f" | attendance: {e.get('attendance','?')}"
            for e in similar
        ]
        rag_block = "Verified past events from our database:\n" + "\n".join(lines)

    # ── 2. Live web search ───────────────────────────────────────────
    web_block = web_search(
        f"{state['event_category']} conference sponsors {state['geography']} 2024 2025",
        max_results=5,
    )

    # ── 3. LLM synthesis ────────────────────────────────────────────
    result = call_llm_json(
        system_prompt="""You are a senior sponsorship strategist who has worked on 100+ tech conferences across India and Southeast Asia.

TASK: Identify the most relevant corporate sponsors for the given conference.

STRICT OUTPUT RULES:
- Respond with ONLY a raw JSON array — no prose, no markdown fences, no explanation.
- Start your response with [ and end with ]
- Each object must have EXACTLY these 6 fields:
    name                      : string  (real company name)
    industry                  : string  (e.g. "Cloud Computing", "Fintech", "Dev Tools")
    relevance_score           : integer 1-10
    reason                    : string  (why THIS company would sponsor THIS specific event — be concrete)
    sponsorship_tier          : string  (one of: Title | Gold | Silver | Community)
    estimated_contribution_inr: integer (realistic INR amount based on tier and event size)

Tier guidelines (adjust to event scale):
  Title     → ₹10L–₹50L  (1 per event)
  Gold      → ₹3L–₹10L   (2-3 per event)
  Silver    → ₹1L–₹3L    (4-6 per event)
  Community → ₹10k–₹1L   (unlimited)

Example of perfectly valid output:
[{"name":"Google Cloud","industry":"Cloud Computing","relevance_score":9,"reason":"Google Cloud aggressively sponsors AI/ML events in India to grow Vertex AI and TPU adoption among practitioners","sponsorship_tier":"Title","estimated_contribution_inr":2000000}]""",

        user_message=f"""Conference details:
- Category  : {state['event_category']}
- Location  : {state['geography']}
- Audience  : {state['audience_size']} expected attendees

{rag_block}

Live web intelligence:
{web_block}

Using the above real data, identify 6 sponsors with the highest strategic fit.
Prioritise companies that:
  1. Have a known presence or offices in {state['geography']}
  2. Have sponsored {state['event_category']} events in India before
  3. Would gain genuine ROI from {state['audience_size']} {state['event_category']} professionals

Output the JSON array now:""",
        fallback=[],
    )

    sponsors = result if isinstance(result, list) else []
    print(f"  ✓ {len(sponsors)} sponsors identified")
    return {"sponsors": sponsors}
