"""
agents/venue.py  — Person 3 owns the prompts inside this file.

Flow:
  1. Qdrant RAG  → where did similar past events take place?
  2. Tavily      → live search for current venue availability / capacity
  3. LLM         → synthesise into structured JSON
"""
# conference_agent/pathfix.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from state import ConferenceState
from tools.search import web_search, query_similar_events
from tools.llm import call_llm_json


def venue_agent(state: ConferenceState) -> dict:
    print("\n🏛️  Venue Agent running...")

    # ── 1. RAG context ──────────────────────────────────────────────
    similar = query_similar_events(state["event_category"], state["geography"])
    rag_block = ""
    if similar:
        lines = [
            f"  • {e.get('event_name','?')} ({e.get('year','?')}) held at: "
            f"{e.get('venue_name','?')} — {e.get('attendance','?')} attendees"
            for e in similar
        ]
        rag_block = "Venues used by similar past events:\n" + "\n".join(lines)

    # ── 2. Tavily ───────────────────────────────────────────────────
    city = state["geography"].split(",")[0].strip()
    web_block = web_search(
        f"best tech conference venues {city} India capacity {state['audience_size']} people 2024",
        max_results=5,
    )

    result = call_llm_json(
        system_prompt="""You are a professional event venue consultant specialising in tech conferences across India.

TASK: Recommend the best venues for the given conference, balancing capacity, cost, and tech-friendliness.

STRICT OUTPUT RULES:
- Respond with ONLY a raw JSON array — no prose, no markdown, no explanation.
- Start with [ and end with ]
- Each object must have EXACTLY these 9 fields:
    name               : string  (full venue name)
    area               : string  (neighbourhood / business district, e.g. "Whitefield, Bangalore")
    capacity           : integer (max standing or theatre-style capacity)
    est_daily_cost_inr : integer (realistic daily hire cost in INR — do not guess wildly)
    past_tech_events   : string  (1-2 real tech events held there, or "N/A")
    fit_score          : integer 1-10 (how well it matches this specific conference)
    pros               : string  (best feature for a tech conference)
    cons               : string  (one genuine drawback)
    venue_type         : string  (one of: Convention Centre | Hotel Ballroom | Co-working Space | Campus Auditorium | Dedicated Event Space)

Capacity + cost benchmarks for Indian cities:
  Small  (≤300 pax)  → ₹50k–₹2L/day
  Medium (300-1000)  → ₹2L–₹8L/day
  Large  (1000+)     → ₹8L–₹30L/day

Example of perfectly valid output:
[{"name":"NIMHANS Convention Centre","area":"Dairy Circle, Bangalore","capacity":2500,"est_daily_cost_inr":1000000,"past_tech_events":"NASSCOM Product Conclave 2023, Google Cloud Next India","fit_score":9,"pros":"Largest purpose-built convention centre in Bangalore with professional AV infrastructure and ample parking","cons":"Distance from CBD means some attendees need 45+ min commute","venue_type":"Convention Centre"}]""",

        user_message=f"""Conference requirements:
- Category        : {state['event_category']}
- City            : {state['geography']}
- Expected pax    : {state['audience_size']} people
- Duration        : typically 1-2 days

{rag_block}

Live web intelligence:
{web_block}

Recommend 4 venues — include a mix:
  1. One premium / flagship option (best experience, higher cost)
  2. Two mid-range options (good value for money)
  3. One budget-friendly / community option

Prioritise venues with:
  - Strong Wi-Fi and AV infrastructure (critical for tech events)
  - Good public transport access
  - Hotels or accommodation nearby for outstation attendees

Output the JSON array now:""",
        fallback=[],
    )

    venues = result if isinstance(result, list) else []
    print(f"  ✓ {len(venues)} venues identified")
    return {"venues": venues}
