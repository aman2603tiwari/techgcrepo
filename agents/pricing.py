"""
agents/pricing.py  — Person 3 owns the prompts inside this file.

Flow:
  1. Qdrant RAG  → what did similar past events charge?
  2. LLM         → build a pricing model anchored in real market data
  Uses venue cost estimates from state for break-even calculation.
  No Tavily needed here — historical data is more reliable than live search for pricing.
"""
# conference_agent/pathfix.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from state import ConferenceState
from tools.search import query_similar_events
from tools.llm import call_llm_json


def pricing_agent(state: ConferenceState) -> dict:
    print("\n💰 Pricing Agent running...")

    # ── 1. RAG: historical pricing data ─────────────────────────────
    similar = query_similar_events(state["event_category"], state["geography"])
    rag_block = ""
    if similar:
        lines = []
        for e in similar:
            lines.append(
                f"  • {e.get('event_name','?')} ({e.get('year','?')})"
                f" — early bird: ₹{e.get('ticket_price_early_bird','?')}"
                f" | standard: ₹{e.get('ticket_price_standard','?')}"
                f" | VIP: ₹{e.get('ticket_price_vip','?')}"
                f" | attendance: {e.get('attendance','?')}"
            )
        rag_block = "Ticket pricing from similar past events:\n" + "\n".join(lines)

    # ── 2. Derive average venue cost from venue agent output ─────────
    avg_venue_cost = 500_000   # default fallback
    if state["venues"]:
        costs = [
            v.get("est_daily_cost_inr", 0)
            for v in state["venues"]
            if isinstance(v, dict) and v.get("est_daily_cost_inr", 0) > 0
        ]
        if costs:
            avg_venue_cost = int(sum(costs) / len(costs))

    sponsor_revenue = sum(
        s.get("estimated_contribution_inr", 0)
        for s in state["sponsors"]
        if isinstance(s, dict)
    )

    result = call_llm_json(
        system_prompt="""You are a financial strategist specialising in tech event economics in India.

TASK: Build a realistic, profitable ticket pricing model for the conference.

STRICT OUTPUT RULES:
- Respond with ONLY a raw JSON object — no prose, no markdown, no explanation.
- Start with { and end with }
- The object must have EXACTLY these 11 fields:
    early_bird_inr          : integer  (price for first 30% of tickets)
    standard_inr            : integer  (regular ticket price)
    vip_inr                 : integer  (includes dinner + front row + speaker access)
    virtual_inr             : integer  (online attendance — typically 15-20% of standard)
    expected_in_person      : integer  (realistic paid attendance given price point)
    total_ticket_revenue_inr: integer  (expected_in_person × blended average ticket price)
    total_expenses_inr      : integer  (venue + catering + AV + production + speaker fees)
    sponsor_revenue_inr     : integer  (confirmed sponsorship revenue)
    profit_inr              : integer  (total_ticket_revenue + sponsor_revenue - total_expenses)
    break_even_attendees    : integer  (minimum ticket sales to cover expenses after sponsorship)
    pricing_rationale       : string   (2-3 sentences: why this pricing, what it benchmarks against)

Rule: profit_inr = total_ticket_revenue_inr + sponsor_revenue_inr - total_expenses_inr
Rule: break_even_attendees = max(0, (total_expenses_inr - sponsor_revenue_inr) / standard_inr)

Example of perfectly valid output:
{"early_bird_inr":1499,"standard_inr":2499,"vip_inr":7999,"virtual_inr":399,"expected_in_person":380,"total_ticket_revenue_inr":949620,"total_expenses_inr":1200000,"sponsor_revenue_inr":4500000,"profit_inr":4249620,"break_even_attendees":192,"pricing_rationale":"Priced between PyCon India (₹800-1200) and NASSCOM events (₹5000+). VIP tier delivers clear added value. Sponsorship revenue makes this event comfortably profitable even at 60% ticket fill rate."}""",

        user_message=f"""Conference details:
- Category          : {state['event_category']}
- Location          : {state['geography']}
- Target audience   : {state['audience_size']} people
- Avg venue cost    : ₹{avg_venue_cost:,} / day
- Confirmed sponsor revenue: ₹{sponsor_revenue:,}

{rag_block}

Indian conference pricing anchors (use these as market benchmarks):
  Budget / community events : ₹0 – ₹500
  Mid-tier tech conferences : ₹500 – ₹3,000
  Premium industry events   : ₹3,000 – ₹15,000
  Enterprise / NASSCOM tier : ₹10,000 – ₹50,000

This is a {state['event_category']} event — position it appropriately.
Account for the fact that sponsorship partially offsets expenses.

Output the JSON object now:""",
        fallback={},
    )

    print(f"  ✓ Pricing model built")
    return {"pricing": result if isinstance(result, dict) else {}}
