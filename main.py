"""
main.py — CLI entry point.

Run from the conference_agent/ directory:
    python main.py
"""

import os
import Silence_warnings   # noqa — suppresses transformers/HF noise
from graph import build_graph, initial_state
from tools.search import load_csv_to_qdrant


# ── Pretty printer ────────────────────────────────────────────────────
def print_results(r: dict) -> None:
    SEP  = "=" * 58
    LINE = "─" * 40

    print(f"\n{SEP}")
    print("✅  CONFERENCE PLAN")
    print(SEP)
    print(f"  {r['event_category']}  |  {r['geography']}  |  {r['audience_size']} pax")

    # Sponsors
    print(f"\n💼  SPONSORS  ({len(r['sponsors'])} found)")
    print(LINE)
    for s in r["sponsors"]:
        print(f"  [{s.get('sponsorship_tier','?'):8s}]  {s.get('name','?')}"
              f"  (score {s.get('relevance_score','?')}/10)"
              f"  ~₹{s.get('estimated_contribution_inr',0):,}")
        print(f"              {s.get('reason','')}")

    # Speakers
    print(f"\n🎤  SPEAKERS  ({len(r['speakers'])} found)")
    print(LINE)
    for s in r["speakers"]:
        print(f"  [{s.get('speaker_type','?'):9s}]  {s.get('name','?')}")
        print(f"               {s.get('current_role','?')}")
        print(f"               \"{s.get('talk_title','?')}\"")

    # Venues
    print(f"\n🏛️   VENUES  ({len(r['venues'])} found)")
    print(LINE)
    for v in r["venues"]:
        print(f"  {v.get('name','?')}  ({v.get('area','?')})")
        print(f"    capacity: {v.get('capacity','?')}  |"
              f"  cost: ₹{v.get('est_daily_cost_inr',0):,}/day  |"
              f"  fit: {v.get('fit_score','?')}/10")
        print(f"    ✓ {v.get('pros','')}")
        print(f"    ✗ {v.get('cons','')}")

    # Pricing
    p = r.get("pricing", {})
    print("\n💰  PRICING")
    print(LINE)
    if p:
        print(f"  Early Bird : ₹{p.get('early_bird_inr',0):,}")
        print(f"  Standard   : ₹{p.get('standard_inr',0):,}")
        print(f"  VIP        : ₹{p.get('vip_inr',0):,}")
        print(f"  Virtual    : ₹{p.get('virtual_inr',0):,}")
        print(f"  Break-even : {p.get('break_even_attendees','?')} attendees")
        print(f"  Est. Profit: ₹{p.get('profit_inr',0):,}")
        print(f"  Rationale  : {p.get('pricing_rationale','')}")

    # GTM
    g = r.get("gtm_plan", {})
    print("\n📢  GO-TO-MARKET PLAN")
    print(LINE)
    if g:
        print(f"  Start : {g.get('timeline_weeks','?')} weeks before event")
        print(f"  Reach : {g.get('estimated_reach',0):,} people")
        print(f"  Message: {g.get('messaging','')}")
        print("\n  Communities:")
        for c in g.get("target_communities", []):
            print(f"    • {c}")
        print(f"\n  Channels: {', '.join(g.get('promotional_channels', []))}")
        print("\n  Content Calendar:")
        for item in g.get("content_calendar", []):
            print(f"    Week {item.get('week','?'):>3}: {item.get('action','')}")

    print(f"\n{SEP}\n")


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load historical events into Qdrant (skipped automatically if already loaded)
    csv_path = os.path.join("data", "events.csv")
    if os.path.exists(csv_path):
        load_csv_to_qdrant(csv_path)
    else:
        print("⚠️  data/events.csv not found — running without RAG context")

    app = build_graph()

    print("=" * 58)
    print("🎯  Conference Planning Multi-Agent System")
    print("=" * 58)

    category  = input("Event category  (e.g. AI/ML): ").strip()
    geography = input("Location        (e.g. Bangalore, India): ").strip()
    audience  = int(input("Expected audience size: ").strip())

    print("\n⏳ Running agents...\n")

    result = app.invoke(initial_state(category, geography, audience))
    print_results(result)