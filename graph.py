"""
graph.py  — Person 1 (you) owns this file.

Builds the LangGraph StateGraph that wires all 5 agents together.
Each agent writes its results into shared ConferenceState.
The pipeline runs sequentially so later agents can use earlier agents' output.

Pipeline order:
  sponsor_agent → speaker_agent → venue_agent → pricing_agent → gtm_agent
  (order matters: pricing uses venue costs; GTM uses speaker + sponsor names)
"""
# conference_agent/pathfix.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from langgraph.graph import StateGraph, END

from state import ConferenceState
from agents.sponsor import sponsor_agent
from agents.speaker import speaker_agent
from agents.venue import venue_agent
from agents.pricing import pricing_agent
from agents.gtm import gtm_agent


def build_graph():
    """Build and compile the conference planning graph."""
    builder = StateGraph(ConferenceState)

    # ── Register nodes ───────────────────────────────────────────────
    builder.add_node("sponsor_agent", sponsor_agent)
    builder.add_node("speaker_agent", speaker_agent)
    builder.add_node("venue_agent",   venue_agent)
    builder.add_node("pricing_agent", pricing_agent)
    builder.add_node("gtm_agent",     gtm_agent)

    # ── Define execution order ───────────────────────────────────────
    builder.set_entry_point("sponsor_agent")
    builder.add_edge("sponsor_agent", "speaker_agent")
    builder.add_edge("speaker_agent", "venue_agent")
    builder.add_edge("venue_agent",   "pricing_agent")
    builder.add_edge("pricing_agent", "gtm_agent")
    builder.add_edge("gtm_agent",     END)

    return builder.compile()


# ── Convenience: empty initial state ────────────────────────────────
def initial_state(event_category: str, geography: str, audience_size: int) -> dict:
    return {
        "event_category": event_category,
        "geography":      geography,
        "audience_size":  audience_size,
        "sponsors":  [],
        "speakers":  [],
        "venues":    [],
        "pricing":   {},
        "gtm_plan":  {},
    }
