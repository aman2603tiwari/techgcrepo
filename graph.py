"""
graph.py — Conference Planning Agent Graph

Builds the LangGraph workflow connecting all conference-planning agents in PARALLEL execution.
All agents run concurrently: sponsor_agent, speaker_agent, venue_agent, pricing_agent, gtm_agent
"""

from langgraph.graph import StateGraph, START, END

from state import ConferenceState
from agents.sponsor import sponsor_agent
from agents.speaker import speaker_agent
from agents.venue import venue_agent
from agents.pricing import pricing_agent
from agents.gtm import gtm_agent


def build_graph():
    """
    Build and compile the LangGraph workflow with PARALLEL agent execution.
    All agents start simultaneously from the same input state.
    """

    builder = StateGraph(ConferenceState)

    # ─────────────────────────────────────────────
    # Register nodes
    # ─────────────────────────────────────────────
    builder.add_node("sponsor_agent", sponsor_agent)
    builder.add_node("speaker_agent", speaker_agent)
    builder.add_node("venue_agent", venue_agent)
    builder.add_node("pricing_agent", pricing_agent)
    builder.add_node("gtm_agent", gtm_agent)

    # ─────────────────────────────────────────────
    # Define PARALLEL execution flow
    # ─────────────────────────────────────────────
    # All agents start from START and run in parallel
    builder.add_edge(START, "sponsor_agent")
    builder.add_edge(START, "speaker_agent")
    builder.add_edge(START, "venue_agent")
    builder.add_edge(START, "pricing_agent")
    builder.add_edge(START, "gtm_agent")

    # All agents converge to END
    builder.add_edge("sponsor_agent", END)
    builder.add_edge("speaker_agent", END)
    builder.add_edge("venue_agent", END)
    builder.add_edge("pricing_agent", END)
    builder.add_edge("gtm_agent", END)

    # Compile graph
    return builder.compile()


def initial_state(
    event_category: str,
    geography: str,
    audience_size: int
) -> dict:
    """
    Create initial shared state for the conference pipeline.
    """

    return {
        "event_category": event_category,
        "geography": geography,
        "audience_size": audience_size,

        # Agent outputs
        "sponsors": [],
        "speakers": [],
        "venues": [],
        "pricing": {},
        "gtm_plan": {},
    }
