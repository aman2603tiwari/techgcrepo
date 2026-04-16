"""
agents/gtm.py  — Person 3 owns the prompts inside this file.

GTM = Go-To-Market: how to PROMOTE and MARKET the conference to attract attendees.
(NOT Google Tag Manager!)

Flow:
  1. Qdrant RAG  → which communities promoted similar past events?
  2. Tavily      → live search for active communities in this domain
  3. LLM         → build full distribution plan using sponsor/speaker/pricing context
"""
# conference_agent/pathfix.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from state import ConferenceState
from tools.search import web_search, query_similar_events
from tools.llm import call_llm_json


def gtm_agent(state: ConferenceState) -> dict:
    print("\n📢 GTM Agent running...")

    # ── 1. RAG context ──────────────────────────────────────────────
    similar = query_similar_events(state["event_category"], state["geography"])
    rag_block = ""
    if similar:
        lines = [
            f"  • {e.get('event_name','?')} ({e.get('year','?')}) promoted via:"
            f" {e.get('communities_promoted','?')}"
            for e in similar
        ]
        rag_block = "Promotion channels used by similar past events:\n" + "\n".join(lines)

    # ── 2. Tavily ───────────────────────────────────────────────────
    web_block = web_search(
        f"{state['event_category']} community India Discord LinkedIn Slack active 2024",
        max_results=5,
    )

    # ── 3. Build context from previous agents ───────────────────────
    speaker_names = [s.get("name", "") for s in state["speakers"] if isinstance(s, dict)][:3]
    sponsor_names = [s.get("name", "") for s in state["sponsors"] if isinstance(s, dict)][:3]
    early_bird    = state["pricing"].get("early_bird_inr", "TBD")
    standard      = state["pricing"].get("standard_inr", "TBD")

    result = call_llm_json(
        system_prompt="""You are a growth marketing specialist who has promoted 50+ tech conferences in India and Southeast Asia.

TASK: Create a concrete go-to-market promotion plan to drive ticket sales and community buzz.

STRICT OUTPUT RULES:
- Respond with ONLY a raw JSON object — no prose, no markdown, no explanation.
- Start with { and end with }
- The object must have EXACTLY these 7 fields:
    target_communities  : list of strings  (real Discord servers, LinkedIn groups, Slack workspaces with rough member counts)
    promotional_channels: list of strings  (platforms: LinkedIn, Twitter/X, Devfolio, Newsletter, etc.)
    messaging           : string           (the punchy core value proposition — why attend THIS event?)
    content_calendar    : list of objects  (each: {"week": int, "action": string} — week 0 = event day)
    influencer_strategy : string           (who to partner with, what ask looks like)
    estimated_reach     : integer          (total unique people reachable across all channels)
    timeline_weeks      : integer          (how many weeks before the event to start promotion)

Guidelines:
  - content_calendar should have 6-8 entries from week -10 to week -1
  - target_communities should include both India-specific and global communities relevant to the topic
  - messaging must mention speakers and sponsors by name for credibility
  - influencer_strategy should be specific (YouTubers, newsletter writers, community moderators)

Example of perfectly valid output:
{"target_communities":["Discord: AI Builders India — 8.2k members","LinkedIn Group: ML India — 145k members","Slack: Kaggle India — 12k members"],"promotional_channels":["LinkedIn","Twitter/X","Devfolio","HasGeek","YourStory"],"messaging":"India first hands-on AI/ML summit featuring speakers from Google DeepMind, Hugging Face and Meta AI — 3 tracks, 12 workshops, ₹1,499 early bird","content_calendar":[{"week":-10,"action":"Announce event, open CFP, publish speaker interest form"},{"week":-8,"action":"Early bird launch + reveal 3 headline speakers"},{"week":-6,"action":"Speaker lineup drop + workshop schedule announcement"},{"week":-4,"action":"Sponsor spotlight posts (1 per week) + community AMA"},{"week":-2,"action":"Final ticket push — last 50 early bird tickets warning"},{"week":-1,"action":"Logistics email to registrants + social hype countdown"}],"influencer_strategy":"Partner with 5 Indian AI YouTubers (50k+ subscribers) for ticket giveaways in exchange for event recap videos; engage 3 popular newsletter authors (Tech in Asia India, The Ken) for sponsored mentions","estimated_reach":320000,"timeline_weeks":10}""",

        user_message=f"""Conference details:
- Category   : {state['event_category']}
- Location   : {state['geography']}
- Audience   : {state['audience_size']} people
- Pricing    : ₹{early_bird} early bird | ₹{standard} standard
- Speakers   : {', '.join(speaker_names) if speaker_names else 'TBD'}
- Sponsors   : {', '.join(sponsor_names) if sponsor_names else 'TBD'}

{rag_block}

Live web intelligence on active communities:
{web_block}

Create a comprehensive 10-week promotion plan. Focus on:
  1. Indian tech communities where {state['event_category']} practitioners are active
  2. Leveraging sponsor brands and speaker names in messaging
  3. Specific, actionable content calendar — not generic advice

Output the JSON object now:""",
        fallback={},
    )

    print(f"  ✓ GTM plan created")
    return {"gtm_plan": result if isinstance(result, dict) else {}}
