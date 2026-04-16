"""
agents/speaker.py  — Person 3 owns the prompts inside this file.

Flow:
  1. Qdrant RAG  → who spoke at similar past events?
  2. Tavily      → live search for active speakers in this domain
  3. LLM         → synthesise into structured JSON
  Uses sponsor list from state as context (speakers should align with sponsor industries).
"""
# conference_agent/pathfix.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from state import ConferenceState
from tools.search import web_search, query_similar_events
from tools.llm import call_llm_json


def speaker_agent(state: ConferenceState) -> dict:
    print("\n🎤 Speaker Agent running...")

    # ── 1. RAG context ──────────────────────────────────────────────
    similar = query_similar_events(state["event_category"], state["geography"])
    rag_block = ""
    if similar:
        lines = [
            f"  • {e.get('event_name','?')} ({e.get('year','?')}) — speakers: {e.get('speakers','?')}"
            for e in similar
        ]
        rag_block = "Speakers who appeared at similar past events:\n" + "\n".join(lines)

    # ── 2. Tavily ───────────────────────────────────────────────────
    web_block = web_search(
        f"top {state['event_category']} speakers India conference 2024 2025",
        max_results=5,
    )

    # ── 3. Build sponsor context so speakers align with sponsors ────
    sponsor_industries = list({
        s.get("industry", "") for s in state["sponsors"] if isinstance(s, dict)
    })
    sponsor_names = [s.get("name", "") for s in state["sponsors"] if isinstance(s, dict)]

    result = call_llm_json(
        system_prompt="""You are a world-class conference programme director who has curated speaker lineups for Google I/O India, PyCon India, NASSCOM AI, and similar flagship events.

TASK: Curate a diverse, credible speaker lineup for the given conference.

STRICT OUTPUT RULES:
- Respond with ONLY a raw JSON array — no prose, no markdown, no explanation.
- Start with [ and end with ]
- Each object must have EXACTLY these 7 fields:
    name           : string  (real person's full name)
    current_role   : string  (current job title + company, e.g. "Principal Engineer, Google DeepMind India")
    expertise      : string  (narrow specialisation, e.g. "Diffusion models & multimodal AI")
    influence_score: integer 1-10 (based on publications, community standing, social reach)
    talk_title     : string  (a specific, compelling session title for this conference)
    why_them       : string  (one concrete reason they are a great fit for this audience)
    speaker_type   : string  (one of: Keynote | Workshop | Panel | Lightning)

Speaker mix guidance:
  - At least 1 international speaker (global credibility)
  - At least 2 India-based practitioners (relatable to local audience)
  - At least 1 workshop-format speaker (hands-on value)
  - Mix of industry + academia where relevant

Example of perfectly valid output:
[{"name":"Soumith Chintala","current_role":"Co-founder PyTorch / Research Scientist, Meta AI","expertise":"Deep learning frameworks and open-source AI infrastructure","influence_score":10,"talk_title":"The Next Five Years of Open-Source AI","why_them":"Creator of PyTorch — every ML practitioner in the audience uses his work daily","speaker_type":"Keynote"}]""",

        user_message=f"""Conference details:
- Category  : {state['event_category']}
- Location  : {state['geography']}
- Audience  : {state['audience_size']} attendees

Confirmed sponsors (align speaker topics where relevant):
  {', '.join(sponsor_names) if sponsor_names else 'TBD'}
Sponsor industries: {', '.join(sponsor_industries) if sponsor_industries else 'TBD'}

{rag_block}

Live web intelligence:
{web_block}

Suggest 6 speakers. Prioritise:
  1. Speakers who have previously spoken at events in {state['geography']} or India
  2. Practitioners with real-world case studies, not just academics
  3. Diversity of seniority (CTO / researcher / indie builder)

Output the JSON array now:""",
        fallback=[],
    )

    speakers = result if isinstance(result, list) else []
    print(f"  ✓ {len(speakers)} speakers identified")
    return {"speakers": speakers}
