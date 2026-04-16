from typing import TypedDict


class ConferenceState(TypedDict):
    event_category: str
    geography: str
    audience_size: int
    sponsors: list
    speakers: list
    venues: list
    pricing: dict
    gtm_plan: dict
