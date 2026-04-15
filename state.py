from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class SoloForgeState(TypedDict):

    # ── Brand Details ──
    brand_name: str
    brand_tone: str
    target_audience: str
    current_offer: str
    industry: str
    competitors: str
    tagline: str
    brand_context: str

    # ── User Request ──
    user_query: str
    query_valid: bool
    need_content: bool
    need_image: bool
    need_both: bool
    need_social: bool

    # ── Agent Outputs ──
    market_insights: str
    captions: str
    strategy: str
    image_url: str        # ← was missing!
    image_prompt: str     # ← was missing!

    # ── Quality Control ──
    critic_score: int
    critic_feedback: str
    revision_count: int
    revision_needed: bool

    # ── Control Flow ──
    next_worker: str
    human_approved: bool  # ← fixed typo (was human_approval)

    # ── Messages ──
    messages: Annotated[list, add_messages]