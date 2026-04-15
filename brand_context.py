from llm import llm
from langchain.messages import HumanMessage,SystemMessage
from state import SoloForgeState
def base_context_maker(state:SoloForgeState)->SoloForgeState:
    """
    This function will make a base context that will act as the base context for other worker nodes
    """
    base_context = llm.invoke([
        SystemMessage(content=""" You are a analyst you job is the contruct a base context for a brand that will help them 
                      create new post , images and caption using that context. The context should be strict and summary based small in nature
                      saving the token of the user when use in LLM"""),
        HumanMessage(content=f"""brand_name": "LumaSkin",
    "band_tagline": "Glow Beyond the Surface",
    "brand_description": "A premium skincare brand focused on science-backed, clean beauty solutions for radiant and healthy skin.",
    "product_type": "Skincare",
    "product_description": "Serums, moisturizers, and cleansers made with natural ingredients and dermatologically tested formulas.",
    "price_range": "High-end",
    "target_aundience": "Urban professionals and skincare enthusiasts",
    "age_group": "25-45",
    "location": "Global",
    "tone": "Elegant and trustworthy",
    "style": "Minimalistic and modern",
    "value": "Transparency, quality, sustainability",
    "competitors": ["The Ordinary", "Drunk Elephant", "Kiehl's"],
    "primary_goal": "Build brand trust and increase online sales",
    "content_preference": ["Educational content", "Before-after results", "User testimonials"],
    "platform": ["Instagram", "YouTube", "Website"]""")])
    return {
        "brand_context":base_context.content[0]["text"]
    }
