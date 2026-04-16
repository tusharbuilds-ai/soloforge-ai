from state import SoloForgeState
from langgraph.graph import StateGraph, START, END
from brand_context import base_context_maker
from langchain_core.messages import HumanMessage, SystemMessage
from llm import llm
import re
import os
from dotenv import load_dotenv

load_dotenv()

# ════════════════════════════════
# HELPERS
# ════════════════════════════════

def get_text(response) -> str:
    """Safely extract text from LLM response regardless of format"""
    try:
        content = response.content
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list) and len(content)>0:
                first = content[0]
                if isinstance(first,dict):
                    return first.get("text","").strip()
                return str(first).strip()
        return str(content).strip()
    except:
        return ""

def safe(value, length: int = 200) -> str:
    """Safely convert any value to string and slice"""
    if value is None:
        return ""
    return str(value)[:length]

# ════════════════════════════════
# GUARDRAIL NODE
# ════════════════════════════════

def guardrails_node(state: SoloForgeState) -> dict:
    user_query = state.get("user_query", "")
    print(f"\n🛡️ Guardrail checking: {safe(user_query, 50)}...")

    response = llm.invoke([
        SystemMessage(content="""You are a content safety checker.
Check if this user query is appropriate for a marketing platform.
It should NOT include:
- Harm to any community, culture, caste or religion
- Abuse or degradation of any gender
- Explicit or adult content
- Hate speech

Return ONLY one word: Valid or Invalid"""),
        HumanMessage(content=user_query)
    ])

    text = get_text(response).strip()
    print(f"   Guardrail result: {text}")

    # Case insensitive check
    is_valid = "valid" in text.lower() and "invalid" not in text.lower()

    return {"query_valid": is_valid}


def guardrail_route(state: SoloForgeState):
    if not state.get("query_valid", False):
        print("   ❌ Query invalid → END")
        return END
    print("   ✅ Query valid → supervisor")
    return "supervisor"

# ════════════════════════════════
# SUPERVISOR NODE
# ════════════════════════════════

def supervisor_node(state: SoloForgeState) -> dict:
    print("\n👔 Supervisor building context and detecting intent...")

    # Build brand context once
    brand_context = base_context_maker(state)
    print(f"   Brand context: {safe(brand_context, 80)}...")

    response = llm.invoke([
        SystemMessage(content="""You are an intent classifier for a marketing AI.
Analyze the user query and return ONLY one of these exact strings:

need_content    - user wants captions or text posts only
need_image      - user wants an image only
need_both       - user wants both captions AND image (full post)
need_social     - user wants posting strategy or social media advice

Rules:
- Return ONLY the intent string, nothing else
- No punctuation, no explanation
- If unsure between content and both, choose need_both"""),
        HumanMessage(content=state.get("user_query", ""))
    ])

    intent = get_text(response).strip().lower()
    print(f"   Intent detected: {intent}")

    result = {
        "brand_context": brand_context,
        "need_content": False,
        "need_image": False,
        "need_both": False,
    }

    if "need_both" in intent:
        result["need_both"] = True
        result["need_content"] = True
        result["need_image"] = True
    elif "need_image" in intent:
        result["need_image"] = True
    elif "need_social" in intent:
        result["need_content"] = True
    else:
        # Default: need_content (covers need_content + unknown)
        result["need_content"] = True

    print(f"   Content: {result['need_content']} | Image: {result['need_image']} | Both: {result['need_both']}")
    return result

# ════════════════════════════════
# MARKETING NODE
# ════════════════════════════════

def marketing_node(state: SoloForgeState) -> dict:
    print("\n📊 Market research agent working...")

    response = llm.invoke([
        SystemMessage(content=f"""You are a senior marketing strategist.
Your job is to research and provide market insights to guide content creation.

BRAND CONTEXT:
{safe(state.get('brand_context'), 400)}

USER REQUEST:
{safe(state.get('user_query'), 200)}

Provide concise market insights including:
1. Current trends relevant to this brand
2. Competitor content strategies
3. Target audience pain points
4. Best content angles for this request

Keep response under 300 words. Plain text only."""),
HumanMessage(content=F"Rsearch the market for this request")
    ])

    insights = get_text(response)
    print(f"   Market insights ready: {safe(insights, 80)}...")

    return {"market_insights": insights}


def marketing_route(state: SoloForgeState):
    need_content = state.get("need_content", False)
    need_image = state.get("need_image", False)

    if need_content:
        print("   → Going to content generator")
        return "content_generator_node"
    elif need_image:
        print("   → Going to image generator")
        return "image_generation_node"
    else:
        # Safety default
        print("   → Default: going to content generator")
        return "content_generator_node"

# ════════════════════════════════
# CONTENT GENERATOR NODE
# ════════════════════════════════

def content_generator_node(state: SoloForgeState) -> dict:
    print("\n✍️ Content creator agent generating captions...")

    response = llm.invoke([
        SystemMessage(content=f"""You are an expert social media copywriter.

BRAND CONTEXT:
{safe(state.get('brand_context'), 400)}

MARKET INSIGHTS:
{safe(state.get('market_insights'), 300)}

TASK: Generate exactly 5 social media captions for the user's request.

RULES:
- Each caption must match the brand tone exactly
- Include the current offer naturally in at least 2 captions
- Keep each caption under 200 characters
- Add 3-5 relevant hashtags at the end of each caption
- Number them 1 to 5
- Plain text only, no markdown, no asterisks
- Make each caption different in style and angle"""),
        HumanMessage(content=safe(state.get("user_query"), 200))
    ])

    captions = get_text(response)
    print(f"   Captions ready: {safe(captions, 80)}...")

    return {"captions": captions}

# ════════════════════════════════
# CONTENT CRITIC NODE
# ════════════════════════════════
def content_critic_node(state: SoloForgeState) -> dict:
    print("\n🔍 Critic agent reviewing...")

    brand_context = safe(state.get("brand_context"), 400)
    market_insights = safe(state.get("market_insights"), 400)
    captions = safe(state.get("captions"), 500)

    response = llm.invoke([
        SystemMessage(content=f"""You are a strict brand quality critic.

        BRAND PROFILE: {brand_context}

        SCORING PARAMETERS (each out of 10):
        1. BRAND ALIGNMENT - does tone match brand voice?
        2. AUDIENCE RELEVANCE - right for target audience?
        3. OFFER INTEGRATION - offer included naturally?
        4. MARKET INSIGHT QUALITY - trends relevant to brand?
        5. ACTIONABILITY - can founder post today as-is?

        RULES:
        - Score each 1-10
        - Calculate OVERALL SCORE (average)
        - APPROVE only if overall >= 5

        OUTPUT FORMAT:
        BRAND ALIGNMENT: X/10
        AUDIENCE RELEVANCE: X/10
        OFFER INTEGRATION: X/10
        MARKET INSIGHT QUALITY: X/10
        ACTIONABILITY: X/10
        OVERALL SCORE: X/10
        STRENGTHS: one line
        IMPROVEMENTS: one line
        VERDICT: APPROVE or REVISE"""),
        HumanMessage(content="Review this content: MARKET INSIGHTS: "
                    + market_insights
                    + " CAPTIONS: "
                    + captions)
        # ↑ string concatenation instead of f-string
        # avoids any dict/type issues
    ])

    content = get_text(response)
    print(f"   Critic output: {safe(content, 100)}...")

    verdict_match = re.search(r'VERDICT:\s*(APPROVE|REVISE)', content.upper())
    verdict = verdict_match.group(1) if verdict_match else "REVISE"
    needs_revision = verdict == "REVISE"

    print(f"   Verdict: {verdict}")

    return {
        "critic_feedback": content,
        "revision_needed": needs_revision
    }

# ════════════════════════════════
# REVISION NODE
# ════════════════════════════════

def revision_node(state: SoloForgeState) -> dict:
    revision_count = state.get("revision_count", 0) + 1
    print(f"\n🔄 Revision node — attempt {revision_count}/3")

    # Inject critic feedback into brand context for improvement
    current_context = state.get("brand_context", "")
    feedback = state.get("critic_feedback", "")
    
    updated_context = current_context
    if feedback:
        updated_context = current_context + f"\n\nPREVIOUS ATTEMPT FEEDBACK:\n{safe(feedback, 300)}\nPlease address these issues."

    return {
        "revision_needed": False,
        "market_insights": "",
        "captions": "",
        "brand_context": updated_context,
        "revision_count": revision_count,
    }

# ════════════════════════════════
# CRITIC ROUTE
# ════════════════════════════════

def critic_route(state: SoloForgeState):
    needs_revision = state.get("revision_needed", False)
    revision_count = state.get("revision_count", 0)
    need_image = state.get("need_image", False)

    if needs_revision and revision_count < 3:
        print(f"   🔄 Needs revision — attempt {revision_count + 1}/3")
        return "revision_node_needed"

    if need_image:
        print("   ✅ Content approved → generating image next")
        return "need_image_also"

    print("   ✅ Content approved → done")
    return "end"

# ════════════════════════════════
# IMAGE GENERATION NODE
# ════════════════════════════════

def image_generation_node(state: SoloForgeState) -> dict:
    print("\n🖼️ Image agent generating visual...")

    api_key = os.getenv("POLLINATIONS_API_KEY")
    market_insight = state.get("market_insights", "")
    captions = state.get("captions", "")
    brand_context = state.get("brand_context", "")

    # Generate a good image prompt
    response = llm.invoke([
        SystemMessage(content="""You are an expert at writing AI image generation prompts.
Create a SHORT, vivid image generation prompt for a social media product post.

Rules:
- Maximum 15 words
- Describe visual elements: objects, colors, lighting, mood, style
- NO text or words in the image
- Professional product photography style
- Return ONLY the prompt, nothing else"""),
        HumanMessage(content=f"""Brand context: {safe(brand_context, 150)}
Market insights: {safe(market_insight, 150)}
Caption context: {safe(captions, 150)}""")
    ])

    image_prompt = get_text(response).strip()
    # Remove any quotes if LLM added them
    image_prompt = image_prompt.strip('"\'')
    
    clean_prompt = image_prompt.replace(" ", "%20").replace(",", "").replace(".", "")[:200]

    if api_key:
        base_url = f"https://gen.pollinations.ai/image/{clean_prompt}?key={api_key}&model=flux"
    else:
        base_url = f"https://image.pollinations.ai/prompt/{clean_prompt}"

    print(f"   Image prompt: {image_prompt}")
    print(f"   Image URL ready ✅")

    return {
        "image_url": base_url,
        "image_prompt": image_prompt
    }

# ════════════════════════════════
# BUILD GRAPH
# ════════════════════════════════

def build_graph():
    graph = StateGraph(SoloForgeState)

    # ── Add Nodes ──
    graph.add_node("gaurdrails_node", guardrails_node)
    graph.add_node("supervisor_node", supervisor_node)
    graph.add_node("marketing_node", marketing_node)
    graph.add_node("content_generator_node", content_generator_node)
    graph.add_node("content_critic_node", content_critic_node)
    graph.add_node("revision_node", revision_node)
    graph.add_node("image_generation_node", image_generation_node)

    # ── Add Edges ──

    # Entry point
    graph.add_edge(START, "gaurdrails_node")

    # Guardrail → supervisor or END
    graph.add_conditional_edges(
        "gaurdrails_node",
        guardrail_route,
        {
            "supervisor": "supervisor_node",
            END: END
        }
    )

    # Supervisor always goes to market
    graph.add_edge("supervisor_node", "marketing_node")

    # Market → content or image based on intent
    graph.add_conditional_edges(
        "marketing_node",
        marketing_route,
        {
            "content_generator_node": "content_generator_node",
            "image_generation_node": "image_generation_node"
        }
    )

    # Content always goes to critic
    graph.add_edge("content_generator_node", "content_critic_node")

    # Critic → revision loop OR image OR end
    graph.add_conditional_edges(
        "content_critic_node",
        critic_route,
        {
            "revision_node_needed": "revision_node",
            "need_image_also": "image_generation_node",
            "end": END
        }
    )

    # After revision → back to market for fresh research
    graph.add_edge("revision_node", "marketing_node")

    # Image always ends
    graph.add_edge("image_generation_node", END)

    app = graph.compile()
    print("✅ SoloForge graph compiled successfully!")
    print(app.get_graph().draw_ascii())
    return app
