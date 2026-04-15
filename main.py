from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from datetime import datetime
import json
import os
from agent import build_graph

load_dotenv()
soloforge = build_graph()
# ── Import your agent ──
# Uncomment when agent.py is ready:


app = FastAPI(title="SoloForge AI")

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Serve static files ──
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Build agent once at startup ──
soloforge = build_graph()
print("✅ SoloForge agent ready!")

# ════════════════════════════════
# JSON HELPERS
# ════════════════════════════════

def save_brand(brand_data: dict):
    """Save brand to brands.json — creates file if not exists"""
    try:
        with open("brands.json", "r") as f:
            brands = json.load(f)
    except FileNotFoundError:
        brands = {}

    brands[brand_data["brand_name"]] = brand_data

    with open("brands.json", "w") as f:
        json.dump(brands, f, indent=2)

def load_brand(brand_name: str):
    """Load brand from brands.json — returns None if not found"""
    try:
        with open("brands.json", "r") as f:
            brands = json.load(f)
        return brands.get(brand_name)
    except FileNotFoundError:
        return None

def save_content(data: dict):
    """Append approved content to generated_content.json"""
    try:
        with open("generated_content.json", "r") as f:
            history = json.load(f)
    except FileNotFoundError:
        history = []

    history.append(data)

    with open("generated_content.json", "w") as f:
        json.dump(history, f, indent=2)

# ════════════════════════════════
# SCHEMAS
# ════════════════════════════════

class OnboardRequest(BaseModel):
    brand_name: str
    brand_tone: str
    target_audience: str
    current_offer: str
    industry: str
    competitors: str
    tagline: str

class GenerateRequest(BaseModel):
    brand_name: str
    user_query: str

class ApproveRequest(BaseModel):
    brand_name: str
    decision: str           # "yes" or "no"
    feedback: str = ""
    captions: str = ""
    image_url: str = ""
    market_insights: str = ""
    critic_score: int = 0

# ════════════════════════════════
# ENDPOINTS
# ════════════════════════════════

@app.get("/")
def home():
    with open("static/index.html") as f:
        return HTMLResponse(f.read())

@app.get("/dashboard")
def dashboard():
    with open("static/dashboard.html") as f:
        return HTMLResponse(f.read())

# ── 1. Onboard ──
@app.post("/onboard")
def onboard(request: OnboardRequest):
    try:
        save_brand(request.dict())
        return {
            "status": "success",
            "message": f"{request.brand_name} onboarded!"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# ── 2. Generate ──
@app.post("/generate")
def generate(request: GenerateRequest):

    # Load brand
    brand = load_brand(request.brand_name)
    if not brand:
        return {
            "status": "error",
            "message": "Brand not found. Please complete onboarding first."
        }

    try:
        # Run agent
        result = soloforge.invoke({
                "brand_name":      brand.get("brand_name", ""),
                "brand_tone":      brand.get("brand_tone", ""),
                "target_audience": brand.get("target_audience", ""),
                "current_offer":   brand.get("current_offer", ""),
                "industry":        brand.get("industry", ""),
                "competitors":     brand.get("competitors", ""),
                "tagline":         brand.get("tagline", ""),
                "brand_context":   "",
                "user_query":      request.user_query,
                "query_valid":     False,
                "need_content":    False,
                "need_image":      False,
                "need_both":       False,
                "need_social":     False,
                "market_insights": "",
                "captions":        "",
                "strategy":        "",
                "image_url":       "",      # ← add this
                "image_prompt":    "",      # ← add this
                "critic_score":    0,
                "critic_feedback": "",
                "revision_count":  0,
                "revision_needed": False,
                "next_worker":     "",
                "human_approved":  False,   # ← matches fixed state
                "messages":        []
            })
                
        return {
            "status":          "success",
            "captions":        result.get("captions", ""),
            "image_url":       result.get("image_url", ""),
            "market_insights": result.get("market_insights", ""),
            "critic_score":    result.get("critic_score", 0)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# ── 3. Approve ──
@app.post("/approve")
def approve(request: ApproveRequest):
    if request.decision == "yes":
        save_content({
            "brand_name":      request.brand_name,
            "generated_at":    str(datetime.now()),
            "captions":        request.captions,
            "image_url":       request.image_url,
            "market_insights": request.market_insights,
            "critic_score":    request.critic_score
        })
        return {
            "status": "saved",
            "message": "Content approved and saved!"
        }
    else:
        return {
            "status": "regenerate",
            "message": "Regenerating with feedback...",
            "updated_query": request.feedback
        }

# ── 4. Health ──
@app.get("/health")
def health():
    return {"status": "SoloForge running ✅"}

# ── 5. History ──
@app.get("/history/{brand_name}")
def history(brand_name: str):
    try:
        with open("generated_content.json", "r") as f:
            all_content = json.load(f)
        brand_history = [
            c for c in all_content
            if c["brand_name"] == brand_name
        ]
        return {"status": "success", "history": brand_history}
    except FileNotFoundError:
        return {"status": "success", "history": []}
