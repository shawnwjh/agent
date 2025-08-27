# logic/agents/paper-critique-agent/main.py
import logging, os
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from .agent import PaperCritiqueAgent

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Models ----------
class PaperContent(BaseModel):
    title: str
    abstract: str
    content: Optional[str] = ""
    authors: Optional[str] = ""
    venue: Optional[str] = ""

class CritiqueRequest(BaseModel):
    paper: PaperContent
    focus_areas: Optional[List[str]] = ["methodology", "novelty", "clarity", "significance"]
    session_id: Optional[str] = "default"

class CritiqueResponse(BaseModel):
    response: str
    session_id: str

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

# ---------- Globals ----------
critique_agent: Optional[PaperCritiqueAgent] = None  # keep ONE definition
init_error: Optional[str] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agent on startup, but NEVER crash the server."""
    global critique_agent, init_error
    try:
        api_key = os.getenv("NVIDIA_API_KEY")
        base_url = os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
        if not api_key:
            raise ValueError("NVIDIA_API_KEY environment variable is required")
        critique_agent = PaperCritiqueAgent(nvidia_api_key=api_key, nvidia_base_url=base_url)
        logger.info("Paper Critique Agent initialized successfully")
    except Exception as e:
        init_error = str(e)
        critique_agent = None
        logger.exception("Agent init failed; service will start, endpoints may 503")
        # DO NOT re-raise
    yield
    if critique_agent:
        await critique_agent.cleanup()

app = FastAPI(
    title="Paper Critique Agent",
    description="LangChain-based agent for academic paper critique and conference prediction",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _ensure_agent():
    if not critique_agent:
        raise HTTPException(status_code=503, detail=f"Agent not ready: {init_error or 'initializing'}")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    _ensure_agent()
    try:
        response = await critique_agent.chat(  # type: ignore
            message=request.message,
            session_id=request.session_id,
            context=request.context,
        )
        return ChatResponse(response=response, session_id=request.session_id)
    except Exception as e:
        logger.exception("Chat request failed")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")), reload=True)