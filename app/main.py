# logic/agents/paper-critique-agent/main.py
import logging, os
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from agent import PaperCritiqueAgent

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

class HealthResponse(BaseModel):
    status: str
    services_available: Dict[str, bool]
    agent_ready: bool
    memory_sessions: int
    init_error: Optional[str] = None

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

# ---------- Health ----------
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/health", response_model=HealthResponse)
async def health():
    services = {
        "embedding_service": False,
        "vector_db_service": False,
        "reranker_service": False,
        "nvidia_llm": False,
    }
    try:
        if critique_agent:
            services["nvidia_llm"] = getattr(critique_agent, "llm", None) is not None
            sc = getattr(critique_agent, "service_client", None)
            if sc and getattr(sc, "client", None):
                try:
                    r = await sc.client.get(f"{sc.embedding_url}/health")
                    services["embedding_service"] = (r.status_code == 200)
                except: pass
                try:
                    r = await sc.client.get(f"{sc.vector_db_url}/health")
                    services["vector_db_service"] = (r.status_code == 200)
                except: pass
                try:
                    r = await sc.client.get(f"{sc.reranker_url}/health")
                    services["reranker_service"] = (r.status_code == 200)
                except: pass
    except Exception:
        logger.exception("health failed (non-fatal)")
    return HealthResponse(
        status="healthy" if all(services.values()) else "partial",
        services_available=services,
        agent_ready=critique_agent is not None,
        memory_sessions=len(getattr(critique_agent, "sessions", {})) if critique_agent else 0,
        init_error=init_error,
    )

# ---------- Endpoints ----------
@app.get("/")
def root():
    return {"service": "agent", "status": "up"}

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

# Optional helper so GET /chat doesn’t 405 during quick tests
@app.get("/chat")
def chat_probe():
    return {"ok": True, "hint": "Use POST /chat with JSON body"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")), reload=True)