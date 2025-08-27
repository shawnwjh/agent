# logic/agents/paper-critique-agent/main.py
import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from agent import PaperCritiqueAgent

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API endpoints
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

# Global agent instance
critique_agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global critique_agent
    
    # Initialize agent
    api_key = os.getenv("NVIDIA_API_KEY")
    base_url = os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    
    if not api_key:
        raise ValueError("NVIDIA_API_KEY environment variable is required")
    
    critique_agent = PaperCritiqueAgent(
        nvidia_api_key=api_key,
        nvidia_base_url=base_url
    )
    
    logger.info("Paper Critique Agent initialized successfully")
    yield
    
    # Shutdown
    if critique_agent:
        await critique_agent.cleanup()

app = FastAPI(
    title="Paper Critique Agent",
    description="LangChain-based agent for academic paper critique and conference prediction",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

critique_agent = None
init_error = None

@app.get("/healthz")
def healthz():
    # liveness-only
    return {"status": "ok"}

@app.get("/health")
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
        # never crash the health route
        pass

    return {
        "status": "healthy" if all(services.values()) else "partial",
        "services_available": services,
        "agent_ready": critique_agent is not None,
        "memory_sessions": len(getattr(critique_agent, "sessions", {})) if critique_agent else 0,
    }

@app.post("/critique", response_model=CritiqueResponse)
async def critique_paper_endpoint(request: CritiqueRequest):
    """Generate paper critique and conference prediction"""
    try:
        paper_data = {
            "title": request.paper.title,
            "abstract": request.paper.abstract,
            "content": request.paper.content,
            "authors": request.paper.authors,
            "venue": request.paper.venue
        }
        
        response = await critique_agent.critique_paper(
            paper_data=paper_data,
            focus_areas=request.focus_areas,
            session_id=request.session_id
        )
        
        return CritiqueResponse(
            response=response,
            session_id=request.session_id
        )
        
    except Exception as e:
        logger.error(f"Critique request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat with the research assistant"""
    try:
        response = await critique_agent.chat(
            message=request.message,
            session_id=request.session_id,
            context=request.context
        )
        
        return ChatResponse(
            response=response,
            session_id=request.session_id
        )
        
    except Exception as e:
        logger.error(f"Chat request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear memory for a specific session"""
    success = critique_agent.clear_session(session_id)
    if success:
        return {"success": True, "message": f"Session {session_id} cleared"}
    return {"success": False, "message": f"Session {session_id} not found"}

@app.get("/sessions")
async def list_sessions():
    """List active sessions"""
    if not critique_agent:
        return {"sessions": []}
    
    sessions = critique_agent.get_sessions()
    return {"sessions": sessions}

@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "Paper Critique Agent",
        "version": "1.0.0",
        "description": "LangChain-based agent for academic paper analysis",
        "features": [
            "Paper critique generation",
            "Conference prediction", 
            "Conversational chat with memory",
            "Tool integration (embedding, vector search, reranking)"
        ],
        "endpoints": {
            "health": "/health",
            "critique": "POST /critique",
            "chat": "POST /chat",
            "sessions": "GET /sessions",
            "clear_session": "DELETE /sessions/{session_id}",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        reload=True
    )