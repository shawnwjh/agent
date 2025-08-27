# logic/agents/paper-critique-agent/agent.py
import logging
import asyncio
from typing import List, Optional, Dict
from dotenv import load_dotenv

# LangChain imports
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationSummaryBufferMemory
from langchain.tools import Tool

load_dotenv()
logger = logging.getLogger(__name__)

class PaperCritiqueAgent:
    """LangChain-based agent for paper critique and prediction"""
    
    def __init__(self, nvidia_api_key: str, nvidia_base_url: str):
        self.service_client = ServiceClient()
        self.llm = NvidiaLLM(api_key=nvidia_api_key, base_url=nvidia_base_url)
        self.sessions = {}  # Store memory sessions
        self.external_tools = ExternalKnowledgeTools(llm=self.llm)  # Pass LLM for term extraction
        
        # Enhanced tools with smart search
        self.tools = [
            Tool(
                name="search_related_papers",
                description="Search for papers related to a given topic using smart keyword extraction. Input should be the research topic or paper details you want to find related work for.",
                func=self._search_related_papers_sync
            ),
            Tool(
                name="extract_key_terms",
                description="Extract key search terms from paper title and abstract. Input should be 'title: X, abstract: Y'.",
                func=self._extract_key_terms_sync
            ),
            Tool(
                name="analyze_paper_quality",
                description="Analyze the technical quality and methodology of a paper. Input should be paper content.",
                func=self._analyze_quality_sync
            )
        ]
    
    def _get_or_create_agent(self, session_id: str):
        """Get or create agent with memory for session"""
        if session_id not in self.sessions:
            memory = ConversationSummaryBufferMemory(
                llm=self.llm,
                max_token_limit=2000,
                return_messages=True,
                memory_key="chat_history",
                output_key="output"
            )
            
            agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                memory=memory,
                verbose=True,
                max_iterations=5,
                handle_parsing_errors=True
            )
            
            self.sessions[session_id] = {
                "agent": agent,
                "memory": memory,
                "created_at": asyncio.get_event_loop().time()
            }
        
        return self.sessions[session_id]["agent"]
    
    # Updated sync wrappers with smart search
    def _search_related_papers_sync(self, query: str) -> str:
        """Smart paper search that extracts key terms first"""
        return self.external_tools.search_external_papers(query)
    
    def _extract_key_terms_sync(self, paper_info: str) -> str:
        """Extract key terms from paper information"""
        try:
            # Parse input like "title: X, abstract: Y"
            parts = paper_info.split(', abstract:')
            title = parts[0].replace('title:', '').strip()
            abstract = parts[1].strip() if len(parts) > 1 else ""
            
            terms = self.external_tools.extract_search_terms(title, abstract)
            return f"Extracted key search terms: {terms}"
        except:
            return f"Could not parse paper info. Expected format: 'title: X, abstract: Y'"
    
    def _analyze_quality_sync(self, paper_content: str) -> str:
        """Sync wrapper for analyzing paper quality"""
        return self._analyze_quality(paper_content)
    
    # Updated async implementations
    async def _search_internal_database(self, query: str) -> str:
        """Search internal vector database first"""
        try:
            # Try internal search first
            embeddings = await self.service_client.generate_embeddings([query])
            query_vector = embeddings[0]
            
            similar_papers = await self.service_client.search_similar_papers(query_vector, top_k=5)
            
            if not similar_papers:
                return "No papers found in internal database. Recommend using external search."
            
            # Format internal results
            results = []
            for paper in similar_papers:
                metadata = paper.get("metadata", {})
                results.append(f"Internal Paper: {metadata.get('title', 'Unknown')}\nAbstract: {metadata.get('abstract', '')[:200]}...")
            
            return f"Found {len(results)} papers in internal database:\n\n" + "\n\n".join(results)
            
        except Exception as e:
            logger.error(f"Internal database search failed: {e}")
            return f"Internal database search failed: {str(e)}. Recommend using external search."
    
    def _analyze_quality(self, paper_content: str) -> str:
        """Analyze paper quality (placeholder - could be enhanced with specific metrics)"""
        content_length = len(paper_content)
        
        quality_indicators = []
        if content_length > 5000:
            quality_indicators.append("Substantial content length")
        if "methodology" in paper_content.lower():
            quality_indicators.append("Contains methodology section")
        if "evaluation" in paper_content.lower() or "experiment" in paper_content.lower():
            quality_indicators.append("Contains evaluation/experimental content")
        if "related work" in paper_content.lower() or "literature" in paper_content.lower():
            quality_indicators.append("Contains related work/literature review")
        
        return f"Paper quality analysis:\n- Content length: {content_length} characters\n- Quality indicators: {', '.join(quality_indicators) if quality_indicators else 'Basic content structure'}"
    
    async def critique_paper(self, paper_data: Dict[str, str], focus_areas: List[str], session_id: str) -> str:
        """Generate comprehensive paper critique"""
        agent = self._get_or_create_agent(session_id)
        
        # Prepare paper content for analysis
        paper_text = f"Title: {paper_data['title']}\n\nAbstract: {paper_data['abstract']}\n\nContent: {paper_data.get('content', '')}"
        
        # Enhanced critique prompt that guides tool usage
        critique_prompt = f"""
        Provide a comprehensive critique of this research paper focusing on: {', '.join(focus_areas)}.
        
        Paper Information:
        {paper_text}
        
        Research Strategy:
        1. First search internal database for related work
        2. If internal database is empty, search external sources (ArXiv, Semantic Scholar) 
        3. Analyze the paper's contribution in context of found literature
        4. Provide structured critique with:
           - Overall assessment and score (0-10)
           - Analysis of each focus area
           - Strengths and weaknesses  
           - Comparison to related work
           - Conference/venue prediction with reasoning
        
        Use the available tools strategically to gather relevant literature before critiquing.
        """
        
        # Get agent response
        response = agent.run(critique_prompt)
        return response
    
    async def chat(self, message: str, session_id: str, context: Optional[Dict] = None) -> str:
        """Handle conversational chat about papers, including uploaded documents"""
        import os
        from pypdf import PdfReader
        agent = self._get_or_create_agent(session_id)
        document_texts = []

        # Parse and read documents if provided
        if context and "documents" in context:
            for doc_url in context["documents"]:
                filename = doc_url.split("/")[-1]
                file_path = os.path.join("frontend", "public", "uploads", filename)
                try:
                    with open(file_path, "rb") as f:
                        reader = PdfReader(f)
                        text = ""
                        for page in reader.pages:
                            text += page.extract_text() or ""
                        document_texts.append(text)
                except Exception as e:
                    logger.error(f"Failed to read {file_path}: {e}")

        # Add document text to context
        if document_texts:
            context_str = f"Documents provided:\n" + "\n---\n".join(document_texts)
        else:
            context_str = str(context) if context else ""

        enhanced_message = f"{context_str}\n\nUser message: {message}"
        response = agent.run(enhanced_message)
        return response
    
    def clear_session(self, session_id: str) -> bool:
        """Clear memory for a specific session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def get_sessions(self) -> List[Dict]:
        """Get information about active sessions"""
        sessions = []
        for session_id, session_data in self.sessions.items():
            sessions.append({
                "session_id": session_id,
                "created_at": session_data["created_at"],
                "memory_length": len(session_data["memory"].chat_memory.messages)
            })
        return sessions
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.service_client.close()