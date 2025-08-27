# logic/agents/paper-critique-agent/agent.py
import logging
import os
import asyncio
from typing import List, Optional, Dict, Any
import httpx
from dotenv import load_dotenv
import arxiv
import requests

# LangChain imports
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationSummaryBufferMemory
from langchain.tools import Tool
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

load_dotenv()
logger = logging.getLogger(__name__)

class ExternalKnowledgeTools:
    """Tools for retrieving external academic knowledge"""
    
    def __init__(self, llm=None):
        self.llm = llm  # Pass LLM for key term extraction
    
    def extract_search_terms(self, paper_title: str, paper_abstract: str = "", focus_areas: List[str] = None) -> str:
        """Extract key search terms from paper content using LLM"""
        try:
            content = f"Title: {paper_title}"
            if paper_abstract:
                content += f"\nAbstract: {paper_abstract}"
            
            focus_context = ""
            if focus_areas:
                focus_context = f"\nFocus areas of interest: {', '.join(focus_areas)}"
            
            extraction_prompt = f"""
            Extract 3-5 key search terms from this academic paper that would be effective for finding related literature.
            
            {content}{focus_context}
            
            Guidelines:
            - Use specific technical terms, not generic words
            - Include methodology names if mentioned
            - Focus on the main contribution/topic
            - Return only the search terms, comma-separated
            - No explanations, just the terms
            
            Example: "transformer architecture, attention mechanism, neural machine translation"
            """
            
            if self.llm:
                import requests
                response = requests.post(
                    f"{self.llm.base_url}/chat/completions",
                    json={
                        "model": self.llm.model,
                        "messages": [{"role": "user", "content": extraction_prompt}],
                        "max_tokens": 100,
                        "temperature": 0.1
                    },
                    headers={
                        "Authorization": f"Bearer {self.llm.api_key}",
                        "Content-Type": "application/json"
                    },
                    timeout=30
                )
                if response.status_code == 200:
                    result = response.json()
                    terms = result["choices"][0]["message"]["content"].strip()
                    # Clean up the terms
                    terms = terms.replace('"', '').replace("'", "")
                    return terms
            
            # Fallback: simple keyword extraction
            keywords = []
            text = f"{paper_title} {paper_abstract}".lower()
            
            # Common academic keywords to look for
            important_terms = [
                "transformer", "attention", "neural", "deep learning", "machine learning",
                "natural language", "computer vision", "reinforcement learning", "clustering",
                "classification", "regression", "optimization", "algorithm", "model",
                "network", "embedding", "representation", "generation", "detection"
            ]
            
            for term in important_terms:
                if term in text:
                    keywords.append(term)
            
            # Add title words (filtered)
            title_words = [word for word in paper_title.lower().split() 
                          if len(word) > 3 and word not in ['the', 'and', 'for', 'with']]
            keywords.extend(title_words[:3])
            
            return ", ".join(keywords[:5]) if keywords else paper_title
            
        except Exception as e:
            logger.error(f"Term extraction failed: {e}")
            # Fallback to title
            return paper_title
    
    def search_external_papers(self, query: str, max_results: int = 10) -> str:
        """Combined external paper search with cleaned query"""
        try:
            # Clean the query input - remove LangChain artifacts
            clean_query = query.split('\n')[0].strip()
            clean_query = clean_query.split('Observation:')[0].strip()
            clean_query = clean_query.replace('Action Input:', '').strip()
            
            # If query is too long or looks contaminated, extract key terms
            if len(clean_query) > 100 or 'Observation' in clean_query:
                # Try to extract key terms from the contaminated query
                words = clean_query.split()
                # Take meaningful words, skip common ones
                meaningful_words = [w for w in words[:10] 
                                  if len(w) > 3 and w.lower() not in ['observation', 'will', 'search', 'papers']]
                clean_query = ' '.join(meaningful_words[:5])
            
            logger.info(f"Searching with cleaned query: '{clean_query}'")
            
            # Search ArXiv
            arxiv_papers = self._search_arxiv(clean_query, max_results//2)
            # Search Semantic Scholar  
            semantic_papers = self._search_semantic_scholar(clean_query, max_results//2)
            
            all_papers = arxiv_papers + semantic_papers
            
            if not all_papers:
                return f"No external papers found for query: '{clean_query}'. Try different search terms."
            
            # Format results
            results = []
            for i, paper in enumerate(all_papers, 1):
                authors_str = ', '.join(paper['authors'][:3])
                if len(paper['authors']) > 3:
                    authors_str += '...'
                
                result = f"""Paper {i}:
Title: {paper['title']}
Authors: {authors_str}
Source: {paper['source']} ({paper['published_date']})
Abstract: {paper['abstract'][:300]}{'...' if len(paper['abstract']) > 300 else ''}
URL: {paper['url']}"""
                
                results.append(result)
            
            return f"Found {len(all_papers)} external papers for '{clean_query}':\n\n" + "\n---\n".join(results)
            
        except Exception as e:
            logger.error(f"External search failed: {e}")
            return f"Error searching external papers: {str(e)}"
    
    def _search_arxiv(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search ArXiv for papers"""
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers = []
            for result in search.results():
                paper = {
                    "title": result.title,
                    "abstract": result.summary,
                    "authors": [str(author) for author in result.authors],
                    "url": result.pdf_url,
                    "source": "ArXiv",
                    "published_date": str(result.published.date())
                }
                papers.append(paper)
            
            return papers
        except Exception as e:
            logger.error(f"ArXiv search error: {e}")
            return []
    
    def _search_semantic_scholar(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search Semantic Scholar for papers"""
        try:
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                'query': query,
                'limit': max_results,
                'fields': 'title,abstract,authors,url,year,citationCount'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            papers = []
            for item in data.get('data', []):
                abstract = item.get('abstract', 'Abstract not available')
                if not abstract:
                    abstract = 'Abstract not available'
                
                paper = {
                    "title": item.get('title', 'Title not available'),
                    "abstract": abstract,
                    "authors": [author.get('name', 'Unknown') for author in item.get('authors', [])],
                    "url": item.get('url', ''),
                    "source": "Semantic Scholar",
                    "published_date": str(item.get('year', 'Unknown'))
                }
                papers.append(paper)
            
            return papers
        except Exception as e:
            logger.error(f"Semantic Scholar search error: {e}")
            return []

# Custom NVIDIA LLM wrapper for LangChain
class NvidiaLLM(LLM):
    """Custom LLM wrapper for NVIDIA API to work with LangChain"""
    
    api_key: str = ""
    base_url: str = ""
    model: str = "meta/llama-3.1-70b-instruct"
    client: Optional[httpx.AsyncClient] = None
    
    def __init__(self, api_key: str, base_url: str, model: str = "meta/llama-3.1-70b-instruct"):
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = None
        
    async def _acall(self, prompt: str, stop: Optional[List[str]] = None, 
                     run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs) -> str:
        """Async call to NVIDIA API"""
        if not self.client:
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
        
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048,
                "temperature": 0.1,
                "stream": False
            }
            
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"NVIDIA LLM call failed: {e}")
            raise Exception(f"LLM generation failed: {str(e)}")
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, 
              run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs) -> str:
        """Synchronous call to NVIDIA API"""
        import requests
        
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048,
                "temperature": 0.1,
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"NVIDIA LLM call failed: {e}")
            raise Exception(f"LLM generation failed: {str(e)}")
    
    @property
    def _llm_type(self) -> str:
        return "nvidia_api"

class ServiceClient:
    """HTTP client for communicating with core services"""
    
    def __init__(self):
        self.embedding_url = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8002")
        self.vector_db_url = os.getenv("VECTOR_DB_SERVICE_URL", "http://localhost:8001")
        self.reranker_url = os.getenv("RERANKER_SERVICE_URL", "http://localhost:8003")
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        response = await self.client.post(
            f"{self.embedding_url}/embeddings",
            json={"texts": texts}
        )
        response.raise_for_status()
        result = response.json()
        return result["embeddings"]
    
    async def search_similar_papers(self, query_vector: List[float], top_k: int = 10) -> List[Dict]:
        """Search for similar papers in vector database"""
        response = await self.client.post(
            f"{self.vector_db_url}/search",
            json={
                "query_vector": query_vector,
                "top_k": top_k
            }
        )
        response.raise_for_status()
        result = response.json()
        return result["results"]
    
    async def rerank_documents(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """Rerank documents based on relevance"""
        # Convert search results to reranker format
        reranker_docs = []
        for doc in documents:
            metadata = doc.get("metadata", {})
            reranker_docs.append({
                "title": metadata.get("title", ""),
                "abstract": metadata.get("abstract", ""),
                "authors": metadata.get("authors", ""),
                "source": metadata.get("source", ""),
                "url": metadata.get("url", ""),
                "published_date": metadata.get("published_date", "")
            })
        
        response = await self.client.post(
            f"{self.reranker_url}/rerank",
            json={
                "query": query,
                "documents": reranker_docs,
                "top_k": top_k
            }
        )
        response.raise_for_status()
        result = response.json()
        return result["reranked_documents"]
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()

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