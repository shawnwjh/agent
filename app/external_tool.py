# logic/agents/paper-critique-agent/external_tools.py
import arxiv
import requests
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ExternalKnowledgeTools:
    """Tools for retrieving external academic knowledge"""
    
    def __init__(self):
        pass
    
    def search_arxiv(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search ArXiv for papers"""
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
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
    
    def search_semantic_scholar(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
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
                    "published_date": str(item.get('year', 'Unknown')),
                    "citation_count": item.get('citationCount', 0)
                }
                papers.append(paper)
            
            return papers
        except Exception as e:
            logger.error(f"Semantic Scholar search error: {e}")
            return []
    
    def search_external_papers(self, query: str, max_results: int = 10) -> str:
        """Combined external paper search - for LangChain tool"""
        try:
            # Search both sources
            arxiv_papers = self.search_arxiv(query, max_results//2)
            semantic_papers = self.search_semantic_scholar(query, max_results//2)
            
            all_papers = arxiv_papers + semantic_papers
            
            if not all_papers:
                return f"No external papers found for query: '{query}'"
            
            # Format results for agent consumption
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
URL: {paper['url']}
"""
                if 'citation_count' in paper:
                    result += f"Citations: {paper['citation_count']}\n"
                
                results.append(result)
            
            return f"Found {len(all_papers)} external papers:\n\n" + "\n---\n".join(results)
            
        except Exception as e:
            logger.error(f"External search failed: {e}")
            return f"Error searching external papers: {str(e)}"

# Updated agent.py with smart tool selection
class EnhancedPaperCritiqueAgent:
    """Enhanced agent with external knowledge capabilities"""
    
    def __init__(self, nvidia_api_key: str, nvidia_base_url: str):
        # ... existing initialization ...
        self.external_tools = ExternalKnowledgeTools()
        
        # Enhanced tool set
        self.tools = [
            Tool(
                name="search_external_papers",
                description="Search ArXiv and Semantic Scholar for recent papers on a topic. Use when you need current literature or when internal database has no results.",
                func=self._search_external_papers_sync
            ),
            Tool(
                name="search_internal_database", 
                description="Search internal vector database for similar papers. Use first to check if we have relevant papers stored.",
                func=self._search_internal_database_sync
            ),
            Tool(
                name="analyze_paper_context",
                description="Analyze how a paper fits within current research trends and literature. Input should be paper content.",
                func=self._analyze_context_sync
            )
        ]
    
    def _search_external_papers_sync(self, query: str) -> str:
        """Sync wrapper for external paper search"""
        return self.external_tools.search_external_papers(query)
    
    def _search_internal_database_sync(self, query: str) -> str:
        """Sync wrapper for internal database search"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._search_internal_database(query))
    
    def _analyze_context_sync(self, paper_content: str) -> str:
        """Analyze paper in context of current literature"""
        # This could use external search to find related work
        # and compare the paper against current trends
        return f"Contextual analysis of paper content (length: {len(paper_content)} chars)"
    
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

# Enhanced critique logic
async def critique_paper_with_smart_search(self, paper_data: Dict[str, str], focus_areas: List[str], session_id: str) -> str:
    """Enhanced critique that intelligently uses available knowledge sources"""
    agent = self._get_or_create_agent(session_id)
    
    paper_text = f"Title: {paper_data['title']}\n\nAbstract: {paper_data['abstract']}\n\nContent: {paper_data.get('content', '')}"
    
    # Enhanced prompt that guides tool usage
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
    
    response = agent.run(critique_prompt)
    return response