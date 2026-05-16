"""
backend search.py
Responsible for fetching academic papers from the arXiv API.
"""
import arxiv
from typing import List, Dict, Any

def fetch_papers(query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    """
    Search arXiv for papers matching the query.
    Returns a list of paper metadata dicts.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers=[]
    for result in client.results(search):
        papers.append({
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "abstract": result.summary,
            "published": result.published.strftime("%Y-%m-%d"),
            "categories": result.categories,
        })
    return papers
