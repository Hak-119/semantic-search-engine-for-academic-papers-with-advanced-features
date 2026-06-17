"""
backend/search.py
Responsible for fetching academic papers from the arXiv API.
Includes rate-limit handling (HTTP 429) with retry/backoff.
"""

import arxiv
import time
from typing import List, Dict, Any


def fetch_papers(query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    """
    Search arXiv for papers matching the query.
    Returns a list of paper metadata dicts.

    Includes built-in delay + retries to handle arXiv's rate limiting (HTTP 429).
    If arXiv is still rate-limiting after retries, raises a clear RuntimeError
    instead of a raw traceback.
    """
    client = arxiv.Client(
        page_size=max_results,
        delay_seconds=5,   # wait 5s between paginated requests
        num_retries=2       # retry twice before giving up
    )
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []
    try:
        for result in client.results(search):
            papers.append({
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "abstract": result.summary,
                "published": result.published.strftime("%Y-%m-%d"),
                "url": result.entry_id,
                "categories": result.categories,
            })
    except arxiv.HTTPError as e:
        raise RuntimeError(
            "arXiv is currently rate-limiting requests (HTTP 429/503). "
            "Please wait a minute and try again."
        ) from e

    return papers