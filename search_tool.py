"""Lightweight web-search wrapper for DeepSeek tool calls.

Primary backend: Brave Search API (2 000 queries/month free).
Fallback backend: DuckDuckGo instant-answer scrape via ``duckduckgo_search``.

Usage:
    from search_tool import web_search
    snippets = web_search("average wind capacity factor nevada")
"""
from __future__ import annotations

import os, json, textwrap, logging
from typing import List

import requests

try:
    from duckduckgo_search import DDGS  # type: ignore
except ImportError:
    DDGS = None  # Graceful fallback if package missing

logger = logging.getLogger(__name__)

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
BRAVE_KEY = os.getenv("BRAVE_SEARCH_API_KEY", "")
HEADERS = {
    "Accept": "application/json",
    "Accept-Encoding": "gzip",
    "X-Subscription-Token": BRAVE_KEY,
}


def _brave_search(query: str, max_results: int = 5) -> List[str]:
    """Query Brave Search API and return a list of snippet strings."""
    if not BRAVE_KEY:
        raise RuntimeError("BRAVE_SEARCH_API_KEY not set in environment")

    params = {"q": query, "count": max_results}
    resp = requests.get(BRAVE_ENDPOINT, headers=HEADERS, params=params, timeout=15)
    resp.raise_for_status()

    data = resp.json()
    items = data.get("results", {}).get("items", [])
    snippets = []
    for item in items[:max_results]:
        title = item.get("title", "")
        desc = item.get("description", "")
        url = item.get("url", "")
        snippet = f"{title}: {desc} ({url})"
        snippets.append(snippet)
    return snippets


def _ddg_search(query: str, max_results: int = 5) -> List[str]:
    """Fallback DuckDuckGo search if Brave quota exhausted or key missing."""
    if DDGS is None:
        raise RuntimeError("duckduckgo_search package not installed")

    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)
    snippets = []
    for r in results:
        snippet = f"{r.get('title', '')}: {r.get('body', '')} ({r.get('href', '')})"
        snippets.append(snippet)
    return snippets


def web_search(query: str, max_results: int = 5) -> str:
    """Run web search and return a plain-text block summarising results."""
    try:
        snippets = _brave_search(query, max_results=max_results)
        source = "Brave"
    except Exception as e:
        logger.warning("Brave search failed – falling back to DuckDuckGo (%s)", e)
        snippets = _ddg_search(query, max_results=max_results)
        source = "DuckDuckGo"

    if not snippets:
        return f"No web results found for query: {query}"

    bullet_list = "\n".join(f"• {s}" for s in snippets)
    return textwrap.dedent(
        f"""
        Web search results via {source} for: {query}
        {bullet_list}
        """
    ).strip() 