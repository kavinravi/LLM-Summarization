"""Lightweight web-search wrapper for LLM tool calls.

Primary backend: Brave Search API (2 000 queries/month free).
Fallback backend: DuckDuckGo instant-answer scrape via ``duckduckgo_search``.

Usage:
    from utils.search_tool import web_search
    snippets = web_search("average wind capacity factor nevada")
"""
from __future__ import annotations

import os, json, textwrap, logging, time
from typing import List

import requests

try:
    from duckduckgo_search import DDGS  # type: ignore
except ImportError:
    DDGS = None  # Graceful fallback if package missing

logger = logging.getLogger(__name__)

def get_env_var(key: str, default: str = "") -> str:
    """Get environment variable from Streamlit secrets or .env file."""
    try:
        import streamlit as st
        return st.secrets[key]
    except:
        return os.getenv(key, default)

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
BRAVE_KEY = get_env_var("BRAVE_SEARCH_API_KEY", "")
HEADERS = {
    "Accept": "application/json",
    "Accept-Encoding": "gzip",
    "X-Subscription-Token": BRAVE_KEY,
}

# Rate limiting variables
_last_request_time = 0
_request_delay = 2.0  # 2 seconds between requests


def _brave_search(query: str, max_results: int = 5) -> List[str]:
    """Query Brave Search API and return a list of snippet strings."""
    global _last_request_time
    
    if not BRAVE_KEY:
        raise RuntimeError("BRAVE_SEARCH_API_KEY not set in environment")

    # Rate limiting: wait if we made a request too recently
    current_time = time.time()
    time_since_last = current_time - _last_request_time
    if time_since_last < _request_delay:
        sleep_time = _request_delay - time_since_last
        logger.info(f"Rate limiting: sleeping {sleep_time:.1f}s before search")
        time.sleep(sleep_time)
    
    _last_request_time = time.time()

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

    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
        snippets = []
        for r in results:
            snippet = f"{r.get('title', '')}: {r.get('body', '')} ({r.get('href', '')})"
            snippets.append(snippet)
        logger.info(f"DuckDuckGo search returned {len(snippets)} results for: {query}")
        return snippets
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed for query '{query}': {e}")
        return []


def web_search(query: str, max_results: int = 5) -> str:
    """Run web search and return a plain-text block summarising results."""
    # Try Brave with exponential backoff for rate limits
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            snippets = _brave_search(query, max_results=max_results)
            source = "Brave"
            break
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429 and attempt < max_retries:
                # Rate limited - wait longer and retry
                backoff_time = (2 ** attempt) * 3  # 3s, 6s, 12s
                logger.info(f"Rate limited, backing off {backoff_time}s (attempt {attempt + 1})")
                time.sleep(backoff_time)
                continue
            else:
                logger.warning("Brave search failed – falling back to DuckDuckGo (%s)", e)
                snippets = _ddg_search(query, max_results=max_results)
                source = "DuckDuckGo"
                break
        except Exception as e:
            logger.warning("Brave search failed – falling back to DuckDuckGo (%s)", e)
            snippets = _ddg_search(query, max_results=max_results)
            source = "DuckDuckGo"
            break

    if not snippets:
        return f"No web results found for query: {query}"

    bullet_list = "\n".join(f"• {s}" for s in snippets)
    return textwrap.dedent(
        f"""
        Web search results via {source} for: {query}
        {bullet_list}
        """
    ).strip() 