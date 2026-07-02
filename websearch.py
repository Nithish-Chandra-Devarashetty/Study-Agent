"""
websearch.py — the "go find it" half of the agent's decision.

The other tools all read from the user's OWN notes. This module is the ONE place
the agent is allowed to reach outside them: a DuckDuckGo web search it can fall
back to when the notes clearly don't cover a question. Keeping it in its own file
mirrors `ingest.py` (the notes half) — the agent picks between the two.

Design choices:
  - DuckDuckGo via the `ddgs` package: free, no API key, no account — the same
    "no key required" spirit as the local sentence-transformers embeddings.
  - Everything degrades gracefully. If `ddgs` isn't installed or the network is
    down, we return an empty list (never raise), so the agent loop keeps working
    and the tool can report an honest "couldn't reach the web" instead of
    crashing.

The results are deliberately shaped like retrieval hits (title / snippet / url)
so the `search_web` tool in agent.py can summarise them the same way the
notes-based tools summarise Chroma chunks.
"""

from typing import List, Dict

import config


def _import_ddgs():
    """Return a DDGS class from whichever package name is installed, or None.

    The library was renamed from `duckduckgo_search` to `ddgs`; we try the new
    name first, fall back to the old one, and return None if neither is present
    so the caller can degrade gracefully instead of crashing.
    """
    try:
        from ddgs import DDGS  # current package name
        return DDGS
    except ImportError:
        pass
    try:
        from duckduckgo_search import DDGS  # older package name
        return DDGS
    except ImportError:
        return None


def web_search(query: str, k: int = None) -> List[Dict[str, str]]:
    """Search the public web and return up to k lightweight result dicts.

    Each dict has: {"title", "snippet", "url"}. Returns an EMPTY list on any
    problem (library missing, no network, no results) — callers treat empty as
    "the web fallback is unavailable / found nothing" and answer honestly.
    """
    if k is None:
        k = config.WEB_SEARCH_RESULTS

    DDGS = _import_ddgs()
    if DDGS is None:
        return []

    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=k))
    except Exception:
        # Network error, rate limit, upstream change — never let it crash the
        # agent loop; an empty list becomes an honest "couldn't reach the web".
        return []

    results: List[Dict[str, str]] = []
    for r in raw:
        # ddgs uses "body" for the snippet and "href" for the link; older
        # versions used "snippet"/"url". Support both.
        results.append({
            "title": (r.get("title") or "").strip(),
            "snippet": (r.get("body") or r.get("snippet") or "").strip(),
            "url": (r.get("href") or r.get("url") or "").strip(),
        })
    return results


def web_search_available() -> bool:
    """True if a DuckDuckGo backend is importable (library installed)."""
    return _import_ddgs() is not None
