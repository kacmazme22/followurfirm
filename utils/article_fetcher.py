"""
Fetches a real article body (not just a headline) for a scraped item, via
trafilatura content extraction — used by google_news_scraper.py and
bigpara_scraper.py to enrich a bounded number of items per ticker
(MAX_ARTICLES_TO_ENRICH_PER_TICKER, config/constants.py) with genuine
multi-sentence content instead of a bare title. Diagnosed 2026-08-13: LLM
synthesis was producing generic filler sentences because Bigpara/Google
News never supplied a real body_snippet, only raw_title — KAP was the only
source with real summary text.

Google News RSS links are opaque redirect tokens
(news.google.com/rss/articles/...) that no longer resolve via a plain HTTP
redirect (verified live) — they require decoding via `googlenewsdecoder`,
which reverse-engineers Google's internal batchexecute endpoint. Same risk
class as KAP's undocumented API: could break silently if Google changes the
token format. Bigpara URLs are already real article URLs, no decode needed.

Never raises: any failure (decode, fetch, or empty extraction) returns
None, so callers can safely keep body_snippet=None and fall back to
title-only rendering, exactly like before this module existed.
"""

from __future__ import annotations

import asyncio
import logging

import trafilatura

from scrapers.base import BaseScraper

logger = logging.getLogger(__name__)

# Imported defensively: googlenewsdecoder is a small reverse-engineering
# package with fragile transitive deps (2026-10-04: selectolax 1.0 removed a
# backend it imports, and the module-level ImportError killed main.py before
# a single ticker ran). Without it we only lose Google News body enrichment —
# headlines still flow — so a broken install must never take the digest down.
try:
    from googlenewsdecoder import gnewsdecoder
except Exception as _import_exc:  # ImportError, or anything its own imports raise
    gnewsdecoder = None
    logger.warning("googlenewsdecoder yüklenemedi, Google News gövde zenginleştirmesi kapalı: %s", _import_exc)

# NewsItem.body_snippet (scrapers/models.py) caps at max_length=2000 —
# trimmed well under that so truncation never collides with the field's own
# validation limit.
MAX_BODY_SNIPPET_CHARS = 1500


async def fetch_article_body(scraper: BaseScraper, url: str, is_google_news: bool) -> str | None:
    """Fetches `url` (decoding it first if it's a Google News redirect
    token) and extracts the real article body via trafilatura. Uses
    `scraper._get()` so the fetch gets the same politeness delay, User-Agent,
    and in-run caching as every other request that scraper makes. Returns
    None on any failure at any stage — decode, fetch, or empty extraction —
    never raises."""
    target_url = url

    if is_google_news:
        if gnewsdecoder is None:
            return None
        try:
            decoded = await asyncio.to_thread(gnewsdecoder, url, interval=1)
        except Exception as exc:
            logger.debug("Google News URL decode failed for %s: %s", url, exc)
            return None
        if not decoded.get("status") or not decoded.get("decoded_url"):
            logger.debug(
                "Google News URL decode returned no URL for %s: %s", url, decoded.get("message")
            )
            return None
        target_url = decoded["decoded_url"]

    try:
        response = await scraper._get(target_url)
    except Exception as exc:
        logger.debug("Article fetch failed for %s: %s", target_url, exc)
        return None

    try:
        extracted = trafilatura.extract(
            response.text, url=target_url, include_comments=False, include_tables=False
        )
    except Exception as exc:
        logger.debug("trafilatura extraction failed for %s: %s", target_url, exc)
        return None

    if not extracted:
        return None

    return extracted[:MAX_BODY_SNIPPET_CHARS]
