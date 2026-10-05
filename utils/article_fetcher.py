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
import re

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

# What's left after cleaning must be at least this long to count as an
# article; shorter means trafilatura only found site chrome.
MIN_BODY_CHARS = 120

# Site chrome trafilatura picked up instead of the article (2026-10-03 mail:
# every KCHOL Bigpara item's "summary" was the Hürriyet footer). A line
# containing any of these is dropped.
_CHROME_MARKERS = (
    "en çok arananlar",
    "copyright",
    "kullanım koşulları",
    "gizlilik politika",
    "login olduğunuz",
    "sıralamayı değiştirmek",
    "bist 100 dolar euro",
    "bildirimler bildirimler",
)

# Everything from here on is the broker/portal legal disclaimer
# ("Burada yer alan yatırım bilgi, yorum ve tavsiyeleri...").
_DISCLAIMER_START = re.compile(r"burada yer alan yatırım bilgi|yatırım danışmanlığı kapsamında değildir", re.IGNORECASE)


def clean_article_text(text: str) -> str | None:
    """Drops site-chrome lines and the trailing legal disclaimer; None if
    nothing article-like is left."""
    match = _DISCLAIMER_START.search(text)
    if match:
        text = text[: match.start()]
    lines = [
        line.strip() for line in text.splitlines()
        if line.strip() and not any(marker in line.lower() for marker in _CHROME_MARKERS)
    ]
    cleaned = "\n".join(lines).replace("*", "").strip()
    return cleaned if len(cleaned) >= MIN_BODY_CHARS else None


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
            logger.info("Google News URL decode failed for %s: %s", url, exc)
            return None
        if not decoded.get("status") or not decoded.get("decoded_url"):
            logger.info(
                "Google News URL decode returned no URL for %s: %s", url, decoded.get("message")
            )
            return None
        target_url = decoded["decoded_url"]

    try:
        response = await scraper._get(target_url)
    except Exception as exc:
        logger.info("Article fetch failed for %s: %s", target_url, exc)
        return None

    try:
        extracted = trafilatura.extract(
            response.text, url=target_url, include_comments=False, include_tables=False
        )
    except Exception as exc:
        logger.info("trafilatura extraction failed for %s: %s", target_url, exc)
        return None

    if not extracted:
        logger.info("trafilatura extracted no text from %s (HTTP %s)", target_url, response.status_code)
        return None

    cleaned = clean_article_text(extracted)
    if cleaned is None:
        logger.info("Only site chrome/disclaimer extracted from %s", target_url)
        return None
    return cleaned[:MAX_BODY_SNIPPET_CHARS]
